import numpy as np
import torch
from torch import nn
from tianshou.utils.net.common import MLP
import matplotlib.pyplot as plt


class MLPEncoder(nn.Module):
    def __init__(
        self,
        state_shape,
        action_shape=0,
        hidden_sizes=(),
        norm_layer=None,
        norm_args=None,
        activation=nn.ReLU,
        act_args=None,
        device="cpu",
        concat=False,
    ):
        super().__init__()
        self.device = device
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        if concat:
            input_dim += action_dim
        # the output dim is hidden dim
        self.model = MLP(
            input_dim=input_dim, 
            hidden_sizes=hidden_sizes, 
            norm_layer=norm_layer, 
            norm_args=norm_args, 
            activation=activation,
            act_args=act_args,
            device=device, 
        )
        self.output_dim = self.model.output_dim

    def forward(self, obs, state=None, info={}):
        """ Mapping: obs -> flatten (inside MLP)-> logits. """
        logits = self.model(obs)
        return logits, state


class CNNEncoder(nn.Module):
    def __init__(self, c, h, w, joint_dim, action_shape=0, joint_hidden_sizes=(), cat_hidden_sizes=(), device="cpu", encoder_type='mlp'):
        super().__init__()
        self.img_dim = int(c*h*w)
        self.joint_dim = joint_dim
        self.device = device
        self.c = c
        self.h = h
        self.w = w
        self.action_dim = int(np.prod(action_shape))
        self.encoder_type = encoder_type

        if self.encoder_type == 'r3m':
            assert self.h == self.w == 224, "R3M encoder requires image size [224, 224]"
            r3m = load_r3m("resnet18") # resnet18, resnet34
            r3m.eval()
            r3m.to(device)
            # freeze all parameters
            self.img_net = nn.Sequential(r3m, nn.Flatten())
            for param in self.img_net.parameters(): 
                param.requires_grad = False
            self.img_scale = 255.0  # r3m expects value [0-255]
        else:
            # build the image network
            self.img_net = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=8, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(inplace=True), 
                nn.Flatten()
            )
            self.img_scale = 1.0
        
        # get the output dimension of CNN
        with torch.no_grad():
            cnn_output_dim = np.prod(self.img_net(torch.zeros(1, c, h, w)).shape[1:])
            print('cnn_output_dim:', cnn_output_dim)

        # build the joint network (also put the action into it if we have)
        def build_mlp(hidden_sizes):
            modules = []
            for h_i in range(len(hidden_sizes)-1):
                modules.append(nn.Linear(hidden_sizes[h_i], hidden_sizes[h_i+1]))
                modules.append(nn.ReLU(inplace=True))
            return nn.Sequential(*modules[:-1])

        joint_hidden_sizes = [joint_dim + self.action_dim] + joint_hidden_sizes
        self.joint_net = build_mlp(joint_hidden_sizes)

        # build the concat network
        cat_hidden_sizes = [joint_hidden_sizes[-1] + cnn_output_dim] + cat_hidden_sizes
        self.cat_net = build_mlp(cat_hidden_sizes)

        # output dim is the last size of cat_hidden_sizes
        self.output_dim = cat_hidden_sizes[-1]

    def forward(self, obs, state=None, info={}):
        """ Mapping: [img + joint (+ action)] -> logits. """
        assert obs.shape[1] == self.img_dim + self.joint_dim + self.action_dim, 'obs dimension is not correct'
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        img_hwc = obs[:, 0:self.img_dim].reshape(-1, self.h, self.w, self.c) # [B, H, W, C]
        img_chw = img_hwc.transpose(1, 3) * self.img_scale # [B, C, H, W]
        joint_action = obs[:, self.img_dim:]
        img_feature = self.img_net(img_chw)
        joint_feature = self.joint_net(joint_action)

        # debug
        # one_img = img_hwc[0].detach().cpu().numpy()
        # print(one_img)
        # plt.imshow(one_img)
        # plt.savefig('./one_img.png', dpi=200)
        # plt.close('all')
        # input('saved')

        concat_feature = torch.cat([img_feature, joint_feature], dim=1)
        output = self.cat_net(concat_feature)
        return output, state
