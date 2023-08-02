import argparse
import os
import pprint
import datetime

import pickle
import wandb
import torch
import yaml

import robosuite as robosuite
from robosuite.wrappers import GymWrapper
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
# from tianshou.utils import MyOwnWandBLogger, DummyLogger
from .sac import get_sac_agent
# from causaldro.utils import load_config

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="Lift", help='PickPlaceCausal, LiftCausal, StackCausal')
parser.add_argument('--wandb', type=bool, default=False, help='use wandb or not')
parser.add_argument('--wandb-dir', type=str, default="./")
parser.add_argument('--train_spurious', type=str, default="xpr", help='[xpr, xnr] for LiftCausal and [lrrb, lbrr] for StackCausal')
parser.add_argument('--test_spurious', type=str, default="xnr")
parser.add_argument('--algo-name', type=str, default="sac")
parser.add_argument('--trail', type=int, default=10)
parser.add_argument('--training-num', type=int, default=10)
parser.add_argument('--test-num', type=int, default=10)
parser.add_argument('--render', type=float, default=0.)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--resume-path', type=str, default=None)
args = parser.parse_args()


# select dro type
# dro_dict = {
#     'noise_type': 'full',
#     'sigma': 0.01,
#     'bound': 0.01,
# }
# if dro_dict['noise_type'] == 'gaussian':
#     dro_arg_name = 'g' + str(dro_dict['sigma'])
# elif dro_dict['noise_type'] == 'uniform':
#     dro_arg_name = 'u' + str(dro_dict['bound'])
# elif dro_dict['noise_type'] == 'none':
#     dro_arg_name = 'na'
# elif dro_dict['noise_type'] == 'causal':
#     dro_arg_name = 'ca'
# elif dro_dict['noise_type'] == 'full':
#     dro_arg_name = 'fu'
# else:
#     raise NotImplementedError()

# select agent and load parameters for agent
agent = {
    'sac': get_sac_agent,
    # 'ppo': get_ppo_agent,
}
# cfg_path = {
#     'sac': {
#         'LiftCausal': './config/sac_LiftCausal.yaml',
#         'StackCausal': './config/sac_StackCausal.yaml',
#         'PickPlaceCausal': './config/sac_PickPlaceCausal.yaml',
#     },
#     'ppo': {
#         'LiftCausal': './config/ppo_LiftCausal.yaml',
#         'StackCausal': './config/ppo_StackCausal.yaml',
#         'PickPlaceCausal': './config/ppo_PickPlaceCausal.yaml',
#     }
# }
# cfg = load_config(cfg_path[args.algo_name][args.task])
cfg = load_config('/Users/jian/.local/lib/python3.8/site-packages/robosuite/Mymodel/config/sac_lift.yaml')
args = argparse.Namespace(**vars(args), **cfg)

print(args)

# select trainer
if args.algo_name in ['sac']:
    trainer = offpolicy_trainer
elif args.algo_name in ['ppo']:
    trainer = onpolicy_trainer
else:
    raise NotImplementedError()

# create environments
env_param = {
    'robots': "Panda",                                           # use Kinova Gen3 robot
    'use_camera_obs': True if args.image_obs else False,           # use pixel observations or not
    'use_object_obs': False if args.image_obs else True,           # use state observations or not
    'has_offscreen_renderer': True if args.image_obs else False,   # not needed since not using pixel obs
    'has_renderer': False,                                         # make sure we can render to the screen
    'reward_shaping': True,                                        # use dense rewards
    'control_freq': 20,                                            # control should happen fast enough so that simulation looks smooth
    'horizon': args.horizon,                                       # length of episode
    'render_camera': args.camera,
    'camera_names': args.camera,
    'camera_heights': args.height,
    'camera_widths': args.width,
}
group_name = os.path.join(
    args.task, 
    args.algo_name,
    args.encoder_type if args.image_obs else 'mlp',
    # args.train_spurious,  # type of spuriousness for training
    # args.test_spurious,   # type of spuriousness for testing
    # dro_arg_name,         # type of noise for DRO
    'h' + str(args.horizon),
    'b' + str(args.batch_size),
    'io' + str(int(args.image_obs)),
    'bs' + str(args.buffer_size),
    'ns' + str(args.n_step),
    'cw' + str(args.width),
    'alr' + str(args.actor_lr),
    'clr' + str(args.critic_lr),
)
# env_param['spurious_type'] = args.train_spurious
# train_envs = SubprocVectorEnv([lambda: GymWrapper(robosuite.make(args.task, **env_param), dro_dict=dro_dict) for _ in range(args.training_num)])
# env_param['spurious_type'] = args.test_spurious
# test_envs = SubprocVectorEnv([lambda: GymWrapper(robosuite.make(args.task, **env_param)) for _ in range(args.test_num)])
# env_param['spurious_type'] = args.train_spurious
train_envs = DummyVectorEnv([lambda: GymWrapper(robosuite.make(args.task, **env_param)) for _ in range(args.training_num)])
# env_param['spurious_type'] = args.test_spurious
test_envs = DummyVectorEnv([lambda: GymWrapper(robosuite.make(args.task, **env_param)) for _ in range(args.test_num)])
print('Finish environment building')

# start training
train_seed_list = [111, 222, 333, 444, 555, 666, 777, 888, 999, 9999, 99999]
test_seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
for t_i in range(args.trail):
    # if seed is not set, all environments share the same seed
    train_envs.seed(train_seed_list[t_i])
    test_envs.seed(test_seed_list[t_i])

    # policy
    policy = agent[args.algo_name](train_envs, args)
    print('Finish policy building')

    # collector
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    print('Finish collector building')

    # logger
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # log_name = os.path.join(group_name, now)
    # if args.wandb:
    #     logger = MyOwnWandBLogger()
    #     run = wandb.init(
    #         project='causal-poisoning', 
    #         entity='wenhaoding', 
    #         group=group_name.replace(os.path.sep, "_"),
    #         name=log_name.replace(os.path.sep, "_"), 
    #         reinit=True, 
    #         config=args,
    #         dir=args.wandb_dir
    #     ) 
    # else:
    #     logger = DummyLogger()

    # trainer
    result = trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        test_in_train=False,
        # logger=logger,
    )
    pprint.pprint(result)

    # save buffer data
    #buffer_filename = './buffer/' + group_name.replace(os.path.sep, "_") + '.pkl'
    #pickle.dump(train_collector.buffer, open(buffer_filename, 'wb'))

    # if args.wandb:
    #     run.finish()