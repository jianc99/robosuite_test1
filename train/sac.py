'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2023-01-27 19:30:48
LastEditTime: 2023-02-04 12:27:23
Description: 
'''

import numpy as np
import torch

from tianshou.policy import SACPolicy
from .network import MLPEncoder, CNNEncoder
from tianshou.utils.net.continuous import ActorProb, Critic


def get_sac_agent(env, args):
    if args.image_obs:
        h, w, c = env.get_env_attr('img_shape')[0]
        joint_dim = env.get_env_attr('joint_dim')[0]
    else:
        observation_space = env.get_env_attr('observation_space')[0]
        state_shape = observation_space.shape or observation_space.n
    
    action_space = env.get_env_attr('action_space')[0]
    action_shape = action_space.shape or action_space.n
    max_action = action_space.high[0]

    # build model
    if args.image_obs:
        cat_hidden_sizes = [int(args.hidden_sizes[-1]/2)]
        net_a = CNNEncoder(
            c, h, w, joint_dim, 
            joint_hidden_sizes=args.hidden_sizes, 
            cat_hidden_sizes=cat_hidden_sizes, 
            device=args.device,
            encoder_type=args.encoder_type
        )
        net_c1 = CNNEncoder(
            c, h, w, joint_dim, 
            action_shape=action_shape, 
            joint_hidden_sizes=args.hidden_sizes, 
            cat_hidden_sizes=cat_hidden_sizes, 
            device=args.device,
            encoder_type=args.encoder_type
        )
        net_c2 = CNNEncoder(
            c, h, w, joint_dim, 
            action_shape=action_shape, 
            joint_hidden_sizes=args.hidden_sizes, 
            cat_hidden_sizes=cat_hidden_sizes, 
            device=args.device,
            encoder_type=args.encoder_type
        )
    else:
        net_a = MLPEncoder(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
        net_c1 = MLPEncoder(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
        net_c2 = MLPEncoder(state_shape, action_shape, hidden_sizes=args.hidden_sizes, concat=True, device=args.device)

    # build actor and critic
    actor = ActorProb(net_a, action_shape, max_action=max_action, device=args.device, unbounded=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)

    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=action_space
    )
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)
    return policy
