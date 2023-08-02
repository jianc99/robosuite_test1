'''
Author: Jian Chen
Email: jianc2@andrew.cmu.edu
Date: 2023-07-29 20:21:01
LastEditTime: 2023-07-30 15:01:10
Description: 
'''
from robosuite.utils.transform_utils import quat2mat
from scipy.spatial.transform import Rotation
import numpy as np

DEBUG = False


def euler_to_so3(roll, pitch, yaw):
    # Convert Euler angles (in degrees) to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Create a rotation object from Euler angles
    r = Rotation.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad], degrees=False)

    # Get the rotation matrix (SO(3))
    rotation_matrix = r.as_matrix()

    return rotation_matrix

def get_move_action(obs, target_position, atol=1e-3, gain=10., close_gripper=False):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    # current_position = observation['observation'][:3]
    current_position = obs['robot0_eef_pos']

    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        gripper_action = -1.0
    else:
        gripper_action = 1.0
    action = np.hstack((action,np.array([0,0,0]),gripper_action))

    return action

def get_rotation_control(obs, target):
    gripper_angl=obs['robot0_eef_quat']
    rotation = Rotation.from_quat(gripper_angl)
    euler = rotation.as_euler('xyz')
    quat2mat
    z_angle = euler[2]



def get_push_control(obs, target, atol=1e-4, block_width=0.11, workspace_height=0.1, block_idx=0):

    #gripper_position = obs['observation'][:3]
    #block_position = obs['observation'][3:6]
    #goal = obs['desired_goal']

    # gripper_position = obs['observation'][:3]
    gripper_position = obs['robot0_eef_pos']
    # gripper_angl=obs['robot0_eef_quat']
    # rotation = Rotation.from_quat(gripper_angl)
    # euler = rotation.as_euler('xyz')
    # z_angle = euler[2]
    # z_angle_deg = np.degrees(z_angle)
    # print("z angle",z_angle_deg)

    # b_idx = block_idx*9 + 10
    # block_position = obs['observation'][b_idx:b_idx+3]
    block_position = obs['cubeA_pos'][:3]
    g_idx = block_idx*3
    goal = target[:3]
    goal[2]=block_position[2]

    desired_block_angle = np.arctan2(goal[0] - block_position[0], goal[1] - block_position[1])
    abs_block_angle=np.abs(desired_block_angle)
    target_ori=0
    if(abs_block_angle<np.pi/4 or abs_block_angle>np.pi*3/4):
        target_ori=np.pi/2
    else:
        target_ori=0
    
    gripper_angle = np.arctan2(goal[0] - gripper_position[0], goal[1] - gripper_position[1])

    push_position = block_position.copy()
    push_position[0] += -1. * np.sin(desired_block_angle) * block_width / 2.
    push_position[1] += -1. * np.cos(desired_block_angle) * block_width / 2.
    push_position[2] += 0.005

    # If the block is already at the place position, do nothing
    if np.sum(np.subtract(block_position[:2], goal[:2])**2) < atol:
        if DEBUG:
            print("The block is already at the place position; do nothing")
        target_pos=push_position
        target_pos[0] += -1. * np.sin(desired_block_angle) * block_width / 2.
        target_pos[1] += -1. * np.cos(desired_block_angle) * block_width / 2.
        target_pos[2]+=workspace_height
        return get_move_action(obs, target_pos, atol=atol, gain=5.0), True
        # return np.array([0., 0., 0., 0., 0., 0., 0.]), True

    # Angle between gripper and goal vs block and goal is roughly the same
    angle_diff = abs((desired_block_angle - gripper_angle + np.pi) % (2*np.pi) - np.pi)

    gripper_sq_distance = (gripper_position[0] - goal[0])**2 + (gripper_position[1] - goal[1])**2
    block_sq_distance = (block_position[0] - goal[0])**2 + (block_position[1] - goal[1])**2

    if (gripper_position[2] - push_position[2])**2 < atol and angle_diff < np.pi/4 and block_sq_distance < gripper_sq_distance:

        # Push towards the goal
        target_position = goal
        target_position[2] = gripper_position[2]
        if DEBUG:
            print("Push")
        return get_move_action(obs, target_position, atol=atol, gain=5.0), False

    # If the gripper is above the push position
    if (gripper_position[0] - push_position[0])**2 + (gripper_position[1] - push_position[1])**2 < 10*atol:

        # Move down to prepare for push
        gripper_angl=obs['robot0_eef_quat']
        rotation = Rotation.from_quat(gripper_angl)
        euler = rotation.as_euler('xyz')
        z_angle = euler[2]
        y_angle=euler[1]
        x_angle = euler[0]
        if DEBUG:
            print("Move down to prepare for push")
        # if np.abs(z_angle-target_ori)<0.01 and np.abs(y_angle)<0.1 and np.pi-np.abs(x_angle)<0.1:
        if np.abs(z_angle-target_ori)<0.01 and np.abs(y_angle)<0.1:
            return get_move_action(obs, push_position, atol=atol), False
        if y_angle>0.1:
            return [0,0,0,0,-0.5*y_angle,0,1],False
        if y_angle<-0.1:
            return [0,0,0,0,-0.5*y_angle,0,1],False
        # if x_angle<0 and x_angle>-np.pi+0.1:
        #     return [0,0,0,0.5*(-np.pi-x_angle),0,0,1],False
        # if x_angle>0 and x_angle<np.pi-0.1:
        #     return [0,0,0,0.5*(np.pi-x_angle),0,0,1],False
        if target_ori==np.pi/2 and z_angle<target_ori:
            return [0,0,0,0,0,0.5*(target_ori-z_angle),1],False
        elif target_ori==0 and z_angle>target_ori:
            return [0,0,0,0,0,0.5*(target_ori-z_angle),1], False

        # if DEBUG:
        #     print("Move down to prepare for push")
        # return get_move_action(obs, push_position, atol=atol), False


    # Else move the gripper to above the push
    target_position = push_position.copy()
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the push position")

    return get_move_action(obs, target_position, atol=atol), False


