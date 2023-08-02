'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2023-04-17 20:21:01
LastEditTime: 2023-08-01 14:53:53
Description: 
'''

import numpy as np
from scipy.spatial.transform import Rotation


CLOSE = 1.0
OPEN = -1.0
KEEP = 0.0

DT = 1 / 20 # 20Hz control frequency

def get_rotation_action(cur, target):
    if(np.abs(cur[0])>0.1):
        return [0,0,0,-0.5*cur[0],0,0,KEEP],False
    if(np.abs(cur[1])>0.1):
        return [0,0,0,0,-0.5*cur[1],0,KEEP],False
    return [0,0,0,0,0,0.5*(target[2]-cur[2]),KEEP],False


def block_is_grasped(gripper_position, gripper_state, relative_grasp_position, pick_position, pick_contact, atol=1e-3):
    has_block = block_inside_grippers(gripper_position, relative_grasp_position, pick_position, atol=atol)
    return has_block and grippers_are_closed(gripper_state) and pick_contact


def block_inside_grippers(gripper_position, relative_grasp_position, pick_position, atol=1e-3):
    relative_position = np.subtract(gripper_position, pick_position)
    return np.sum(np.subtract(relative_position, relative_grasp_position)**2) < atol


def grippers_are_closed(gripper_state):
    if gripper_state > 0.35:
        return True
    else:
        return False


def grippers_are_open(gripper_state):
    if gripper_state < 0.2:
        return True
    else:
        return False


class PickPlaceController():
    def __init__(self, Kp=10, Kd=1, atol=1e-3):
        self.Kp = Kp
        self.Kd = Kd
        self.atol = atol
        self.error_pre = np.zeros(3)

    def controller(self, gripper_position, target_position, gripper=KEEP, Kp=10., Kd=10.):
        error = np.subtract(target_position, gripper_position)
        action = Kp * error + Kd * (error - self.error_pre) / DT
        action = np.hstack((action,[0,0,0], gripper))
        self.error_pre = error
        return action

    def step(self, obs, relative_grasp_position=(0., 0., -0.03), move_height=0.1, DEBUG=False):
        """
            relative_grasp_position: distance between gripper clip and end-effector center
            pick_height: the height above the block to pick and move
            place_height: the height above the place position to place the block
        """
        gripper_position = obs['gripper_position']  # [3,]
        gripper_state = obs['gripper_state']        # [1,]
        gripper_ori = obs['gripper_quat']           # [4,]
        pick_ori = obs['pick_quat']                 # [4,]
        pick_position = obs['pick_position']        # [3,]
        place_position = obs['place_position']      # [3,]
        pick_contact = obs['pick_contact']          # [4,]
        place_atol = 2e-4

        ### The priority is gradually decreasing from top to bottom ###

        # If the block is already at the place position
        if np.sum(np.subtract(pick_position, place_position)**2) < place_atol and grippers_are_open(gripper_state):
            target_height = pick_position[2] + move_height - relative_grasp_position[2]
            if np.abs(target_height - gripper_position[2]) < self.atol:
                if DEBUG:
                    print("Do nothing")
                return np.array([0., 0., 0., 0., 0., 0., KEEP]), True
            
            # the gripper should return to a high position
            if DEBUG:
                print("The block is at the place position, move up the gripper")
            target_position = np.array([gripper_position[0], gripper_position[1], target_height])
            return self.controller(gripper_position, target_position, gripper=KEEP, Kp=self.Kp*2, Kd=0.0), False

        # If the gripper is already grasping the block
        if block_is_grasped(gripper_position, gripper_state, relative_grasp_position, pick_position, pick_contact, atol=self.atol):
            # If the gripper is hover the place position
            xy_dist = (gripper_position[0] - place_position[0])**2 + (gripper_position[1] - place_position[1])**2
            if xy_dist < place_atol:
                if np.sum(np.subtract(pick_position, place_position)**2) < place_atol:
                    if not grippers_are_open(gripper_state):
                        if DEBUG:
                            print("Open gripper to release the object")
                        return np.array([0., 0., 0, 0., 0., 0., OPEN]), False
                    else:
                        return np.array([0., 0., 0., 0., 0., 0., KEEP]), False

                if DEBUG:
                    print("Move down to place the object", pick_position, place_position)
                return self.controller(pick_position, place_position, gripper=KEEP, Kp=self.Kp, Kd=self.Kd), False
            # If the gripper has not reached above the place position
            else:
                # Move to the place position while keeping the gripper closed
                target_position = place_position + relative_grasp_position
                target_position[2] += move_height
                if DEBUG:
                    print("Move to above the place position")
                return self.controller(gripper_position, target_position, gripper=KEEP, Kp=self.Kp, Kd=self.Kd), False

        # If the block is ready to be grasped (the center of gripper is close to the block)
        if block_inside_grippers(gripper_position, relative_grasp_position, pick_position, atol=self.atol):
            # Close the grippers
            if DEBUG:
                print("Close the grippers")
            if not grippers_are_closed(gripper_state):
                return np.array([0., 0., 0.,0., 0., 0., CLOSE]), False
            else:
                return np.array([0., 0., 0.,0., 0., 0., KEEP]), False

        # If the gripper is hover the pick position
        if (gripper_position[0] - pick_position[0])**2 + (gripper_position[1] - pick_position[1])**2 < self.atol:
            # If the grippers are closed, open them
            if not grippers_are_open(gripper_state):
                if DEBUG:
                    print("Open the grippers")
                return np.array([0., 0., 0.,0., 0., 0., OPEN]), False
            gripper_angle=Rotation.from_quat(gripper_ori).as_euler('xyz')
            object_angle=Rotation.from_quat(pick_ori).as_euler('xyz')
            if np.abs(gripper_angle[2]-object_angle[2])>0.01:
                return get_rotation_action(gripper_angle,object_angle)
            

            # Move down to grasp
            target_position = pick_position + relative_grasp_position
            if DEBUG:
                print("Move down to grasp")
            return self.controller(gripper_position, target_position, gripper=KEEP, Kp=self.Kp, Kd=self.Kd), False

        # Else move the gripper to above the block
        target_position = pick_position + relative_grasp_position
        target_position[2] += move_height
        if DEBUG:
            print("Move to above the block")
        return self.controller(gripper_position, target_position, gripper=KEEP, Kp=self.Kp, Kd=self.Kd), False
