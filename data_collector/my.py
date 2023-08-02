import numpy as np
import robosuite as suite
from robosuite.Mymodel.push_controller import get_push_control
from robosuite.Mymodel.path import find_path
from scipy.spatial.transform import Rotation
import random
import d3rlpy

def calculate_pos(target):
    pos=[0,0,0]
    pos[0]=-0.2+target[0]*0.1+0.05
    pos[1]=-0.3+target[1]*0.1+0.05
    return pos

def w_to_grid(w):
    grid=np.zeros(shape=(4,6))
    for i in range(24):
        if w[i]==2:
            grid[i//6][i%6]=1
    start_pos=[w[24],w[25]]
    grid[w[26]][w[27]]=1
    grid[w[28]][w[29]]=1
    return grid,start_pos

def generate_w():
    w = []
    for i in range(24):
        w.append(random.randint(1, 2))
    w.append(random.randint(0, 3))
    w.append(random.randint(0, 5))
    w.append(random.randint(0, 3))
    w.append(random.randint(0, 5))
    w.append(random.randint(0, 3))
    w.append(random.randint(0, 5))
    w.append(0.3)
    return w



# create environment instance
controller_configs = suite.load_controller_config(default_controller='OSC_POSE')

env = suite.make(
    env_name="My", # try with other tasks like "Stack" and "Door"
    robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    controller_configs= controller_configs,
)

# reset the environment
# w=[2,2,0,0,0,2,1,2,1,2,1,1,2,2,2,0,0,1,1,1,1,1,0,1,0,1,3,1,1,3,0.4]
# 1000 steps of observations with shape of (100,)
observations = np.empty((0,3))

# 1000 steps of actions with shape of (4,)
actions = np.empty((0,7))

# 1000 steps of rewards
rewards = np.empty((0,1))

# 1000 steps of terminal flags
terminals = np.empty((0,1))

epis=0
while epis<100:
    # w=[1,1,2,1,1,1,1,1,2,1,2,1,2,2,1,1,2,1,2,1,1,1,1,1,3,1,1,1,3,5,0.3]
    w=generate_w()
    if(w[w[24]*6+w[25]]==2 or w[w[26]*6+w[27]]==2 or w[w[28]*6+w[29]]==2):
        continue
    grid,start_pos=w_to_grid(w)
    points=find_path(grid,start_pos,[1,5])
    # print(points)
    if(points is None):
        continue
    obs=env.reset(w)
    done_skill=False
    # points=[[3,3],[2,3],[2,5]]
    # points=[[3,3]]
    i=0
    desired_goal=calculate_pos(points[i])
    print(obs)
    count=0
    done=False
    num=0
    while not (i==len(points)-1 and done_skill):
        if(num==1000):
            break
        if(done_skill):
            if(count==5):
                count=0
                if(i<len(points)-1):
                    i+=1
                desired_goal=calculate_pos(points[i])
            elif(count<5):
                count+=1
        action,done_skill = get_push_control(obs,desired_goal)
        # action = np.random.randn(4) # sample random action
        # print(action)
        # gripper_angl=obs['robot0_eef_quat']
        # rotation = Rotation.from_quat(gripper_angl)
        # euler = rotation.as_euler('xyz')
        # x_angle = euler[0]
        # y_angle = euler[1]
        # z_angle = euler[2]
        # x_angle_deg = np.degrees(x_angle)
        # y_angle_deg = np.degrees(y_angle)
        # z_angle_deg = np.degrees(z_angle)
        # print("x angle",x_angle_deg)
        # print("y angle",y_angle_deg)
        # print("z angle",z_angle_deg)
        # if z_angle_deg<90:
        #     action= [0,0,0,0,0,0.05,1]
            # i+=1
            # print(obs)
        # elif (i==3):
        #     action= [0,0,0,0,0,(np.pi/2-1.5),0]
        #     i+=1
        # else:
        #     action=[0,0,0,0,0,0,0]
        next_obs, reward, done, info = env.step(action)  # take action in the environment
        num+=1
        # print(obs['cubeA_pos'])
        observations=np.append(observations,np.expand_dims(obs['cubeA_pos'],axis=0),axis=0)
        actions=np.append(actions,np.expand_dims(action,axis=0),axis=0 )
        rewards=np.append(rewards,reward)
        terminals=np.append(terminals,done)
        obs=next_obs
        # print(obs['cube_pos'])
        # env.render()  # render on display
    if(done_skill):
        epis+=1
    print(epis)
    print(observations.shape)
    
dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
)
    
