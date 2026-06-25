#%%

import sys
import os
import matplotlib.pyplot as plt

import torch, random
import numpy as np

os.chdir(r'/home/ted/Desktop/curious_maze_test')

from utils import args
from live_view import LiveView

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'encoders'))
from action_encoder import Encode_Action 
from obs_encoder import Encode_Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'decoders'))
from action_decoder import Decode_Action
from obs_decoder import Decode_Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from maze_runner import Maze_Runner

from general_FEP_RL.agent import Agent
from general_FEP_RL.plot_training_log import plot_training_log

# Set random seed.
np.random.seed(args.init_seed)
random.seed(args.init_seed)
torch.manual_seed(args.init_seed)
torch.cuda.manual_seed(args.init_seed)



observation_dict = {
    'see_image' : {
        'encoder' : Encode_Image,
        'encoder_arg_dict' : {                
            'encode_size' : 64,
            'zp_zq_sizes' : [64]},
        'decoder' : Decode_Image,
        'decoder_arg_dict' : {},
        'upsilon_obs' : 10,                               
        'beta_obs' : 1,                                 
        'eta_before_clamp' : 1,
        'eta' : 2}}

action_dict = {
    'make_velocity' : {
        'encoder' : Encode_Action,
        'encoder_arg_dict' : {                
            'encode_size' : 64,
            'zp_zq_sizes' : [64]},
        'decoder' : Decode_Action,
        'decoder_arg_dict' : {},
        'target_entropy' : 1,
        'alpha_normal' : 1,
        'lr_alpha' : .01,
        'initial_alpha' : .5,
        'delta' : 0}}



# Make agent, train agent. 
agent = Agent(
    observation_dict = observation_dict,       
    action_dict = action_dict,       
    hidden_state_sizes = [256],
    time_scales = [1],
    beta_hidden = [],
    eta_before_clamp = [],
    eta = [],
    upsilon_reward = .1,
    number_of_critics = 2, 
    tau = .1,
    lr = .001,
    weight_decay = .00001,
    gamma = .99,
    capacity = 128, 
    max_steps = 32,
    #verbose = True
    )

view = LiveView(image_size=8)



for i, maze_name in enumerate(args.maze_list):
    maze_runner = Maze_Runner(maze_name = maze_name, args = args)
    for e in range(args.epochs[i]):
        print(f'epoch {e}')
        agent.begin()
        push_list = []
        for s in range(args.max_steps):
            image, speed = maze_runner.obs()
            image = image.unsqueeze(1)
            obs = {'see_image' : image}
            step_dict = agent.step_in_episode(obs)
            action = step_dict['action']['make_velocity']
            yaw = action[0][0][0].item()
            spe = action[0][0][1].item()
            reward, wall_punishment, which, end, action_name = maze_runner.action(yaw, spe)
            push_list.append(
                {'observation_dict' : obs, 
                'action_dict' : step_dict['action'], 
                'reward' : reward + wall_punishment, 
                'done' : end, 
                'best_action_dict' : None,
                'step_dict' : step_dict})
            if s > 0:
                real_image = obs['see_image'].squeeze().squeeze()
                pred_image = step_dict['pred_obs_q']['see_image'].squeeze().squeeze()
                view.update(
                    real_image, pred_image,
                    action_text=f"Yaw: {yaw}.\n"
                                f"Speed: {spe}.",
                    extra_text=f"epoch {e}, step {s}.\nreward {reward + wall_punishment}.")
            if(end):
                image, speed = maze_runner.obs()
                image = image.unsqueeze(1)
                final_obs = {'see_image' : image}
                maze_runner.begin()
                break
        for j in range(len(push_list)):
            if j == len(push_list) - 1:
                agent.buffer.push(
                    push_list[j]['observation_dict'], 
                    push_list[j]['action_dict'], 
                    push_list[j]['reward'], 
                    final_obs, 
                    push_list[j]['done'], 
                    best_action_dict = None)
                print("\t\tLAST REWARD:", push_list[j]['reward'])
            else:
                agent.buffer.push(
                    push_list[j]['observation_dict'], 
                    push_list[j]['action_dict'], 
                    push_list[j]['reward'], 
                    push_list[j+1]['observation_dict'], 
                    push_list[j]['done'], 
                    best_action_dict = None)
        agent.epoch(32)
        if e % 100 == 0:
            fig = plot_training_log(agent)
            fig.savefig("hi.png")
            plt.close(fig)
# %%
