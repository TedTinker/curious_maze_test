#%%
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from utils import args

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, mu_std



# Encode Action (ea).
class Encode_Reward(nn.Module):
    def __init__(
            self, 
            arg_dict = {
                "encode_size" : 16,
                "zp_zq_sizes" : [16]}, 
            verbose = False):
        super(Encode_Reward, self).__init__()
        
        self.arg_dict = arg_dict
                
        self.example_input = torch.zeros((1, 1, 1))
        if(verbose):
            print("\nER Start:", self.example_input.shape)
            
        episodes, steps, [example] = model_start([(self.example_input, "lin")])
        
        self.a = nn.Sequential(
            nn.Linear(1, self.arg_dict['encode_size']),
            nn.PReLU(),
            nn.Linear(self.arg_dict['encode_size'], self.arg_dict['encode_size']),
            nn.PReLU())
                
        example = self.a(self.example_input)
        if(verbose): 
            print("\toutput:", example.shape)
        
        [example] = model_end(episodes, steps, [(example, "lin")])
        self.example_output = example
        if(verbose):
            print("ER End:")
            print("\toutput:", example.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, reward):
        episodes, steps, [reward] = model_start([(reward, "lin")])
        output = self.a(reward)
        [output] = model_end(episodes, steps, [(output, "lin")])
        return(output)
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    er = Encode_Reward(verbose = True)
    print("\n\n")
    print(er)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(er, er.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    
    
    
