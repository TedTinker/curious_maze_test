
#%%





import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

from general_FEP_RL.utils_torch import init_weights, model_start, model_end, mu_std



# Decode Reward (dar).
class Decode_Reward(nn.Module):
    def __init__(
            self, 
            hidden_state_size, 
            encoded_action_size = 0, 
            entropy = False, 
            arg_dict = {}, 
            verbose = False):
        super(Decode_Reward, self).__init__()
                
        self.example_input = torch.zeros((32, 16, hidden_state_size + encoded_action_size))
        if(verbose): 
            print("\nDR Start:", self.example_input.shape)

        episodes, steps, [example] = model_start([(self.example_input, "lin")])
        
        self.a = nn.Sequential(
            nn.Linear(
                in_features = hidden_state_size,
                out_features = hidden_state_size),
            nn.PReLU(),
            nn.Linear(
                in_features = hidden_state_size,
                out_features = 16),
            nn.PReLU())
        
        example = self.a(example)
        if(verbose): 
            print("\ta:", example.shape)
                
        mu = nn.Sequential(
            nn.Linear(
                in_features = 16,
                out_features = 1),
            nn.Tanh())
        
        self.mu_std = mu_std(mu, entropy = entropy)
        
        example_output, example_log_prob = self.mu_std(example)
        if(verbose): 
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape)
        
        [example_output, example_log_prob] = model_end(episodes, steps, [(example_output, "lin"), (example_log_prob, "lin")])
        self.example_output = example_output
        if(verbose): 
            print("DR End:")
            print("\toutput:", example_output.shape)
            print("\tlog_prob:", example_log_prob.shape, "\n")
        
        self.apply(init_weights)
        
        
        
    def forward(self, hidden_state):
        episodes, steps, [hidden_state] = model_start([(hidden_state, "lin")])
        a = self.a(hidden_state)
        output, log_prob = self.mu_std(a)
        [output, log_prob] = model_end(episodes, steps, [(output, "lin"), (log_prob, "lin")])
        return(output, log_prob)
    
    
    
    @staticmethod
    def loss_func(predicted_values, target_values):
        loss_value = F.mse_loss(predicted_values, target_values, reduction = 'none')
        return loss_value
    
    
    
# Let's check it out!
if(__name__ == "__main__"):
    dr = Decode_Reward(hidden_state_size = 12, verbose = True)
    print("\n\n")
    print(dr)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dr, dr.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    

    dr = Decode_Reward(hidden_state_size = 128, entropy = True, verbose = True)
    print("\n\n")
    print(dr)
    print()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            print(summary(dr, dr.example_input.shape))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
    
    
    
    
    