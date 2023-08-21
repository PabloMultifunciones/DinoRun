import matplotlib.pyplot as plt
import torch

def compute_advantage_and_value_targets(rewards, values, dones, gamma, lam=0.95):

    advantage_values = []
    advantage_accumulation = torch.tensor(0.0)

    for t in reversed(range(len(rewards)-1)):
            
        if dones[t]:
            advantage_accumulation = torch.tensor(0.0)
            
        delta_t = rewards[t] + (gamma*(values[t+1])*int(not dones[t+1])) - values[t]
            
        A_t = delta_t + gamma*lam*advantage_accumulation

        advantage_values.append(A_t)
            
        advantage_accumulation = delta_t + gamma*lam*advantage_accumulation

    advantage_values.reverse()
        
    return advantage_values

def draw_plot(history_score, color = "red"):
    plt.title("Historial de Puntajes")
    plt.xlabel("Ciclos")
    plt.ylabel("Puntaje")
    plt.plot(history_score, color)
    plt.show()
