import torch
from agent import Agent
from skip_and_frames_env import SkipAndFramesEnv
from chrome_manager import ChromeManager
from environment import Environment

def test(arguments):
    learning_rate = arguments.lr
    gamma = arguments.gamma
    n_updates = arguments.epochs
    clip = arguments.clip
    stacked_frames = arguments.stacked_frames
    c1 = arguments.c1
    c2 = arguments.c2
    minibatch_size = arguments.minibatch_size
    in_channels = 4
    n_outputs = 2

    agente = Agent(in_channels, n_outputs, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2)

    raw_env = ChromeManager()
    raw_env = Environment(raw_env) 
    env = SkipAndFramesEnv(raw_env, stacked_frames)

    observation = env.reset()

    while True:
        observation = torch.FloatTensor(observation).unsqueeze(0)
                    
        action = agente.get_action_max_prob(observation)

        observation, _, done, _ = env.step(action.item())

        if done:
            observation = env.reset()

if __name__ == '__main__':
    test()


















