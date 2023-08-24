from agent import Agent
from skip_and_frames_env import SkipAndFramesEnv
from chrome_manager import ChromeManager
from environment import Environment
from worker import Worker
import torch
import numpy as np
from auxiliars import compute_gae, draw_plot

def train(arguments):
    learning_rate = arguments.lr
    gamma = arguments.gamma
    n_updates = arguments.epochs
    clip = arguments.clip
    stacked_frames = arguments.stacked_frames
    c1 = arguments.c1
    c2 = arguments.c2
    minibatch_size = arguments.minibatch_size
    batch_size = arguments.batch_size
    cicles = arguments.cicles
    lam = arguments.lam
    in_channels = 4
    n_outputs = 2

    agente = Agent(in_channels, n_outputs, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2)

    raw_env = ChromeManager()
    raw_env = Environment(raw_env) 
    env = SkipAndFramesEnv(raw_env, stacked_frames)

    env_runner = Worker(env, agente, batch_size)

    scores_history = []
    for _ in range(cicles):
        batch_observations, batch_actions, batch_rewards, batch_dones, batch_values, batch_old_action_prob, max_score = env_runner.run()

        scores_history.append(max_score)

        advantages = compute_gae(batch_rewards, batch_values, batch_dones, gamma, lam)

        batch_observations = torch.stack(batch_observations[:-1])
        batch_actions = np.stack(batch_actions[:-1])
        batch_advantages = torch.stack(advantages)
        batch_old_action_prob = torch.stack(batch_old_action_prob[:-1])

        agente.update(batch_observations, batch_actions, batch_advantages, batch_old_action_prob)
        agente.save_models()


    env.end()

    draw_plot(scores_history)


if __name__ == '__main__':
    train()


















