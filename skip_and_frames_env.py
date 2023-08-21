import gym
import numpy as np

class SkipAndFramesEnv(gym.Wrapper):
    def __init__(self, env, num_frames = 4):
        super(SkipAndFramesEnv, self).__init__(env)

        self.k = num_frames
        self.frames_stack = []
    
    def reset(self, **kwargs):
        self.last_life_count = 0
        observation = self.env.reset(**kwargs)
        observation = np.stack([observation, observation, observation, observation])
        return observation

    def step(self, action):
        total_reward = 0
        score = 0
        frames = []

        for _ in range(self.k):
            observation, reward, done, score = self.env.step(action)
            frames.append(observation)
            total_reward += reward

            if done:
                break
        
        self.step_frame_stack(frames)
        
        if done:
            print('Score final: ', score)
        
        total_reward = 4 if total_reward > 0 else -15

        return self.frames_stack, total_reward, done, score

    def step_frame_stack(self, frames):
        
        num_frames = len(frames)

        if num_frames == self.k:
            self.frames_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frames_stack = np.array(frames[-self.k::])
        else:
            self.frames_stack[0: self.k - num_frames] = self.frames_stack[num_frames::]
            self.frames_stack[self.k - num_frames::] = np.array(frames)  
