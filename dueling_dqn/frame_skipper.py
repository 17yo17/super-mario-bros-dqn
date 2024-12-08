
from gym import Wrapper

class FrameSkipper(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        reward = 0.0
        done = False
        for _ in range(self.skip):
            new_state, current_reward, done, truncated, extra = self.env.step(action)
            reward += current_reward
            if done:
                break
        return new_state, reward, done, truncated, extra
