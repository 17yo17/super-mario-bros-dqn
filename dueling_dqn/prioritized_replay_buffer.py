from collections import deque
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, transition, priority=None):
        self.memory.append(transition)
        if priority is None:
            max_priority = max(self.priorities, default=1.0)
            self.priorities.append(max_priority)
        else:
            self.priorities.append(priority)
    
    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / sum(scaled_priorities)

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        transitions = [self.memory[idx] for idx in indices]

        # Importance-sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        
        return transitions, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
