
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import dqn_model
import prioritized_replay_buffer

GRAPH_PATH = './graphs/'

class AgentDQN:
    def __init__(self, env, args, model_path, time_now, continue_training):
        self.env = env
        self.num_actions = env.action_space.n
        self.model_path = model_path
        self.time_str = str(time_now)
        
        # Set DQN parameters here
        self.learning_rate = .00025
        self.gamma = .9
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
        self.decay = .99999
        self.epsilon_min = .1
        self.batch_size = 32
        self.replay_buffer_size = 100000
        self.train_start = 10000
        self.update_target_every = 10000
        
        # Replay buffer
        self.memory = deque(maxlen=self.replay_buffer_size)
        # Try prioritized replay buffer
        # self.memory = prioritized_replay_buffer.PrioritizedReplayBuffer(self.replay_buffer_size)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.online_net = dqn_model.DuelingDQN(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net = dqn_model.DuelingDQN(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        
        if args.model is not None and continue_training:
            print("Loading existing model:", self.model_path)
            checkpoint = torch.load(model_path)
            self.online_net.load_state_dict(checkpoint['online_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            
            # Load graphing data
            self.episode_rewards = checkpoint.get('rewards', [])
            self.avg_rewards = checkpoint.get('avg_rewards', [])
            self.losses = checkpoint.get('losses', [])
            self.q_values = checkpoint.get('q_values', [])
            self.time_str = checkpoint.get('chart_time', self.time_str)
            print("Model loaded successfully!")
        else:
            # Initialize variables for graphing
            self.episode_rewards = [] # store rewards for graph
            self.avg_rewards = [] # store average rewards over time
            self.losses = []
            self.q_values = []

        if args.test:
            print("Loading existing model:", self.model_path)
            checkpoint = torch.load(model_path)
            self.online_net.load_state_dict(checkpoint['online_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            
            # Load graphing data
            self.episode_rewards = checkpoint.get('rewards', [])
            self.avg_rewards = checkpoint.get('avg_rewards', [])
            self.losses = checkpoint.get('losses', [])
            self.q_values = checkpoint.get('q_values', [])
            self.time_str = checkpoint.get('chart_time', self.time_str)
            print("Model loaded successfully!")
            self.online_net.eval()
    
    def take_action(self, observation, test=True):
        if test:
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            # observation = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(observation)
                action = torch.argmax(q_values).item()
        else:
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.num_actions)
            else:
                observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                # observation = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
                with torch.no_grad():
                    q_values = self.online_net(observation)
                    action = torch.argmax(q_values).item()
        return action

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay_buffer(self):
        if (len(self.memory) < self.train_start):
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.stack(states)).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        # states = torch.FloatTensor(np.stack(states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.FloatTensor(np.stack(next_states)).permute(0, 3, 1, 2).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones
    '''
    # Old prioritized replay buffer implementation
    def replay_buffer(self, beta=0.4):
        if len(self.memory.memory) < self.train_start:
            return None
        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    '''

    def train(self, iterations):
        state, _ = self.env.reset()
        score = 0
        episode = 0
        looper = iterations
        counter = 0

        while(looper > 0):
            looper -= 1
            action = self.take_action(state, test=False)
            print("Training iteration: " + str(counter + 1))
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            score += reward
            self.push(state, action, reward, next_state, terminated or truncated)
            state = next_state

            if terminated or truncated:
                episode += 1
                self.episode_rewards.append(score)
                self.avg_rewards.append(np.mean(self.episode_rewards[-100:])) # avg last 100 episodes
                print("Episode:", episode, "Score:", score, "Epsilon:", self.epsilon)
                state, _ = self.env.reset()
                score = 0
                if episode % 10 == 0: # update graph every 10
                    self.plot_rewards(self.episode_rewards, self.avg_rewards, self.time_str)
                    self.plot_loss(self.losses, self.time_str)
                    self.plot_q_values(self.q_values, self.time_str)
            elif (looper == 0):
                looper += 1 # Finish the current episode
            if counter % self.update_target_every == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if counter > self.train_start:
                batch = self.replay_buffer()
                if batch is None:
                    continue

                states, actions, rewards, next_states, dones = batch

                # Compute TD error
                current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                # Compute V(s_{t+1}) for all next states
                next_q_values = self.target_net(next_states).max(1)[0]
                expected_q_values = (next_q_values * self.gamma) * (~dones) + rewards

                loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none').mean()
                self.losses.append(loss.item())
                
                self.q_values.append(current_q_values.mean().item())

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.online_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon - (self.epsilon_start - self.epsilon_min) / self.decay)
            counter += 1
        # plot last graph
        self.plot_rewards(self.episode_rewards, self.avg_rewards, self.time_str)
        self.plot_loss(self.losses, self.time_str)
        self.plot_q_values(self.q_values, self.time_str)
        # Save model
        checkpoint = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'rewards': self.episode_rewards,
            'avg_rewards': self.avg_rewards,
            'losses': self.losses,
            'q_values': self.q_values,
            'chart_time': self.time_str
        }
        torch.save(checkpoint, self.model_path)
        print("Model saved to {self.model_path}")

    def plot_rewards(self, episode_rewards, avg_rewards, time_now_str):
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label='Reward v Episode')
        plt.plot(avg_rewards, label='Average Reward (100 Episodes)', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward vs. Episodes')
        plt.legend()
        plt.grid()
        plt.savefig(GRAPH_PATH + 'reward_' + time_now_str + '.png')
        plt.close()

    def plot_loss(self, losses, time_now_str):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Loss v Training Steps', color='red')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Loss vs. Training Steps')
        plt.legend()
        plt.grid()
        plt.savefig(GRAPH_PATH + 'loss_' + time_now_str + '.png')
        plt.close()

    def plot_q_values(self, q_values, time_now_str):
        plt.figure(figsize=(10, 6))
        plt.plot(q_values, label='Q Value v Training Steps', color='green')
        plt.xlabel('Training Steps')
        plt.ylabel('Q Value')
        plt.title('Q Value vs. Training Steps')
        plt.legend()
        plt.grid()
        plt.savefig(GRAPH_PATH + 'q_values_' + time_now_str + '.png')
        plt.close()
