
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
# from gym.wrappers.monitoring import video_recorder
import argparse
import time
import numpy as np
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
#from gymnasium.wrappers.monitoring import video_recorder
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import frame_skipper
import agent_dqn
import joypad_ppo

TEST_EPISODES = 10 # How many episodes to go through when testing
MAX_ITER = 15000 # Max iterations when training
MODEL_PATH = './models/'
VIDEO_PATH = './test_videos/'
GRAPH_PATH = './graphs/'

def parse():
    parser = argparse.ArgumentParser(description="Super Mario RL Project 4")
    parser.add_argument('--train', action='store_true', help='whether to train DQN')
    parser.add_argument('--test', action='store_true', help='whether to test DQN')
    parser.add_argument('--model', default=None, help='model file name. If given with train, will continue training from this model')
    parser.add_argument('--display', action='store_true', help='whether to show the gameplay, do not use with record video')
    parser.add_argument('--record_video', action='store_true', help='whether to record video during testing, do not use with display.')
    parser.add_argument('--ppo', action='store_true', help='whether to run with PPO, else will default to DQN')
    args = parser.parse_args()
    return args

def make_env_ppo(rank, display):
    def _init():
        if (not display):
            env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
        else:
            env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
        #env = JoypadSpace(env, RIGHT_ONLY)
        env = joypad_ppo.JoyPadSpacePPO(env, RIGHT_ONLY)
        env = frame_skipper.FrameSkipper(env, skip=4)
        env = ResizeObservation(env, shape=84)
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=4, lz4_compress=True)
        return env
    return _init

def setup_env(args):
    if (args.ppo): # Use PPO with parallelization
        num_envs = 8
        if (args.test):
            num_envs = 1
        envs = SubprocVecEnv([make_env_ppo(i, args.display) for i in range(num_envs)])
        return envs
    else:
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human' if args.display else 'rgb_array', apply_api_compatibility=True)
        # all of these help with performance
        env = JoypadSpace(env, RIGHT_ONLY) # agent only needs to go right
        env = frame_skipper.FrameSkipper(env, skip=4)
        env = ResizeObservation(env, shape=84)
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=4, lz4_compress=True)
        return env

def test(args, env, video_path, agent, start_time):
    rewards = []
    vid = None
    '''
    if args.record_video:
        vid = video_recorder.VideoRecorder(env=env.env, path=video_path)
    '''
    
    for _ in tqdm(range(TEST_EPISODES)):
        episode_reward = 0.0
        truncated = False
        state, _ = env.reset()
        terminated = False
        while not terminated and not truncated:
            action = agent.take_action(state, test=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if args.record_video:
                vid.capture_frame()

        if truncated:
            break

    rewards.append(episode_reward)

    if args.record_video:
        vid.close()  # Ensure the video recorder is properly closed

    env.close()

    print('Run %d episodes' % (TEST_EPISODES))
    print('Mean:', np.mean(rewards))
    print('rewards', rewards)
    print('running time', time.time()-start_time)

def train_ppo(env):
    # Save checkpoints every 100,000 steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=MODEL_PATH, name_prefix='ppo_')

    # Define PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,  # Number of steps to collect before updating
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epochs to train for each update
        clip_range=0.2,  # Clipping range
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE smoothing factor
        vf_coef=0.5,  # Value function loss weight
        ent_coef=0.01,  # Entropy coefficient
        tensorboard_log="./ppo_tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=5000000, callback=checkpoint_callback)

    # Save final model
    model.save("./models/ppo_super_mario_final")
    print("PPO training complete and model saved")
    

def test_ppo(env, model_path, display=False):
    # Load the trained model
    model = PPO.load(model_path)
    obs = env.reset()
    for _ in range(10):  # Test for 10 episodes
        done = [False]
        total_reward = 0
        while not done[0]:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if (display):
                env.render()  # Optional: Render the environment for visualization
        print(f"Episode reward: {total_reward}")
    env.close()

def run(args, env):
    time_now = time.time()
    time_now_str = str(time_now)
    if (args.test):
        if (not args.ppo):
            if (args.model == None):
                print("Please provide the model file name to test")
                return
            # Run test with provided model
            model = MODEL_PATH + str(args.model)
            agent = agent_dqn.AgentDQN(env, args, model, time_now, False)
            vid_path = VIDEO_PATH + 'video_' + time_now_str + '.mp4'
            test(args, env, vid_path, agent, time_now)
        else:
            # Test PPO
            if (args.model == None):
                print("Please provide the model file name to test")
                return
            model = MODEL_PATH + str(args.model)
            test_ppo(env, model, args.display)
    elif (args.train):
        if (not args.ppo):
            continue_training = False
            if (args.model):
                continue_training = True
            model = MODEL_PATH + ('model_' + time_now_str + '.pth' if args.model is None else args.model)
            agent = agent_dqn.AgentDQN(env, args, model, time_now, continue_training)
            agent.train(MAX_ITER)
        else: # PPO
            print("Starting PPO training...")
            train_ppo(env)
    else:
        print("No action provided, exiting...")
        return

if __name__ == "__main__":
    args = parse()
    env = setup_env(args)
    run(args, env)
