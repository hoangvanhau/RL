# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import frame stacker wrapper amd grayscaling wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt
# Import os for the file path management
import os
# Import PPO for algo
from stable_baselines3 import PPO
# Import Base Callback for saving model
from stable_baselines3.common.callbacks import BaseCallback

# 1. Create the game environtment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the control
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. GrayScale the observation
env = GrayScaleObservation(env, keep_dim=True) 
# 4. Wrap inside the dummy Environment.
env = DummyVecEnv([lambda: env])
# 5. Stack the frames.
env = VecFrameStack(env, 4, channels_order='last')

# state = env.reset()
# state, reward, done, info = env.step([5])
# state, reward, done, info = env.step([5])
# state, reward, done, info = env.step([5])
# state, reward, done, info = env.step([5])
# print(state.shape)
# plt.figure(figsize=(20,16))
# for idx in range(state.shape[3]):
#     plt.subplot(1, 4, idx+1)
#     plt.imshow(state[0][:,:, idx])
# plt.show()

""" Train the RL model"""
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

CHECKPOINT_DIR = './train'
LOG_DIR = './logs'

# # Setup model saving callback
# callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
# # Train the AI model, this is where the AI model starts to learn
# model.learn(total_timesteps=1000000, callback=callback)
"""Test it out"""
# Load model
model = PPO.load('./train/best_model_1000000')
# Start the game
state = env.reset()
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()