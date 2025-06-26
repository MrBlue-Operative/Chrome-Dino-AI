from mss import mss
import pydirectinput
import numpy as np
import cv2
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# Import for AI
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker 
from stable_baselines3 import DQN
 
import matplotlib
matplotlib.use('TkAgg')

class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Extraction for the game
        self.cap = mss()
        self.game_location = {'top':180, 'left':370, 'width':600, 'height':300}
        self.done_location = {'top':200, 'left':570, 'width':280, 'height':70}

    def step(self, action):
        action_object = {
            0:'space', 1:'down', 2:'no_op'
        }
        if action != 2:
            pydirectinput.press(action_object[action])

        done, done_cap = self.get_done()
        new_observation = self.get_observation()
        # Reward system
        if done:
            reward = -50
        else:
            reward = 10
        info = {}
        return new_observation, reward, done, False, info

    def render(self):
        # plt.imshow(cv2.cvtColor(self.get_obs()[0], cv2.COLOR_BGR2RGB))
        # plt.show()
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
            

    def close(self):
        cv2.destroyAllWindows()

    def reset(self, *, seed=None, options=None):
        time.sleep(0.1)
        pydirectinput.click(x=200, y=200)
        pydirectinput.press('space')
        return self.get_observation(), {}
    
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
        resized = cv2.resize(gray, (100, 83))  # Resize to (100, 83)
        gray = np.reshape(resized, (1, 83, 100))  # Shape: (1, 83, 100)
        return gray
    
    def get_done(self):
        doneCap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        strings = ['GAME', 'GAHE']
        
        done = False
        res = pytesseract.image_to_string(doneCap)[:4]
        if res in strings:
            done = True
        return done, doneCap


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
        return True
# Check the callback function and i think everything else is good
# os.makedirs('./train/', exist_ok=True)
# os.makedirs('./logs/', exist_ok=True)

env = WebGame()
CHECK_DIR = 'C:/Users/DELL 7290/Desktop/Dino Game/train/'  # Absolute path
CHECK_LOGS = 'C:/Users/DELL 7290/Desktop/Dino Game/logs/'

print("Testing environment...")
env_checker.check_env(env, warn=True)

last_model_path = os.path.join(CHECK_DIR, "final_modelPart2")  # Adjust based on available save
if os.path.exists(last_model_path + ".zip"):
    model = DQN.load(last_model_path)
    print(f"Loaded model from {last_model_path}.zip")

    # i am changing the learning rate from 0.0001 to 0.002 so that it could learn faster 
    # Here is the implementation 

    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = 0.001  # Set new learning rate

else:
    print(f"No saved model found at {last_model_path}.zip. Starting fresh.")
    model = DQN('CnnPolicy', env, tensorboard_log=CHECK_LOGS, verbose=1, buffer_size=4000, learning_starts=100)

# Set the environment for the loaded model
model.set_env(env)


callback = TrainAndLoggingCallback(check_freq=2000, save_path=CHECK_DIR)

# model = DQN('CnnPolicy', env, tensorboard_log=CHECK_LOGS, verbose=1, buffer_size=2000, learning_starts=50)
model.learn(total_timesteps=100, callback=callback)
def new_func(CHECK_DIR, model):
    save_path = os.path.join(CHECK_DIR, "Final_Result")
    os.makedirs(CHECK_DIR, exist_ok=True)  # Ensure directory exists
    model.save(save_path)  # Save the model
    print(f"Model saved to {save_path}.zip")

new_func(CHECK_DIR, model)


