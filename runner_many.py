from datetime import date, timedelta
import math
import time

from intraday.providers import RandomWalkProvider
from intraday.processor import IntervalProcessor
from intraday.features import EMA, Copy, PriceEncoder, AbnormalTrades, \
ADL, Delta, AverageTrade, CMF, KAMA, LogReturn
from intraday.actions import BuySellCloseAction, PingPongAction
from intraday.rewards import BalanceReward
from intraday.env import SingleAgentEnv
import numpy as np
from stable_baselines3 import A2C, DQN, PPO, TD3

from run_helper import FlattenDictWrapper



provider = RandomWalkProvider(step_limit=1000, volume_limit=20, date_from=date(2018, 5, 1), 
                              date_to=date(2024, 9, 30),
                              walking_threshold=0.47,
                              seed=math.floor(time.time()))
processor = IntervalProcessor(method='volume', interval=5*60)
#period = 1000
period: tuple = (2, 10, 5)
frames_ema_period = 20


atr_name = f'ema_{frames_ema_period}_true_range'
features_pipeline = [
    AbnormalTrades(), 
    AverageTrade(),
    PriceEncoder(source='close', write_to='both'),
    # Stochastic(),
    ADL(), 
    LogReturn(),
    # AverageTrade(), 
    CMF(),
    Delta(),
    
    EMA(period=frames_ema_period, source='true_range', write_to='both'),
    KAMA(period=period, source='low', write_to='both'),
    KAMA(period=period, source='high', write_to='both'),
    Copy(source=['volume']),
]
action_scheme = BuySellCloseAction()
reward_scheme = BalanceReward()
raw_env = SingleAgentEnv(
    provider=provider,
    processor=processor,
    features_pipeline=features_pipeline,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    initial_balance=100000,
    max_trades=20,
    warm_up_time=timedelta(hours=1),
    # n_agents=1
)

names = [
    f'kama_{"_".join(str(x) for x in period)}_low',
    f'kama_{"_".join(str(x) for x in period)}_high',
    f'ema_{frames_ema_period}_true_range'
]

print(f"names:  {names}")


counter = 0

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
import random
from tqdm import tqdm

# # Example usage
# env = gym.make('YourGymEnv')  # Replace with your actual environment
env = FlattenDictWrapper(raw_env)
check_env(env)


def print_stats(reward_over_episodes):
    """  Print Balance  """

    avg = np.mean(reward_over_episodes)
    min = np.min(reward_over_episodes)
    max = np.max(reward_over_episodes)

    print (f'Min. Balance          : {min:>10.3f}')
    print (f'Avg. Balance          : {avg:>10.3f}')
    print (f'Max. Balance          : {max:>10.3f}')

    return min, avg, max

class ProgressBarCallback(BaseCallback):

    def __init__(self, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.progress_bar = tqdm(total=self.model._total_timesteps, desc="model.learn()")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.progress_bar.update(self.check_freq)
        return True
    
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.progress_bar.close()



# TRAINING + TEST
def train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps=10_000):
    """ if model=None then execute 'Random actions' """

    # reproduce training and test
    print('-' * 80)
    if seed is not None:
        env.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    obs = env.reset()

    # vec_env = None

    if model is not None:
        print(f'model {type(model)}')
        print(f'policy {type(model.policy)}')
        # print(f'model.learn(): {total_learning_timesteps} timesteps ...')

        # custom callback for 'progress_bar'
        model.learn(total_timesteps=total_learning_timesteps, callback=ProgressBarCallback(100))
        # model.learn(total_timesteps=total_learning_timesteps, progress_bar=True)
        # ImportError: You must install tqdm and rich in order to use the progress bar callback. 
        # It is included if you install stable-baselines with the extra packages: `pip install stable-baselines3[extra]`

        # vec_env = model.get_env()
        obs = env.reset()
    else:
        print ("RANDOM actions")

    reward_over_episodes = []
    balance_over_episodes = []

    tbar = tqdm(range(total_num_episodes))

    for episode in tbar:

        obs = env.reset()      

        total_reward = 0
        balances = []
        done = False

        while not done:
            if model is not None:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
            else: # random
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)

            total_reward += reward
            if done:
                break
        

        balance = env.account.balance if model is not None else total_reward
        reward_over_episodes.append(total_reward)
        balance_over_episodes.append(balance)

        if episode % 10 == 0:
            avg_reward = np.mean(reward_over_episodes)
            avg_balance = np.mean(balance_over_episodes)
#            tbar.set_description(f'Episode: {episode}, Avg. Reward: {avg_reward:.3f}, avg balance {avg_balance:.3f}')
            tbar.set_description(f'Episode: {episode}, avg balance {avg_balance:.3f}')
            tbar.update()

    tbar.close()
    avg_reward = np.mean(reward_over_episodes)
    avg_balance = np.mean(balance_over_episodes)


#    return reward_over_episodes
    return balance_over_episodes


seed = None #42  # random seed
total_num_episodes = 100

print ("----------------------------")
print ("Env                      :", env)
print ("seed                     :", seed)

# INIT matplotlib
plot_settings = {}
plot_data = {'x': [i for i in range(1, total_num_episodes + 1)]}

# Random actions
model = None 
total_learning_timesteps = 0
rewards = train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps)
min, avg, max = print_stats(rewards)
class_name = f'Random actions'
label = f'Avg. {avg:>7.2f} : {class_name}'
plot_data['rnd_rewards'] = rewards
plot_settings['rnd_rewards'] = {'label': label}

learning_timesteps_list_in_K = [25, 50]
# learning_timesteps_list_in_K = [50, 250, 500]
# learning_timesteps_list_in_K = [500, 1000, 3000, 5000]

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
model_class_list = [A2C, PPO, DQN]

for timesteps in learning_timesteps_list_in_K:
    total_learning_timesteps = timesteps * 1000
    step_key = f'{timesteps}K'

    for model_class in model_class_list:
        policy_dict = model_class.policy_aliases
        # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        # MlpPolicy or MlpLstmPolicy
        policy = policy_dict.get('MlpPolicy')
        if policy is None:
            policy = policy_dict.get('MlpLstmPolicy')
        print ('policy:', policy, 'model_class:', model_class)

        try:
            model = model_class(policy, env, verbose=0)
            class_name = type(model).__qualname__
            plot_key = f'{class_name}_rewards_'+step_key
            rewards = train_test_model(model, env, seed, total_num_episodes, total_learning_timesteps)
            min, avg, max, = print_stats(rewards)
            label = f'Avg. {avg:>7.2f} : {class_name} - {step_key}'
            plot_data[plot_key] = rewards
            plot_settings[plot_key] = {'label': label}     
                   
        except Exception as e:
            print(f"ERROR: {str(e)}")
            continue
