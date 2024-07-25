from datetime import date, timedelta
import datetime
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
from stable_baselines3 import A2C, DQN, PPO

from run_helper import FlattenDictWrapper

provider = RandomWalkProvider(step_limit=1000, volume_limit=20, date_from=date(2018, 5, 1), 
                              date_to=date(2024, 9, 30),
                              walking_threshold=0.49,
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
env = SingleAgentEnv(
    provider=provider,
    processor=processor,
    features_pipeline=features_pipeline,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    initial_balance=100000,
    max_trades=20,
    warm_up_time=timedelta(hours=1),
)

names = [
    f'kama_{"_".join(str(x) for x in period)}_low',
    f'kama_{"_".join(str(x) for x in period)}_high',
    f'ema_{frames_ema_period}_true_range'
]

print(f"names:  {names}")


counter = 0

from stable_baselines3.common.env_checker import check_env
from gym import spaces


# # Example usage
# env = gym.make('YourGymEnv')  # Replace with your actual environment
wrapped_env = FlattenDictWrapper(env)
check_env(wrapped_env)

mode_train = True
# mode_train = False
total_timesteps = 50000

model_type = DQN

if mode_train:
    model = model_type('MlpPolicy', wrapped_env, verbose=0)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(f"dqn_model-{total_timesteps}")

model = model_type.load(f"dqn_model-{total_timesteps}", force_reset=True)

unwrapped_eval_env = SingleAgentEnv(
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
eval_env = FlattenDictWrapper(unwrapped_eval_env)

balances = []
net_profits = []
rewards = []
episodes = 50
# obs = eval_env.reset()
do_render = False #True

for i in range(episodes):
    done = False
    obs = eval_env.reset()
    total_reward = 0
    while not done:
        action, _states = model.predict(obs)

        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        # print(f'Action {action}, reward {reward}, frame {info}, \n\t obs {obs} {eval_env.unwrap(obs)}')
        if do_render:
            eval_env.render('human')
    balance = eval_env.account.balance
    net_profit = eval_env.account.report.net_profit
    if balance is not None:
        balances.append(balance)
    if net_profit is not None:
        net_profits.append(net_profit)

    print(f'{i} Balance: {balance}, profit {net_profit}, reward {total_reward}')

#eval_env.print_summary()

print(f"Balance: non-None: {len(balances)}, mean {np.mean(balances)}, max {max(balances)}, min {min(balances)}")
print(f"Net profit: non-None: {len(net_profits)}, mean {np.mean(net_profits)}, max {max(net_profits)}, min {min(net_profits)}")
print(f"Reward: non-None: {len(rewards)}, mean {np.mean(rewards)}, max {max(rewards)}, min {min(rewards)}")


wrapped_env.close()
eval_env.close()