from SingleAgentC import SingleRLAgent, TestEnv_Both, numberToBase
import omegaconf
from sb3_contrib import MaskablePPO
from eval_env import evaluate_policy2
from stable_baselines3.common.vec_env import DummyVecEnv



cfg = omegaconf.OmegaConf.load("config_params.yaml")
#env = SingleRLAgent(cfg.env_params)
env = TestEnv_Both(cfg.env_params)

model = MaskablePPO.load("best_model2.zip", env=env)



train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = None) for i in range(2000)])
total_solved, aaa = evaluate_policy2(model, train_env, n_eval_episodes=100000)
print("this is a signpost")
print(total_solved, aaa)


train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 1) for i in range(2000)])
total_solved, bbb = evaluate_policy2(model, train_env, n_eval_episodes=100000)
print("this is a signpost")
print(total_solved, bbb)

train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 2) for i in range(2000)])
total_solved, ccc = evaluate_policy2(model, train_env, n_eval_episodes=100000)
print("this is a signpost")
print(total_solved, ccc)

train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 4) for i in range(2000)])
total_solved, ddd = evaluate_policy2(model, train_env, n_eval_episodes=100000)
print("this is a signpost")
print(total_solved, ddd)

train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 8) for i in range(2000)])
total_solved, eee = evaluate_policy2(model, train_env, n_eval_episodes=100000)
print("this is a signpost")
print(total_solved, eee)

train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 16) for i in range(2000)])
total_solved, fff = evaluate_policy2(model, train_env, n_eval_episodes=100000)
print("this is a signpost")
print(total_solved, fff)


import pickle

# open a file, where you ant to store the data
file = open('important0_2', 'wb')

# dump information to that file
data = [aaa, bbb, ccc, ddd, eee, fff]
pickle.dump(data, file)

# close the file
file.close()