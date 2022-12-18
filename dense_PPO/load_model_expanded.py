from SingleAgentB import SingleRLAgent, TestEnv_Both, numberToBase
import omegaconf
from sb3_contrib import MaskablePPO
from eval_env import evaluate_policy2
from stable_baselines3.common.vec_env import DummyVecEnv



cfg = omegaconf.OmegaConf.load("config_params.yaml")
#env = SingleRLAgent(cfg.env_params)
env = TestEnv_Both(cfg.env_params)

model = MaskablePPO.load("stacked_model.zip", env=env)




train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 2) for i in range(1000)])
total_solved, failed = evaluate_policy2(model, train_env, n_eval_episodes=10000)
print(total_solved, failed)

train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 4) for i in range(1000)])
total_solved, failed = evaluate_policy2(model, train_env, n_eval_episodes=10000)
print(total_solved, failed)

train_env = DummyVecEnv([lambda: TestEnv_Both(cfg.env_params, number_of_digits = 8) for i in range(1000)])
total_solved, failed = evaluate_policy2(model, train_env, n_eval_episodes=10000)
print(total_solved, failed)