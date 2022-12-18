from SingleAgentB import Both_operation
from net_arch import CustomCombinedExtractor, exp_schedule, CustomCallback
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import hydra

import wandb
from wandb.integration.sb3 import WandbCallback

@hydra.main(version_base=None, config_path=".", config_name="config_params")
def train_and_save_model(cfg):    
    parmas = cfg.env_params
    wandb.tensorboard.patch(root_logdir="runs")
    run = wandb.init(
        project="sb3_abacus",
        config=parmas,
        sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game, doesn't work
        save_code=False,  # optional
    )

    env = Both_operation(parmas)
    
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        features_extractor_class=CustomCombinedExtractor,
        net_arch=[2048, 1024, 512, 256, 128],
    )
    
    
    model = DQN('MultiInputPolicy', env, verbose=1, learning_rate=exp_schedule(cfg.train_params["lr"]), 
                buffer_size=100000, learning_starts=5000, tensorboard_log="runs", batch_size=64, policy_kwargs=policy_kwargs, 
                exploration_fraction=0.2, exploration_initial_eps=0.05, exploration_final_eps=0)
    
    

    
    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='./models/', name_prefix='rl_model')
    other_callback = CustomCallback(env)
    callback = CallbackList([other_callback, checkpoint_callback])
    
    model.learn(total_timesteps=35000000, callback=callback)
    
    model.save("stacked_model")
    run.finish()
 
if __name__ == '__main__':
    train_and_save_model()
