from SingleAgentB import Both_operation
from net_arch import CustomCombinedExtractor, exp_schedule, CustomCallback
import torch
from sb3_contrib import MaskablePPO
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
        net_arch=[dict(pi=[2048, 1024, 512, 256, 128], vf=[2048, 1024, 512, 256, 128])],
    )
    
    
    model = MaskablePPO('MultiInputPolicy', env, verbose=1, vf_coef = 0.01, n_steps=cfg.train_params["n_steps"], learning_rate = exp_schedule(cfg.train_params["lr"]), policy_kwargs=policy_kwargs, 
                        target_kl  = 0.2, tensorboard_log="runs", clip_range=0.2, ent_coef = cfg.train_params["ent_coef"])
    
    checkpoint_callback = CheckpointCallback(save_freq=500000, save_path='./models/', name_prefix='rl_model')
    other_callback = CustomCallback(env)
    callback = CallbackList([other_callback, checkpoint_callback])
    
    model.learn(total_timesteps=35000000, callback=callback)
    
    model.save("stacked_model")
    run.finish()
 
if __name__ == '__main__':
    train_and_save_model()
