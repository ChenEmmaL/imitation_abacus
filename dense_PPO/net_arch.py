from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable
import torch
import torch.nn as nn
import gym
import numpy as np

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.enviroment = env
        #I know, I know, I should use a much more elegant data structure
        self.prev_timestep = 0
        self.prev_best_mean_reward = 200
        self.counter = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps%2048 == 0:
            self.enviroment.reset_accumulation = True
            accuracy = self.enviroment.accumulated_accuracy / self.enviroment.total_accumulation
            self.logger.record('number of successful sums', accuracy)
            if accuracy > 1:
                true_accuracy = 1/accuracy
            else:
                true_accuracy = 1
            self.logger.record('accuracy', true_accuracy)
            true_true_accuracy = 1 - true_accuracy
            self.logger.record('pseudo accuracy', true_true_accuracy)
            
            if self.prev_timestep+2000000 < self.num_timesteps:
                self.prev_best_mean_reward = 200

            if accuracy > self.prev_best_mean_reward:
                self.prev_best_mean_reward = accuracy
                if self.prev_timestep+2000000 < self.num_timesteps:
                    self.counter = self.counter+1
                self.prev_timestep = self.num_timesteps
                counter_for_path = self.counter
                path_to_save_it = "good_model0/best_model" + str(counter_for_path)
                self.model.save(path_to_save_it)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        #coef = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        #self.model.ent_coef = 3.5*10**(-coef - 1)
        #if coef >= 20:
        #    self.model.ent_coef = 0
        
        self.enviroment.counter_that_is_used_for_debbuging = 1
        
    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "space1":
                # Don't bother too much the comment below ##############################
                # We will just downsample one channel of the image by 4x4 and flatten.##
                # Assume the image is single-channel (subspace.shape[0] == 0)         ##
                ########################################################################
                extractors[key] = CustomCNN(subspace, 1024)
                total_concat_size += 1024
            elif key == "space0a":
                # Run through a simple MLP
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
            elif key == "space0b":
                # Run through a simple MLP
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
            elif key == "space0c":
                # Run through a simple MLP
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
            elif key == "sign_space":
                # Run through a simple MLP
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

import math

def exp_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        #((1-np.tanh((1-progress_remaining)*5-2))*0.5*initial_value)
        return( progress_remaining*(math.sin((progress_remaining*87.5- (math.pi/2)))+1) *initial_value+1e-10)

    return func

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func