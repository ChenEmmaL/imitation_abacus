# Imitation abacus
RL agent that learns a predetermined algorithm. Code for PPO archittecture and DQN.
Each folders contains the code version for each model (minor differences from each other, mainly the reward function).
Run
```
python run_env.py
```
to train the architecture and save the models.
In folder **rendering_code** there is the experimental code for generating the gifs of the model (weights needed)
E.g. run
```
python render_env.py 3+4-21
```