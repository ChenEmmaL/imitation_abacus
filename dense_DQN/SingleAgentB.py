"""
This file contains the implementation of the environment from the point of view of a single agent. The environment class SingleRLAgent embeds three subclasses (FingerLayer, ExternalRepresentation, OtherInteractions) which implement the dynamics of the different environment parts.
"""


import copy
import numpy as np

import gym
from gym import spaces


def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def create_column_position_encoding(n):
    return(np.tile([0.1, 0.4, 0.7], n//3 + 1)[0:n])
    

class SingleRLAgent(gym.Env):
    """
    This class implements the environment as a whole.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, agent_params):
        
        self.params = agent_params
        
        #stuff for debugging
        self.counter_that_is_used_for_debbuging = 0
        self.list_of_action = [] #used for printing the list of action, nothing more, nothing less
        self.info = {}
        
        self.sign = False
        self.pos_number_reading = 0
        
        
        #declaring spaces dimensions for SB3
        
        if (self.params['task'] == 'sum_predetermined'):
            self.update_state = self.update_state_standard
            pre_dictionary_spaces  = {"space0" : spaces.Discrete(self.params['base_input_number']+1), 
                                      "space1" : spaces.Box(low=0, high=1,shape=(self.params['n_frames'], 2, self.params['ext_shape'][0], self.params['ext_shape'][1]), dtype=np.float64)}
            self.observation_space = spaces.Dict(pre_dictionary_spaces)
            
            
        if(self.params['task'] == 'simplified'):
            self.update_state = self.update_state_localized
            pre_dictionary_spaces  = {"space0" : spaces.Discrete(self.params['base_input_number']+1), 
                                      "space1" : spaces.Box(low=0, high=1,shape=(self.params['n_frames'], 2, 2, self.params['ext_shape'][1]+1), dtype=np.float64)}
            self.observation_space = spaces.Dict(pre_dictionary_spaces)
            
        if(self.params['task'] == 'scaling'):
            self.update_state = self.update_state_localized_scaling
            
            if self.params['auto_signpost']:
                space_len = self.params['ext_shape'][1]
            else:
                space_len = self.params['ext_shape'][1]+1
            
            pre_dictionary_spaces = {"space0a" : spaces.Discrete(space_len), "space0b" : spaces.Discrete(space_len), "space0c" : spaces.Discrete(space_len),
                                      "space1" : spaces.Box(low=0, high=1,shape=(self.params['n_frames'], 2, 2, self.params['ext_shape'][1]+1), dtype=np.float64)}
            self.observation_space = spaces.Dict(pre_dictionary_spaces)
        
        if self.params['auto_signpost']:
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.Discrete(8)
        
        
        
        #Initialization of the World
        self.ext_shape = agent_params['ext_shape']        
        # Initialize external representation (the piece of paper the agent is writing on)
        self.ext_repr = choose_external_representation(self.params['external_repr_tool'], self.ext_shape, self.params['task']) #ExternalRepresentation(self.obs_dim)
        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        self.fingerlayer = FingerLayer(self)
        # Initialize other interactions: 'submit', "s_left", "s_right"
        self.otherinteractions = OtherInteractions(self, self.params['task'])


        # Initialize action
        self.all_actions_list, self.all_actions_dict = self.merge_actions([self.ext_repr.actions, self.fingerlayer.actions, self.otherinteractions.actions])
        self.rewrite_all_action_keys()
        self.action_dim = len(self.all_actions_list)
        
        # Initialize reward function
        if (self.params['task'] == 'sum_predetermined') or (self.params['task'] == 'simplified') or (self.params['task'] == 'scaling'):
            self.reward_done_function = reward_done_function_predetermined
        if self.params['reward_shaping_custom']:
            self.reward_done_function = reward_done_function_custom
        
        # Initialization of probability array
        
        if self.params['base_input_number'] > 9:
            #actually an uniform
            self.probs_sampling = [1/self.params['base_input_number']]*self.params['base_input_number']
        else:
            benford_probs = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
            truncated_benford_probs = benford_probs[0:self.params['base_input_number']]
            self.probs_sampling = [x/sum(truncated_benford_probs) for x in truncated_benford_probs]
        
        if (self.params['task'] == 'sum_predetermined') or (self.params['task'] == 'simplified'):
            self.sampler = self.simple_sampler
        if (self.params['task'] == 'scaling'):
            self.sampler = self.scaling_sampler
    
    def simple_sampler(self):
        x = np.random.choice(list(range(1,self.params['base_input_number']+1)), p=self.probs_sampling)
        return(x)

    def scaling_sampler(self):
        x = np.random.choice([1,2,3])
        if x == 1:
            y = np.random.randint(1,5, size=1)[0]
        elif x == 2:
            y = np.random.randint(5,3125, size=1)[0] #y = np.random.randint(5,125, size=1)[0]
        else:
            y = np.random.randint(3125,78125, size=1)[0]
        return(y)

    def step(self, action):
        
        #invalid action penalty
        valid_actions = self.action_masks()
        invalid_penalty = 0
        if valid_actions[action]==False:
            invalid_penalty = 2
        
        
        #this is used in the reward, however I should icluded in the list
        self.prev_pos_x = self.fingerlayer.pos_x
        self.prev_pos_y = self.fingerlayer.pos_y
        
        

        for i in range(self.params['n_frames']-2):
            self.list_previous[i] = self.list_previous[i+1]
            self.list_previous_pos[i] = self.list_previous_pos[i+1]
 
        self.list_previous[self.params['n_frames']-2] = np.stack([self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
        self.list_previous_pos[self.params['n_frames']-2] = self.fingerlayer.pos_y
        
        self.action = copy.deepcopy(self.all_actions_dict[action])
        self.list_of_action.append(self.action)

        
        self.timestep += 1

        if(self.action in self.fingerlayer.actions):
            self.fingerlayer.step(self)

        self.ext_repr.step(self)
        
        reward, self.done, self.solved = self.reward_done_function(self)
        
        #invalid action penalty
        valid_actions = self.action_masks()
        if valid_actions[action]==False:
            reward -= invalid_penalty
        
        self.state = self.update_state()

        if(self.timestep > self.max_episode_length):
            self.done = True
            
        self.info['solved'] += self.solved
        

        return self.state, reward, self.done, self.info

    def update_state_standard(self):
        current = np.stack([self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
        self.state = {"space0": self.addendum_1,  "space1": np.stack(self.list_previous + [current])}
        return self.state

    def update_state_localized(self):
        show = []
        if self.fingerlayer.pos_y == 0:
            finger_padded = np.pad(self.fingerlayer.fingerlayer, ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
            repr_padded = np.pad(self.ext_repr.externalrepresentation, ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
            current = np.stack([finger_padded, repr_padded])
        else:
            current = np.stack([self.fingerlayer.fingerlayer[self.fingerlayer.pos_y-1:self.fingerlayer.pos_y+1, :], self.ext_repr.externalrepresentation[self.fingerlayer.pos_y-1:self.fingerlayer.pos_y+1, :]])

        for index, element in enumerate(self.list_previous):
            if self.list_previous_pos[index] == 0:
                finger_padded = np.pad(element[0], ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
                repr_padded = np.pad(element[1], ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
                show.append(np.stack([finger_padded, repr_padded]))
            else:
                show.append(np.stack([element[0][self.list_previous_pos[index]-1:self.list_previous_pos[index]+1, :], element[1][self.list_previous_pos[index]-1:self.list_previous_pos[index]+1, :]]))
        
        self.state = {"space0": self.addendum_1,  "space1": np.stack(show + [current])}
        return self.state

    def update_state_localized_scaling (self):
        show = []
        if self.fingerlayer.pos_y == 0:
            finger_padded = np.pad(self.fingerlayer.fingerlayer, ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
            repr_padded = np.pad(self.ext_repr.externalrepresentation, ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
            current = np.stack([finger_padded, repr_padded])
        else:
            current = np.stack([self.fingerlayer.fingerlayer[self.fingerlayer.pos_y-1:self.fingerlayer.pos_y+1, :], self.ext_repr.externalrepresentation[self.fingerlayer.pos_y-1:self.fingerlayer.pos_y+1, :]])

        for index, element in enumerate(self.list_previous):
            if self.list_previous_pos[index] == 0:
                finger_padded = np.pad(element[0], ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
                repr_padded = np.pad(element[1], ((1,0),(0,0)), "constant",  constant_values = 0.5)[0:2, : ]
                show.append(np.stack([finger_padded, repr_padded]))
            else:
                show.append(np.stack([element[0][self.list_previous_pos[index]-1:self.list_previous_pos[index]+1, :], element[1][self.list_previous_pos[index]-1:self.list_previous_pos[index]+1, :]]))
        
        if self.sign:
            input_agent = self.params['base_input_number']+1
        else:
            input_agent = self.addendum_1_list_of_digits[self.pos_number_reading]
        
        self.input_stack[1:3] = self.input_stack[0:2]
        self.input_stack[0] = input_agent
        
        
        self.state = {"space0a" : input_agent, "space0b" :self.input_stack[1], "space0c" : self.input_stack[2], "space1": np.stack(show + [current])}
        
        return self.state
       

    def change_n_object(self, number_objects):
        self.max_objects = number_objects

    def render(self, display_id=None):
        return(self.state)


    def reset(self, specific = None, input_number = None): 
        
        self.info = {"solved": 0}
        
        self.input_stack = [self.params['ext_shape'][1], self.params['ext_shape'][1], self.params['ext_shape'][1]]
        
        #specific is a tuple that is ((finger_pos_y, finger_pos_x), ext_obs, list_of_prev_ext_repr, represented_number)
        #if specific is used, than you have also to use input number
        if (specific is None):
            self.addendum_2 = self.sampler()
            self.ext_repr.random_reset(self.addendum_2, self.params['ext_shape'][1])
            #self.ext_repr.reset()
            self.addendum_1 = self.sampler()
            self.n_objects = self.addendum_2 + self.addendum_1
            self.fingerlayer.reset()  
            self.list_previous = []
            self.list_previous_pos = []
            for i in range(self.params['n_frames']-1):
                self.list_previous.append(np.full((2, self.ext_shape[0], self.ext_shape[1]+1),0.5))
                self.list_previous_pos.append(0)
        else:
            self.addendum_2 = specific[3]
            self.ext_repr.externalrepresentation = specific[1]
            #####
            self.fingerlayer.reset(specific[0][0], specific[0][1])        
            self.list_previous = specific[2]
            
        self.max_episode_length = 30
        
        if input_number is not None:
            self.addendum_1 = input_number
            self.n_objects = self.addendum_2 + self.addendum_1
        
        
        self.list_of_digits = numberToBase(self.n_objects, self.params['ext_shape'][1])
        self.list_of_digits.reverse()
        
        self.addendum_1_list_of_digits = numberToBase(self.addendum_1, self.params['ext_shape'][1])
        self.addendum_1_list_of_digits.reverse()
        
        #signpost is the position of the number in "addendum_1_list_of_digits" you are summing
        self.signpost = 0
        self.pos_number_reading = 0
        if not self.params['auto_signpost']:
            self.sign = False
        
        
        # Initialize whole state space: concatenated observation and external representation

        self.state = self.update_state()

        # Initialize other interactions: e.g. 'submit', 'larger'/'smaller,
        self.otherinteractions = OtherInteractions(self, self.params['task'])
        self.action = np.zeros(self.action_dim)
        self.done = False
        self.timestep = 0
        
        self.info['addendum_1'] = self.addendum_1
        self.info['addendum_2'] = self.addendum_2


        #self.counter_that_is_used_for_debbuging = self.counter_that_is_used_for_debbuging+1
        
        if self.counter_that_is_used_for_debbuging%256 == 1:
            print(self.list_of_action)
            self.counter_that_is_used_for_debbuging = 0
        self.list_of_action = []
        self.reward = 0
        
        return self.state
    
    def merge_actions(self, action_dicts):
        """This function creates the actions dict for the complete environment merging the ones related to the individual environment parts.
        """
        self.all_actions_list = []
        self.all_actions_dict = {}
        _n = 0
        for _dict in action_dicts:
            rewritten_individual_dict = {}
            for key,value in _dict.items():
                if(isinstance(value, str) and value not in self.all_actions_list):
                    self.all_actions_list.append(value)
                    self.all_actions_dict[_n] = value
                    rewritten_individual_dict[_n] = value
                    _n += 1
            _dict = rewritten_individual_dict
        #self.all_actions_dict = sorted(self.all_actions_dict.items())
        self.all_actions_list = [value for key, value in self.all_actions_dict.items()]
        return self.all_actions_list, self.all_actions_dict

    def rewrite_all_action_keys(self):
        self.all_actions_dict_inv = dict([reversed(i) for i in self.all_actions_dict.items()])
        int_to_int = {}
        for key, value in self.all_actions_dict_inv.items():
            int_to_int[value] = value
        self.all_actions_dict_inv.update(int_to_int)
        # Rewrite keys of individual action-spaces, so they do not overlap in the global action space
        self.ext_repr.actions = self.rewrite_action_keys(self.ext_repr.actions)
        self.fingerlayer.actions = self.rewrite_action_keys(self.fingerlayer.actions)

    def rewrite_action_keys(self, _dict):
        """Function used to rewrite keys of individual action-spaces, so they do not overlap in the global action space.
        """
        rewritten_dict = {}
        for key, value in _dict.items():
            if(isinstance(key, int)):
                rewritten_dict[self.all_actions_dict_inv[value]] = value
        str_to_str = {}
        for key,value in rewritten_dict.items():
            str_to_str[value] = value
        rewritten_dict.update(str_to_str)
        return rewritten_dict


    def action_masks(self):
        #we assume 6 action, 0 modpoint, 1-4 movement, 5 submit:
        if self.params['auto_signpost']:
            valid_actions_mask = np.full(6, True)
        else:
            valid_actions_mask = np.full(8, True)
            if(self.signpost == self.fingerlayer.max_y):
                valid_actions_mask[5] = False
            if(self.signpost == 0):
                valid_actions_mask[6] = False
        if(self.fingerlayer.pos_y == self.fingerlayer.max_y):
            valid_actions_mask[2] = False
        if(self.fingerlayer.pos_y == 0):
            valid_actions_mask[1] = False
        if(self.fingerlayer.pos_x == 0):
            valid_actions_mask[3] = False
        if (self.fingerlayer.pos_x == self.fingerlayer.max_x):
            valid_actions_mask[4] = False
        if (self.ext_repr.externalrepresentation[self.fingerlayer.pos_y,self.fingerlayer.pos_x] == 0):
            valid_actions_mask[0] = False
        return(valid_actions_mask)


class FingerLayer():
    """
    This class implements the finger movement part of the environment.
    """
    def __init__(self, agent):
        self.ext_shape = agent.ext_shape
        self.second_finger_layer = np.full(self.ext_shape[0], 0.2)
        self.second_finger_layer[0] += 0.7
        self.fingerlayer = np.c_[np.zeros(self.ext_shape), self.second_finger_layer] 
        self.max_y = agent.ext_shape[0]-1
        self.max_x = agent.ext_shape[1]-1
        self.pos_y = 0 #random.randint(0, dim-1)
        self.pos_x = 0 #random.randint(0, dim-1)
        self.fingerlayer[self.pos_y, self.pos_x] = 1
        # This dictionary translates the total action-array to the Finger-action-strings:
        # Key will be overwritten when merged with another action-space
        self.actions = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }
        if(agent.params['finger_auto'] == False):
            self.actions[4] = 's_right'
            self.actions[5] = 's_left'
        # revd=dict([reversed(i) for i in finger_movement.items()])
        # Add each value as key as well. so in the end both integers (original keys) and strings (original values) can be input
        str_to_str = {}
        for key, value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def step(self, agent):
        move_action_str = agent.action
        if(move_action_str=="down"):
            if(self.pos_y<self.max_y):
                self.pos_y = (self.pos_y + 1)
        elif(move_action_str=="up"):
            if(self.pos_y > 0):
                self.pos_y = (self.pos_y - 1)
        elif(move_action_str=="left"):
            if(self.pos_x > 0):
                self.pos_x = (self.pos_x - 1)
        elif(move_action_str=="right"):
            if (self.pos_x < self.max_x):
                self.pos_x = (self.pos_x + 1)
        elif(move_action_str == 's_right'):
            if (agent.signpost < self.max_y):
                agent.signpost += 1
        elif(move_action_str == 's_left'):
            if (agent.signpost > 0):
                agent.signpost -= 1
        signpost_appended = np.full(agent.params['ext_shape'][0], 0.2)
        signpost_appended[agent.signpost] += 0.7
        signpost_appended[agent.signpost-1] += 0.3
        
                                     
        self.fingerlayer = np.c_[np.zeros(self.ext_shape), signpost_appended]
        self.fingerlayer[self.pos_y, self.pos_x] = 1

    def reset(self, pos_y = 0, pos_x = 0):
        self.pos_y = pos_y #random.randint(0, dim-1)
        self.pos_x = pos_x #random.randint(0, dim-1)
        self.fingerlayer = np.c_[np.zeros(self.ext_shape), self.second_finger_layer]
        self.fingerlayer[self.pos_y, self.pos_x] = 1


def choose_external_representation(external_representation_tool, dim, task):
    if(external_representation_tool == 'MoveAndWrite'):
        return MoveAndWrite(dim)
    elif(external_representation_tool == 'WriteCoord'):
        return WriteCoord(dim)
    elif(external_representation_tool == 'Abacus'):
        return Abacus(dim, task)
    else:
        print("No valid 'external repr. tool was given! ")

# Parent Class ExternalTool. Not usefully used so far. See empty fct-declarations
class ExternalTool():
    def __init__(self):
        pass
    def init_externalrepresentation(self):
        pass
    def step(self):
        pass
    def reset(self):
        pass

class MoveAndWrite(ExternalTool):
    """
    This class implements the external representation in the environment.
    """
    def __init__(self, ext_shape):
        self.ext_shape = ext_shape
        self.init_externalrepresentation(ext_shape)
        self.actions = {
            0: 'mod_point',      # Keys will be overwritten when merged with another action-space
        }
        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, ext_shape):
        self.externalrepresentation = np.zeros(ext_shape)

    def draw(self, draw_pixels):
        self.externalrepresentation += draw_pixels

    def step(self, action, agent):
        # This line implements if ext_repr[at_curr_pos]==0 --> set it to 1. if==1 set to 0.
        if(action == 'mod_point'):
            pos_y = agent.fingerlayer.pos_y
            pos_x = agent.fingerlayer.pos_x
            self.externalrepresentation[pos_y, pos_x] = -self.externalrepresentation[pos_y, pos_x] + 1

    def reset(self):
        self.externalrepresentation = np.zeros(self.ext_shape)
        
    def reset_random(self):
        self.externalrepresentation = np.random.randint(2, size=self.ext_shape)



class WriteCoord(ExternalTool):
    def __init__(self, ext_shape):
        self.ext_shape = ext_shape
        self.init_externalrepresentation(ext_shape)
        self.actions = {}
        for coord_x in range(ext_shape[0]):
          for coord_y in range(ext_shape[1]):
            associated_tuple = coord_y+coord_x*ext_shape[0] #ext_shape[0] or ext_shape[1]?
            self.actions[associated_tuple] = "write_on_" + str(coord_x) + "," + str(coord_y)
        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, ext_shape):
        self.externalrepresentation = np.zeros(ext_shape)

    def step(self, action, agent):
        # This line implements if ext_repr[coord]==0 --> set it to 1. if==1 set to 0.
        # Doesn't work after 9*9
        coord_x_int = int(action[-1])
        coord_y_int = int(action[-3])
        self.externalrepresentation[coord_x_int, coord_y_int] = 1 #If you want to be able to delete as well use: -self.externalrepresentation[coord_int, 0] + 1

    def reset(self):
        self.externalrepresentation = np.zeros(self.ext_shape)

class Abacus(ExternalTool):
    """
    This class implements the external representation in the environment.
    """
    def __init__(self, ext_shape, task):
        
        self.ext_shape = ext_shape
        self.externalrepresentation = np.zeros(self.ext_shape)

        self.init_externalrepresentation(self.ext_shape)

        self.actions = {
            0: 'move_and_slide'
        }

        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, ext_shape):
        self.externalrepresentation = np.ones(ext_shape)
        for rowy in range(self.ext_shape[0]):
            self.externalrepresentation[rowy, 0] = 0

    def step(self, agent):

        if(agent.action == 'move_and_slide'):
            #version without encoding
            #self.externalrepresentation[agent.fingerlayer.pos_y, :] = 1
            
            self.externalrepresentation[agent.fingerlayer.pos_y, :-1] = 1
            self.externalrepresentation[agent.fingerlayer.pos_y,agent.fingerlayer.pos_x] = 0
            

    def reset(self):
        self.externalrepresentation = np.c_[np.ones(self.ext_shape), np.tile([0.1, 0.4, 0.7], self.ext_shape[0]//3 + 1)[0:self.ext_shape[0]]]
        for rowy in range(self.ext_shape[0]):
            self.externalrepresentation[rowy, 0] = 0
    
    def random_reset(self, number, base):
        list_of_digits_addendum = numberToBase(number, base)
        list_of_digits_addendum.reverse()
        for i in range(self.ext_shape[0]):
            list_of_digits_addendum.append(0)
        self.externalrepresentation = np.c_[np.ones(self.ext_shape), np.tile([0.1, 0.4, 0.7], self.ext_shape[0]//3 + 1)[0:self.ext_shape[0]]]
        for rowy in range(self.ext_shape[0]):
            self.externalrepresentation[rowy, list_of_digits_addendum[rowy]] = 0


class OtherInteractions():
    """
    This class implements the environmental responses to actions related to communication with the other agent ('submit') or to the communication of the final answer ('larger', 'smaller').
    """
    def __init__(self, agent, task='comparison', max_n=1):
        # Define task-dependent actions. # Keys will be overwritten when merged with another action-space
        if(task == 'compare'):
            self.actions = {
                0: 'submit',
                1: 'larger',
                2: 'smaller',
                3: 'equal',
             }
        elif(task == 'preset'):
            self.actions = {
             }
        elif (task == 'classify'):
            self.actions = {i: str(i) for i in range(0, max_n+1)}
            self.actions[max_n + 1] = 'wait'
            if(agent.params['IsSubmitButton']):
                self.actions[max_n + 2] = 'submit'

        elif (task == 'reproduce'):
            self.actions = {
                1: '1',
            }
        elif (task == "sum_predetermined") or (task == 'simplified') or (task == 'scaling') :
            self.actions = {}
            if(agent.params['IsSubmitButton']):
              self.actions[0] = 'submit'
        elif (task == "fixed_repr"):
            self.actions = {
                1: "wait"
                }
            if(agent.params['IsSubmitButton']):
                self.actions[2] = 'submit'
        else:
            print("No valid 'task' given")

        # Add each value as key as well. so in the end both integers (original keys) and strings (original values) can be input
        str_to_str = {}
        for key, value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def step(self, action, agent):
        pass




def reward_done_function_predetermined(agent):
    info = False
    done = False
    reward = 0
    
    i,j = np.where((agent.ext_repr.externalrepresentation == 0)) #silly reminder for me, i is the row, j is the columm

    partial_addendum = 0
    for index, el in enumerate(agent.addendum_1_list_of_digits):
        partial_addendum += (agent.params['ext_shape'][1]**index)*el
        if index == agent.signpost:
            break
    
    partial_addendum += agent.addendum_2
    partial_list_digit = numberToBase(partial_addendum, agent.params['ext_shape'][1])
    partial_list_digit.reverse()
    
    if (agent.action == 'move_and_slide'):
        agent.max_episode_length = agent.max_episode_length+10
        
        if (len(partial_list_digit) <= agent.fingerlayer.pos_y):
            done = True
        else:    

            flag = True
            for index in range(agent.fingerlayer.pos_y + 1):
                if partial_list_digit[index] != j[index]:
                    flag = False
                    break

            if flag:
                reward += 1
            else:
                done = True

    elif (agent.action == 'submit'):
        flag = True
        different = -1
        for index, el in enumerate(partial_list_digit):
            if partial_list_digit[index] != j[index]:
                flag = False
                different = index
                break
            
        if flag:
            reward += 1
            if agent.signpost+1 == len(agent.addendum_1_list_of_digits):
                agent.addendum_1 = agent.sampler()
                agent.addendum_2 = agent.n_objects
                agent.n_objects = agent.addendum_2 + agent.addendum_1
                
                agent.list_of_digits = numberToBase(agent.n_objects, agent.params['ext_shape'][1])
                agent.list_of_digits.reverse()
                
                agent.addendum_1_list_of_digits = numberToBase(agent.addendum_1, agent.params['ext_shape'][1])
                agent.addendum_1_list_of_digits.reverse()
                
                agent.signpost = 0
                agent.ext_repr.externalrepresentation[:, -1] =  np.tile([0.1, 0.4, 0.7], agent.params['ext_shape'][0]//3 + 1)[0:agent.params['ext_shape'][0]]
                agent.ext_repr.externalrepresentation[agent.signpost, -1] += 0.2
                agent.ext_repr.externalrepresentation[agent.signpost-1, -1] += 0.1
                
                agent.max_episode_length = agent.max_episode_length+30
                info = True
                
                if len(agent.list_of_digits) > agent.params['ext_shape'][0]:
                    reward += 50
                    done = True
            else:
                agent.signpost += 1
                agent.ext_repr.externalrepresentation[:, -1] = np.tile([0.1, 0.4, 0.7], agent.params['ext_shape'][0]//3 + 1)[0:agent.params['ext_shape'][0]]
                agent.ext_repr.externalrepresentation[agent.signpost, -1] = 0.2
                agent.ext_repr.externalrepresentation[agent.signpost-1, -1] = 0.1
                #agent.fingerlayer.fingerlayer[:, -1] = 0.1
                #agent.fingerlayer.fingerlayer[agent.signpost, -1] = 0.75
                #agent.fingerlayer.fingerlayer[agent.signpost-1, -1] = 0.5
        else:
            done = True   
            
    else: #if the action was a movement
    
        different = agent.signpost
        for index, el in enumerate(partial_list_digit):
            if partial_list_digit[index] != j[index]:
                different = index
                break

        if agent.prev_pos_y < agent.fingerlayer.pos_y: #moved down
           if different < agent.fingerlayer.pos_y: 
               reward -= 1
           else:
               reward += 0.1
        elif agent.prev_pos_y > agent.fingerlayer.pos_y: #moved up
           if different <= agent.fingerlayer.pos_y:
               reward += 0.1
           else:
               reward -= 1
    
        if partial_list_digit[0] != j[0]: #if it the first unit, we check also x axis
            if agent.prev_pos_x > agent.fingerlayer.pos_x:
                if agent.prev_pos_x > agent.list_of_digits[0]:
                    reward += 0.1
                else:
                    reward -= 1
            if agent.prev_pos_x < agent.fingerlayer.pos_x:
                if agent.prev_pos_x < agent.list_of_digits[0]:
                     reward += 0.1
                else:
                     reward -= 1
    return reward, done, info


"""
(When applicable) Rewards are accompanied by their corrispective penalty in case the agent does the opposite
first_colums: bool:
	True: gives reward for moving up/down correctly, for moving left/right correctly and for move_and_slide correctly. Works regardless other parameters
	False: gives reward as for the other columns
move_and_slide_rew: bool:
	True: gives reward for doing the correct move_and_slide
inside_columns: bool:
    True: gives reward for doing the correct up/down
through_columns: bool:
	True: gives reward for doing the correct left/right
auto_reset: bool:
	True: the position of the finger automatically moves the signpost after a submit
	False: the finger position stay invariate
auto_signpost: bool:
	True: the signpost automoves correctly: moves once right after a submit when in inside a number, moves at the start of the abacus after finishing the number
	False: the agent can move left/right the signpost
signpost_reward: bool:
	True: Works only if auto_signpost is False, gives reward if the agent correctly moves 
"""


def reward_done_function_custom(agent):
    info = False
    done = False
    reward = agent.params['base_reward']
    
    i,j = np.where((agent.ext_repr.externalrepresentation == 0)) #silly reminder for me, i is the row, j is the columm


    partial_addendum = 0
    for index, el in enumerate(agent.addendum_1_list_of_digits):
        partial_addendum += (agent.params['ext_shape'][1]**index)*el
        if index == agent.pos_number_reading:
            break
    
    partial_addendum += agent.addendum_2
    partial_list_digit = numberToBase(partial_addendum, agent.params['ext_shape'][1])
    partial_list_digit.reverse()

        
    if(agent.action == 'move_and_slide'):
        agent.max_episode_length = agent.max_episode_length+10
        
        if (len(partial_list_digit) <= agent.fingerlayer.pos_y):
            done = True
        else:    
            flag = True
            for index in range(agent.fingerlayer.pos_y + 1):
                if partial_list_digit[index] != j[index]:
                    flag = False
                    break

            if flag and (agent.signpost == agent.pos_number_reading):
                if agent.params["move_and_slide_rew"] or (agent.params["first_colums"] and agent.fingerlayer.pos_y == 0):
                    reward += 1
            else:
                done = True

    elif (agent.action == 'submit'):
        flag = True
        different = -1
        for index, el in enumerate(partial_list_digit):
            if partial_list_digit[index] != j[index]:
                flag = False
                different = index
                break
            
        if (flag or agent.sign) and (agent.pos_number_reading == agent.signpost):
            reward += 1
            if agent.pos_number_reading+1 == len(agent.addendum_1_list_of_digits) or agent.sign:
                agent.pos_number_reading = 0
                if (not agent.sign) and (not agent.params['auto_signpost']):
                    agent.sign = True
                else:
                    agent.sign = False
                    agent.addendum_1 = agent.sampler()
                    agent.addendum_2 = agent.n_objects
                    agent.n_objects = agent.addendum_2 + agent.addendum_1
                    
                    agent.list_of_digits = numberToBase(agent.n_objects, agent.params['ext_shape'][1])
                    agent.list_of_digits.reverse()
                    
                    agent.addendum_1_list_of_digits = numberToBase(agent.addendum_1, agent.params['ext_shape'][1])
                    agent.addendum_1_list_of_digits.reverse()
                    
                    agent.max_episode_length = agent.max_episode_length+30
                    info = True
                    
                    if len(agent.list_of_digits) > agent.params['ext_shape'][0]:
                        reward += 50
                        done = True
                    
                    if agent.params['auto_signpost']:
                        agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    if agent.params['auto_reset']:
                        agent.fingerlayer.pos_y = 0

            else:
                agent.pos_number_reading +=1
                if agent.params['auto_signpost']:
                    agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    
                
        else:
            reward -= 1
            done = True
    
    elif agent.action == "s_right" and agent.params['signpost_reward']:
        if agent.signpost <= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    elif agent.action == "s_left" and agent.params['signpost_reward']:
        if agent.signpost >= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    else: #if the action was a movement
        if not agent.sign:
            different = agent.pos_number_reading
            for index, el in enumerate(partial_list_digit):
                if partial_list_digit[index] != j[index]:
                    different = index
                    break
        else:
            different = 0
        if (agent.action == 'down'):
           if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
               if different < agent.fingerlayer.pos_y:
                   reward -= 1
               else:
                   reward += 0.1
        elif (agent.action == 'up'):
            if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
                if different <= agent.fingerlayer.pos_y:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'left'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x > partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'right'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x < partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
    return reward, done, info



class TestEnv(SingleRLAgent):
    def __init__(self, agent_params, number_of_digits = 8):
        super().__init__(agent_params)
        self.do_infjection = False
        self.reward_done_function = reward_pair
        if self.do_infjection:
            self.inject_1 = 0
            self.inject_2 = 0
            self.injected_1 = False
            self.sampler = self.extract_with_fixed_len
        else:
            self.number_of_digits = number_of_digits
            self.sampler = self.sampler_with_chosen_len
        
    def extract_with_fixed_len(self):
        if not self.injected_1:
            y = self.inject_1
            self.injected_1 = True
        else:
            y = self.inject_2
            self.injected_1 = False   
        return(y)
    
    def sampler_with_chosen_len(self):
        y = np.random.randint(5**(self.number_of_digits-1),5**self.number_of_digits, size=1, dtype=np.int64)[0]
        return(y)
        
def reward_pair(agent): #used in Test_env
    info = False
    done = False
    reward = agent.params['base_reward']
    
    i,j = np.where((agent.ext_repr.externalrepresentation == 0)) #silly reminder for me, i is the row, j is the columm


    partial_addendum = 0
    for index, el in enumerate(agent.addendum_1_list_of_digits):
        partial_addendum += (agent.params['ext_shape'][1]**index)*el
        if index == agent.pos_number_reading:
            break
    
    partial_addendum += agent.addendum_2
    partial_list_digit = numberToBase(partial_addendum, agent.params['ext_shape'][1])
    partial_list_digit.reverse()

        
    if(agent.action == 'move_and_slide'):
        agent.max_episode_length = agent.max_episode_length+10
        
        if (len(partial_list_digit) <= agent.fingerlayer.pos_y):
            done = True
        else:    
            flag = True
            for index in range(agent.fingerlayer.pos_y + 1):
                if partial_list_digit[index] != j[index]:
                    flag = False
                    break

            if flag and (agent.signpost == agent.pos_number_reading):
                if agent.params["move_and_slide_rew"] or (agent.params["first_colums"] and agent.fingerlayer.pos_y == 0):
                    reward += 1
            else:
                done = True

    elif (agent.action == 'submit'):
        flag = True
        different = -1
        for index, el in enumerate(partial_list_digit):
            if partial_list_digit[index] != j[index]:
                flag = False
                different = index
                break
            
        if (flag or agent.sign) and (agent.pos_number_reading == agent.signpost):
            reward += 1
            if agent.pos_number_reading+1 == len(agent.addendum_1_list_of_digits) or agent.sign:
                agent.pos_number_reading = 0
                if (not agent.sign) and (not agent.params['auto_signpost']):
                    agent.sign = True 
                else:
                    agent.sign = False
                    agent.addendum_1 = agent.sampler()
                    agent.addendum_2 = agent.n_objects
                    agent.n_objects = agent.addendum_2 + agent.addendum_1
                    
                    agent.list_of_digits = numberToBase(agent.n_objects, agent.params['ext_shape'][1])
                    agent.list_of_digits.reverse()
                    
                    agent.addendum_1_list_of_digits = numberToBase(agent.addendum_1, agent.params['ext_shape'][1])
                    agent.addendum_1_list_of_digits.reverse()
                    
                    agent.max_episode_length = agent.max_episode_length+30
                    info = True
                    done = True

                    if len(agent.list_of_digits) > agent.params['ext_shape'][0]:
                        reward += 50
                        done = True
                    
                    if agent.params['auto_signpost']:
                        agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    if agent.params['auto_reset']:
                        agent.fingerlayer.pos_y = 0

            else:
                agent.pos_number_reading +=1
                if agent.params['auto_signpost']:
                    agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    
                
        else:
            reward -= 1
            done = True
    
    elif agent.action == "s_right" and agent.params['signpost_reward']:
        if agent.signpost <= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    elif agent.action == "s_left" and agent.params['signpost_reward']:
        if agent.signpost >= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    else: #if the action was a movement
        if not agent.sign:
            different = agent.pos_number_reading
            for index, el in enumerate(partial_list_digit):
                if partial_list_digit[index] != j[index]:
                    different = index
                    break
        else:
            different = 0
        if (agent.action == 'down'):
           if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
               if different < agent.fingerlayer.pos_y:
                   reward -= 1
               else:
                   reward += 0.1
        elif (agent.action == 'up'):
            if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
                if different <= agent.fingerlayer.pos_y:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'left'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x > partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'right'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x < partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
    return reward, done, info 


class Both_operation(SingleRLAgent): 
    def __init__(self, agent_params, number_of_digits = 8):
        super().__init__(agent_params)
        self.reward_done_function = both_reward
        
        if(self.params['task'] == 'scaling'):
            self.update_state = self.update_state_localized_scaling
            
            if self.params['auto_signpost']:
                space_len = self.params['ext_shape'][1]
            else:
                space_len = self.params['ext_shape'][1]+1

            pre_dictionary_spaces = {"sign_space": spaces.Discrete(2), "space0a" : spaces.Discrete(space_len), "space0b" : spaces.Discrete(space_len), "space0c" : spaces.Discrete(space_len),
                                     "space1" : spaces.Box(low=0, high=1,shape=(self.params['n_frames'], 2, 2, self.params['ext_shape'][1]+1), dtype=np.float64)}
            self.observation_space = spaces.Dict(pre_dictionary_spaces)
            
            
            self.accumulated_accuracy = 0
            self.total_accumulation = 0
            self.info = {"solved": 0}
            self.reset_accumulation = False

            
    def reset(self): 
        
        #self.counter_that_is_used_for_debbuging = self.counter_that_is_used_for_debbuging+1
        if self.counter_that_is_used_for_debbuging%256 == 1:
            print(self.list_of_action)
            print(self.info)
            self.counter_that_is_used_for_debbuging = 0
        self.list_of_action = []
        
        if self.reset_accumulation:
            self.accumulated_accuracy = 0
            self.total_accumulation = 0
            self.reset_accumulation = False
            

        self.accumulated_accuracy += self.info["solved"]
        self.total_accumulation += 1
        
        self.info = {"solved": 0}
        self.max_episode_length = 30
        
        self.input_stack = [self.params['ext_shape'][1], self.params['ext_shape'][1], self.params['ext_shape'][1]]
        
        self.addendum_2 = self.sampler()
        self.ext_repr.random_reset(self.addendum_2, self.params['ext_shape'][1])
        
        self.addendum_1 = self.sampler()

        if self.addendum_1 >  self.addendum_2:
            self.is_addition = 1
        else:
            self.is_addition = np.random.randint(0, 2, size=1)[0]

        if self.is_addition:
            self.n_objects = self.addendum_2 + self.addendum_1
        else:
            self.n_objects = self.addendum_2 - self.addendum_1


        
        self.fingerlayer.reset()  
        self.list_previous = []
        self.list_previous_pos = []
        for i in range(self.params['n_frames']-1):
            self.list_previous.append(np.full((2, self.ext_shape[0], self.ext_shape[1]+1),0.5))
            self.list_previous_pos.append(0)

        
        
        self.list_of_digits = numberToBase(self.n_objects, self.params['ext_shape'][1])
        self.list_of_digits.reverse()
        
        self.addendum_1_list_of_digits = numberToBase(self.addendum_1, self.params['ext_shape'][1])
        self.addendum_1_list_of_digits.reverse()
        
        #signpost is the position of the number in "addendum_1_list_of_digits" you are summing
        self.signpost = 0
        self.pos_number_reading = 0
        if not self.params['auto_signpost']:
            self.sign = False
        
        
        # Initialize whole state space: concatenated observation and external representation

        self.state = self.update_state()

        # Initialize other interactions: e.g. 'submit', 'larger'/'smaller,
        self.otherinteractions = OtherInteractions(self, self.params['task'])
        self.action = np.zeros(self.action_dim)
        self.done = False
        self.timestep = 0
        
        self.info['addendum_1'] = self.addendum_1
        self.info['addendum_2'] = self.addendum_2

        self.reward = 0 #uselss
        
        return self.state
    
    def update_state_localized_scaling(self):
        state = super().update_state_localized_scaling()
        state['sign_space'] = self.is_addition
        return(state)


def both_reward(agent):
    info = False
    done = False
    reward = agent.params['base_reward']
    
    i,j = np.where((agent.ext_repr.externalrepresentation == 0)) 
    
    #should be moved in reset() and in step -> submit check, no reason to recalc it everytime
    partial_addendum = 0
    for index, el in enumerate(agent.addendum_1_list_of_digits):
        partial_addendum += (agent.params['ext_shape'][1]**index)*el
        if index == agent.pos_number_reading:
            break
        
    if not agent.is_addition:
        partial_addendum = agent.addendum_2 - partial_addendum
    else:
        partial_addendum += agent.addendum_2
    partial_list_digit = numberToBase(partial_addendum, agent.params['ext_shape'][1])
    partial_list_digit.reverse()
    partial_list_digit.append(0)
    ##############
    
    if(agent.action == 'move_and_slide'):
        #agent.max_episode_length = agent.max_episode_length+10
        flag = True
        for index in range(agent.fingerlayer.pos_y + 1):
            if len(partial_list_digit) <= index:
                if j[index] != 0:
                    flag = False
                    break
            else:
                if partial_list_digit[index] != j[index]:
                    flag = False
                    break

        if flag and (agent.signpost == agent.pos_number_reading):
            if agent.params["move_and_slide_rew"] or (agent.params["first_colums"] and agent.fingerlayer.pos_y == 0):
                reward += 1
        else:
            done = True

    elif (agent.action == 'submit'):
        flag = True
        different = -1
        for index, el in enumerate(partial_list_digit):
            if partial_list_digit[index] != j[index]:
                flag = False
                different = index
                break 
        if (flag or agent.sign) and (agent.pos_number_reading == agent.signpost):
            reward += 1
            if agent.pos_number_reading+1 == len(agent.addendum_1_list_of_digits) or agent.sign:
                agent.pos_number_reading = 0
                if (not agent.sign) and (not agent.params['auto_signpost']):
                    agent.sign = True 
                else:
                    agent.sign = False
                    agent.addendum_1 = agent.sampler()
                    agent.addendum_2 = agent.n_objects
                    if agent.addendum_1 > agent.addendum_2:
                        agent.is_addition = 1
                    else:
                        agent.is_addition = np.random.randint(0, 2, size=1)[0]
                    
                    if agent.is_addition:
                        agent.n_objects = agent.addendum_2 + agent.addendum_1
                    else:
                        agent.n_objects = agent.addendum_2 - agent.addendum_1

                    agent.info['addendum_1'] = agent.addendum_1
                    agent.info['addendum_2'] = agent.addendum_2
                    
                    agent.list_of_digits = numberToBase(agent.n_objects, agent.params['ext_shape'][1])
                    agent.list_of_digits.reverse()
                    
                    agent.addendum_1_list_of_digits = numberToBase(agent.addendum_1, agent.params['ext_shape'][1])
                    agent.addendum_1_list_of_digits.reverse()
                    
                    agent.max_episode_length = agent.max_episode_length+50
                    info = True

                    if len(agent.list_of_digits) >= (agent.params['ext_shape'][0]-2):
                        reward += 50
                        done = True
                    
                    if agent.params['auto_signpost']:
                        agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    if agent.params['auto_reset']:
                        agent.fingerlayer.pos_y = 0

            else:
                agent.pos_number_reading +=1
                if agent.params['auto_signpost']:
                    agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    
                
        else:
            reward -= 1
            done = True
    
    elif agent.action == "s_right" and agent.params['signpost_reward']:
        if agent.signpost <= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    elif agent.action == "s_left" and agent.params['signpost_reward']:
        if agent.signpost >= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    else: #if the action was a movement
        if not agent.sign:
            different = agent.pos_number_reading
            for index, el in enumerate(partial_list_digit):
                if partial_list_digit[index] != j[index]:
                    different = index
                    break
        else:
            different = 0
        if (agent.action == 'down'):
           if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
               if different < agent.fingerlayer.pos_y:
                   reward -= 1
               else:
                   reward += 0.1
        elif (agent.action == 'up'):
            if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
                if different <= agent.fingerlayer.pos_y:
                    reward += 0.1
                elif agent.addendum_1_list_of_digits[agent.pos_number_reading]==0 and agent.signpost < agent.fingerlayer.pos_y:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'left'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x > partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'right'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x < partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
    return reward, done, info 



class TestEnv_Both(Both_operation):
    def __init__(self, agent_params, number_of_digits = 8):
        super().__init__(agent_params)
        self.do_infjection = False
        self.reward_done_function = reward_pair_both
        if self.do_infjection:
            self.inject_1 = 0
            self.inject_2 = 0
            self.injected_1 = False
            self.sampler = self.extract_with_fixed_len
        else:
            self.number_of_digits = number_of_digits
            self.sampler = self.sampler_with_chosen_len
        
    def extract_with_fixed_len(self):
        if not self.injected_1:
            y = self.inject_1
            self.injected_1 = True
        else:
            y = self.inject_2
            self.injected_1 = False   
        return(y)
    
    def sampler_with_chosen_len(self):
        y = np.random.randint(5**(self.number_of_digits-1),5**self.number_of_digits, size=1, dtype=np.int64)[0]
        return(y)
        
def reward_pair_both(agent): #used in Test_env
    info = False
    done = False
    reward = agent.params['base_reward']
    
    i,j = np.where((agent.ext_repr.externalrepresentation == 0)) 
    
    #should be moved in reset() and in step -> submit check, no reason to recalc it everytime
    partial_addendum = 0
    for index, el in enumerate(agent.addendum_1_list_of_digits):
        partial_addendum += (agent.params['ext_shape'][1]**index)*el
        if index == agent.pos_number_reading:
            break
        
    if not agent.is_addition:
        partial_addendum = agent.addendum_2 - partial_addendum
    else:
        partial_addendum += agent.addendum_2
    partial_list_digit = numberToBase(partial_addendum, agent.params['ext_shape'][1])
    partial_list_digit.reverse()
    partial_list_digit.append(0)
    ##############
    
    if(agent.action == 'move_and_slide'):
        agent.max_episode_length = agent.max_episode_length+10
        flag = True
        for index in range(agent.fingerlayer.pos_y + 1):
            if len(partial_list_digit) <= index:
                if j[index] != 0:
                    flag = False
                    break
            else:
                if partial_list_digit[index] != j[index]:
                    flag = False
                    break

        if flag and (agent.signpost == agent.pos_number_reading):
            if agent.params["move_and_slide_rew"] or (agent.params["first_colums"] and agent.fingerlayer.pos_y == 0):
                reward += 1
        else:
            done = True

    elif (agent.action == 'submit'):
        flag = True
        different = -1
        for index, el in enumerate(partial_list_digit):
            if partial_list_digit[index] != j[index]:
                flag = False
                different = index
                break 
        if (flag or agent.sign) and (agent.pos_number_reading == agent.signpost):
            reward += 1
            if agent.pos_number_reading+1 == len(agent.addendum_1_list_of_digits) or agent.sign:
                agent.pos_number_reading = 0
                if (not agent.sign) and (not agent.params['auto_signpost']):
                    agent.sign = True 
                else:
                    agent.sign = False
                    agent.addendum_1 = agent.sampler()
                    agent.addendum_2 = agent.n_objects
                    if agent.addendum_1 > agent.addendum_2:
                        agent.is_addition = 1
                    else:
                        agent.is_addition = np.random.randint(0, 2, size=1)[0]
                    
                    if agent.is_addition:
                        agent.n_objects = agent.addendum_2 + agent.addendum_1
                    else:
                        agent.n_objects = agent.addendum_2 - agent.addendum_1

                    agent.info['addendum_1'] = agent.addendum_1
                    agent.info['addendum_2'] = agent.addendum_2
                    
                    agent.list_of_digits = numberToBase(agent.n_objects, agent.params['ext_shape'][1])
                    agent.list_of_digits.reverse()
                    
                    agent.addendum_1_list_of_digits = numberToBase(agent.addendum_1, agent.params['ext_shape'][1])
                    agent.addendum_1_list_of_digits.reverse()
                    
                    agent.max_episode_length = agent.max_episode_length+30
                    info = True
                    done = True

                    if len(agent.list_of_digits) >= (agent.params['ext_shape'][0]-2): 
                        done = True
                    if agent.timestep > 50000:
                        done = True
                    
                    if agent.params['auto_signpost']:
                        agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                        agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    if agent.params['auto_reset']:
                        agent.fingerlayer.pos_y = 0

            else:
                agent.pos_number_reading +=1
                if agent.params['auto_signpost']:
                    agent.fingerlayer.fingerlayer[:, -1] = np.full(agent.params['ext_shape'][0], 0.2)
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading, -1] += 0.7
                    agent.fingerlayer.fingerlayer[agent.pos_number_reading-1, -1] += 0.3
                    
                
        else:
            reward -= 1
            done = True
    
    elif agent.action == "s_right" and agent.params['signpost_reward']:
        if agent.signpost <= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    elif agent.action == "s_left" and agent.params['signpost_reward']:
        if agent.signpost >= agent.pos_number_reading:
            reward += 0.1
        else:
            reward -= 1
            done = True
    else: #if the action was a movement
        if not agent.sign:
            different = agent.pos_number_reading
            for index, el in enumerate(partial_list_digit):
                if partial_list_digit[index] != j[index]:
                    different = index
                    break
        else:
            different = 0
        if (agent.action == 'down'):
           if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
               if different < agent.fingerlayer.pos_y:
                   reward -= 1
               else:
                   reward += 0.1
        elif (agent.action == 'up'):
            if agent.params['through_columns'] or (agent.params["first_colums"] and different == 0):
                if different <= agent.fingerlayer.pos_y:
                    reward += 0.1
                elif agent.addendum_1_list_of_digits[agent.pos_number_reading]==0 and agent.signpost < agent.fingerlayer.pos_y:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'left'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x > partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
        elif (agent.action == 'right'):
            if (agent.params['inside_columns'] or (agent.params["first_colums"] and different == 0)) and (len(partial_list_digit) > different ):
                if agent.prev_pos_x < partial_list_digit[different]:
                    reward += 0.1
                else:
                    reward -= 1
    return reward, done, info 