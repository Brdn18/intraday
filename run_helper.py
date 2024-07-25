import gym
from gym import spaces
import numpy as np
from collections import OrderedDict


class FlattenDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenDictWrapper, self).__init__(env)
        # Get the dictionary observation space
        self.dict_obs_space = self.observation_space
        assert isinstance(self.dict_obs_space, spaces.Dict), "Observation space must be of type gym.spaces.Dict"
        
        # Calculate the total size of the flattened observation
        flattened_size = 0
        self.flat_obs_keys = []
        for key, space in self.dict_obs_space.spaces.items():
            assert isinstance(space, spaces.Box), "Only Box spaces are supported"
            flattened_size += np.prod(space.shape)
            self.flat_obs_keys.append((key, np.prod(space.shape)))

        # Define the new observation space as a single Box space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32)

    def observation(self, observation):
        # Flatten the dictionary observation into a single array

#        flat_obs = np.concatenate([observation[key].flatten() for key, size in self.flat_obs_keys])
        flat_obs = np.concatenate([np.array([observation[key]]).flatten() if np.isscalar(observation[key]) \
                                   else observation[key].flatten() for key, size in self.flat_obs_keys], dtype='float32')
        return flat_obs
    
    def unwrap(self, observation):
        out = OrderedDict()
        print("observation", self.flat_obs_keys, observation)
        for i, k in enumerate(self.flat_obs_keys):
            print(k[0], i, observation[i])
            out[k[0]] = observation[i]

        return out
