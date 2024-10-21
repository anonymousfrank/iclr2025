import os
import sys
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import cv2
from math import sqrt
import numpy as np
import torch as th
from torch import nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import gym as gym26
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize, VecTransposeImage, is_vecenv_wrapped
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

import matplotlib.pyplot as plt


class SelfAttnCNNPPO(BaseFeaturesExtractor):
    """
    Define our custom feature extractor (note: this is only the state representation block of the Actor-Critic network)
    For the CNN network here, we are following the PPO paper's setup.
    Note that the PPO paper follows the network architecture used in the both papers below.
    “Asynchronous methods for deep reinforcement learning”. In: arXiv preprint arXiv:1602.01783 (2016).
    "Playing atari with deep reinforcement learning". In NIPS Deep Learning Workshop. 2013.

    :param observation_space: (gym.Space) gym.spaces.Box
        This should be in the format of channel-first since the observation is an image for Atari envs.
    :param features_dim: (int) Number of features extracted (e.g. features from a CNN). default: 256 (follows the PPO paper)
        This corresponds to the number of unit for the last layer of the feature extractor.
        The last layer of feature extractor is also the first layer (input layer) of the net_arch block.
        We can also change the hidden layers of the net_arch by passing the net_arch parameter to policy_kwargs. 
        net_arch specifies a (fully-connected) network that maps the features to actions/value. Its architecture is controlled by the ``net_arch`` parameter.
        For more details, please visit https://github.com/DLR-RM/stable-baselines3/blob/378d197b00938579a6a5e04c739f41fec23fd805/docs/guide/custom_policy.rst
    :param self_attn: (str) Type of self-attention layer, default: 'NA' = No-Self-Attention. 
    :param ada_attn: (str) Type of adaptive attention configuration, default: None
    :param norm_softmax: (bool) Flag for whether normalization and softmax will be performed. default: True
        Here we follow the standard "Transformer"'s' self-attention definition. 
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=256, self_attn: str='NA', ada_attn: str=None, norm_softmax: bool=True):
        super(SelfAttnCNNPPO, self).__init__(observation_space, features_dim)
        self.observation_space = observation_space
        self.n_input_channels = observation_space.shape[0] # this should be 4 if frame_stack is 4
        # self.features_dim = features_dim # this line causes "Can't set attribute" error
        self.self_attn = self_attn
        self.ada_attn = ada_attn
        self.norm_softmax = norm_softmax
        # self_attn type must not be None, pick one from ['NA', 'SWA', 'CWRA', 'CWCA', 'CWRCA']
        if self.self_attn is not None:
            if self.self_attn != 'NA':
                print("{} self-attention layer will be added.".format(self.self_attn))
                # initialize the given self-attention layer 
                self.self_attn_layer = MultiHeadAttention(size=16, self_attn=self.self_attn, ada_attn=self.ada_attn, norm_softmax=self.norm_softmax)
            else:
                print("No self-attention layer will be added.")
            # initialize all CNN layers
            self.c1 = nn.Conv2d(self.n_input_channels, 16, 8, stride=4) # default padding=0, output feature map size=(84-8)/4 + 1=20
            self.c2 = nn.Conv2d(16, 32, 4, stride=2) # output feature map size=(20-4)/2 + 1=9
            # get flattened layer size 
            self.n_flatten = self._get_flatten_size(self.observation_space.shape) # this should return 2592
            # initialize the first linear layer (i.e., the last layer of the feature extractor)
            self.l1 = nn.Linear(self.n_flatten, features_dim) # 2592*256
        else:
            sys.exit("self_attn argument must not be None! Please choose one from ['NA', 'SWA', 'CWRA', 'CWCA', 'CWRCA']")

    def custom_cnn_block(self, observations: th.Tensor) -> th.Tensor:
        h = F.relu(self.c1(observations)) # h's dimension: (batch, channel, height, width)
        # add self-attn layer between c1 and c2 based on the type of self-attention layer (only do this when self.self_attn is not 'NA') 
        if self.self_attn != 'NA':
            h = self.self_attn_layer(h) # pass h to 3 1x1 cnn kernels to generate q (query), k (key), v (value)
        h = F.relu(self.c2(h)) # h is of size (1,32,9,9) where n_flatten = 1*32*9*9=2592
        return h

    def _get_flatten_size(self, obs_shape: Tuple) -> int:
        with th.no_grad():
            # Compute flattened layer size by doing one forward pass through the custom_cnn_block 
            x = th.zeros(1, *obs_shape) # x is a zero tensor of size (1,4,84,84)
            h = self.custom_cnn_block(x) # h is of size (1,32,9,9) 
            return int(np.prod(h.size())) # np.prod([1,32,9,9]) = 2592

    def custom_feature_extractor(self, observations: th.Tensor) -> th.Tensor:
        h = self.custom_cnn_block(observations)
        h = h.reshape(-1).view(-1, self.n_flatten) # reshape(-1) converts h to a 1-D vector (row-major), view(-1,2592) converts a 1-D vector to a 2-D tensor (matrix) 
        h = F.relu(self.l1(h))
        return h

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.custom_feature_extractor(observations)


class MultiHeadAttention(nn.Module):
    """
    Define the multi-head attention block (we define 4 types of self-attention operations based on the feature dimensions selected for performing self-attention upon)
        'SWA': spatial-wise self-attention
        'CWRA': channel-wise-row self-attention
        'CWCA': channel-wise-column self-attention 
        'CWRCA': channel-wise-row-column self-attention
    """
    def __init__(self, size=16, self_attn=None, ada_attn=None, norm_softmax=True):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        assert self.self_attn is not None, "self_attn type cannot be None in multi-head attention block!"
        self.ada_attn = ada_attn
        self.norm_softmax = norm_softmax

        # initialize 3 1x1 CNN kernels for q,k,v generation 
        self.w_qs = nn.Conv2d(size, size, 1)
        self.w_ks = nn.Conv2d(size, size, 1)
        self.w_vs = nn.Conv2d(size, size, 1)

        # initialize self attn layer based on the type of self-attention operation
        if self.self_attn == 'SWA':
            self.self_attention = SpatialWiseAttention(require_scaling=self.norm_softmax)
        if self.self_attn == 'CWRA':
            self.self_attention = ChannelWiseRowAttention(require_scaling=self.norm_softmax)
        if self.self_attn == 'CWCA':
            self.self_attention = ChannelWiseColumnAttention(require_scaling=self.norm_softmax)
        if self.self_attn == 'CWRCA':
            self.self_attention = ChannelWiseRowColumnAttention(require_scaling=self.norm_softmax)

        # initialize adaptive attn coefficients regardless of user input (we will check user input in forward() function)
        self.alpha = th.nn.Parameter(th.zeros(1)) # initialize alpha = 0
        self.beta = th.nn.Parameter(th.ones(1)) # initialize beta = 1

    def forward(self, h: th.Tensor) -> th.Tensor:
        # print("multi-head attention forward starts")
        cnn_features = h # h = c1's output tensors with shape = (N,C,H,W)
        # get actual query, key, value tensors 
        q = self.w_qs(h)
        k = self.w_ks(h)
        v = self.w_vs(h) 

        attention = self.self_attention(q, k, v)

        # we can clamp the self.alpha and self.beta to be in the range of [0,1]
        # self.alpha = th.clamp(self.alpha, min=0.0, max=1.0)

        # based on user input on adaptive-attention setup, we define the output equation accordingly 
        if self.ada_attn is not None:
            if self.ada_attn == 'alpha':
                out = self.alpha*attention + cnn_features
            if self.ada_attn == 'beta':
                out = attention + self.beta*cnn_features
            if self.ada_attn == 'alpha_beta':
                out = self.alpha*attention + self.beta*cnn_features
        else:
            out = attention + cnn_features   
        # print("multi-head attention forward pass succeed")
        return out


class SpatialWiseAttention(nn.Module):
    """
    Perform self-attention on spatial dimensions (column and row), no permutation of q,k,v dimensions where q, k, v are of size (N, C, H, W) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.SWA = ScaledDotProductAttention()
        else:
            self.SWA = DotProductAttention()

    def forward(self, q, k, v):
        if self.require_scaling:
            # get the size of the last dimension of non-permuted key 
            n = k.shape[-1] # width dim is the second dimension where k.shape=(32,32,20,20) 
            output = self.SWA(q, k, v, n)
        else:
            output = self.SWA(q, k, v)
        
        return output


class ChannelWiseRowAttention(nn.Module):
    """
    Perform self-attention on channel and row dimensions, permutation of q,k,v dimensions: (N, C, H, W) --> (N, W, H, C) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.CWRA = ScaledDotProductAttention()
        else:
            self.CWRA = DotProductAttention()

    def forward(self, q, k, v):
        # permute dimensions so that the last 2 dims of q,k,v are (H,C)=(row, channel)
        q = q.permute(0, 3, 2, 1)
        k = k.permute(0, 3, 2, 1)
        v = v.permute(0, 3, 2, 1)

        if self.require_scaling:
            # get the channel length of permuted key 
            n = k.shape[-1] # channel dim is the last dimension 
            output = self.CWRA(q, k, v, n)
        else:
            output = self.CWRA(q, k, v)
        
        return output.permute(0, 3, 2, 1) # we need to convert (N, W, H, C) back to (N, C, H, W) before adding it to the cnn_features


class ChannelWiseColumnAttention(nn.Module):
    """
    Perform self-attention on channel and column dimensions, permutation of q,k,v dimensions: (N, C, H, W) --> (N, H, W, C) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.CWCA = ScaledDotProductAttention()
        else:
            self.CWCA = DotProductAttention()

    def forward(self, q, k, v):
        # permute dimensions so that the last 2 dims of q,k,v are (W,C)=(column, channel)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        if self.require_scaling:
            # get the channel length of permuted key 
            n = k.shape[-1] # channel dim is the last dimension 
            output = self.CWCA(q, k, v, n)
        else:
            output = self.CWCA(q, k, v)
        
        return output.permute(0, 3, 1, 2) # we need to convert (N, H, W, C) back to (N, C, H, W) before adding with cnn_features


class ChannelWiseRowColumnAttention(nn.Module):
    """
    Combine self-attention performed on both channel-row and channel-column dimensions 
    permutation of q,k,v dimensions: (N, C, H, W) --> (N, W, H, C) 
    permutation of q,k,v dimensions: (N, C, H, W) --> (N, H, W, C) 
    """
    def __init__(self, require_scaling=True):
        super().__init__()
        self.require_scaling = require_scaling
        if self.require_scaling:
            self.CWRA = ScaledDotProductAttention()
            self.CWCA = ScaledDotProductAttention()
        else:
            self.CWRA = DotProductAttention()
            self.CWCA = DotProductAttention()

    def forward(self, q, k, v):
        # permute dimensions so that the last 2 dims of q,k,v are (H,C)=(row, channel)
        q_r = q.permute(0, 3, 2, 1)
        k_r = k.permute(0, 3, 2, 1)
        v_r = v.permute(0, 3, 2, 1)
        # permute dimensions so that the last 2 dims of q,k,v are (W,C)=(column, channel)
        q_c = q.permute(0, 2, 3, 1)
        k_c = k.permute(0, 2, 3, 1)
        v_c = v.permute(0, 2, 3, 1)

        if self.require_scaling:
            # get the channel length of permuted key 
            n = k.shape[-1] # channel dim is the last dimension 
            output_r = self.CWRA(q_r, k_r, v_r, n).permute(0, 3, 2, 1) # convert back to (N, C, H, W)
            output_c = self.CWCA(q_c, k_c, v_c, n).permute(0, 3, 1, 2) # convert back to (N, C, H, W)
        else:
            output_r = self.CWRA(q_r, k_r, v_r).permute(0, 3, 2, 1) # convert back to (N, C, H, W)
            output_c = self.CWCA(q_c, k_c, v_c).permute(0, 3, 1, 2) # convert back to (N, C, H, W)
        
        return output_r + output_c # we can also make this adaptive 


class DotProductAttention(nn.Module):
    """
    Perform dot product attention (without normalization and softmax)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = th.matmul(q, k.transpose(2, 3))
        output = th.matmul(attn, v)
        return output


class ScaledDotProductAttention(nn.Module):
    """
    Perform scaled dot product attention (with normalization and softmax)
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, n):
        attn = th.matmul(q, k.transpose(2, 3))
        normed_attn = attn / sqrt(n)
        scaled_attn = F.softmax(normed_attn, dim=-1)
        output = th.matmul(scaled_attn, v)
        return output


def get_random_obs(n_obs: int, low: int=200, high: int=1000) -> List[np.ndarray]:
    """
    randomly samples the actions and collect n observations

    :param n_obs: number of observations to generate
    :param low: the lowest integer to be drawn from the distribution
    :param high: the highest integer to be drawn from the distribution
    """
    obs_list = []
    vec_env.reset()
    for _ in range(n_obs):
        for _ in range(np.random.randint(low, high=high)):
            obs, _ , _ , _ = vec_env.step([vec_env.action_space.sample()])
        obs_list.append(obs)

    return obs_list


def convert_to_heatmap(input_tensor: th.Tensor):
    """
    Make heatmap from a torch tensor. The input_tensor needs to be normalized and have H, W dimension after squeezing.
    
    :param input_tensor (torch.tensor): shape of (1, 1, H, W) and each element has value in range [0, 1]
    :return: heatmap img shape of (3, H, W)
    """
    heatmap = (255 * input_tensor.squeeze()).type(th.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = th.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = th.cat([r, g, b])

    return heatmap


class GradCAM:
    """
    Calculate GradCAM salinecy map.
    We modified the way the logit is computed to make it compatible with stable-baselines3's policy's return values.

    :param input: input observations with shape of (1, 4, H, W)
    :param deterministic: whether to use greedy action or random action
    :param retain_graph: whether to retain the graph
    :return: saliency_map, action, value, logit, score, gradients, activations, attended_feature_maps
    """

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module, sat: str=None):
        """
        :param arch: the complete model, e.g., model.policy
        :param target_layer: the target layer, e.g., model.policy.features_extractor.c1
        :param sat: self-attention type, e.g., 'NA'
        """
        assert sat is not None, f"sat must be given, use one of 'NA', 'SWA', 'CWRA', 'CWCA', 'CWRCA'!"
        self.model_arch = arch # this is the policy object
            
        # containers for action_net & self_attn_layer layer outputs (apply to all layers in a SB3 policy)
        self.activations_sb3 = {}
        # define a wrapper for forward hook so that we can use it for multiple layers
        def get_activation(name):
            def hook(model, input, output):
                # self.activations_sb3[name] = output.detach() # we cannot detach the tensor from the graph
                self.activations_sb3[name] = output
            return hook

        self.model_arch.action_net.register_forward_hook(get_activation('action_net')) # forward hook of the action_net layer
        if sat != 'NA':
            self.model_arch.features_extractor.self_attn_layer.register_forward_hook(get_activation('self_attn_layer')) #forward hook of the self_attn_layer layer

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0] # stores back-propagated gradients of the target layer

        def forward_hook(module, input, output):
            self.activations['value'] = output # stores feed-forward feature maps of the target layer

        target_layer.register_forward_hook(forward_hook) # register forward hook for target layer
        target_layer.register_full_backward_hook(backward_hook) # register backward hook for target layer

    def forward(self, input, deterministic=None, retain_graph=False):
        assert deterministic is not None, f"'deterministic' must be set to either True or False!"
        b, c, h, w = input.size()

        # logit = self.model_arch(input)
        action, value, action_log_prob = self.model_arch(input, deterministic=deterministic) # deterministic = False for Atari by default
        attended_feature_maps = self.activations_sb3.get('self_attn_layer') # the actual output of the self_attn_layer layer
        logit = self.activations_sb3.get('action_net') # the actual output of the action_net layer
        assert logit is not None, f"action_net output cannot be None!"
        # get the actual value of the neuron based on the action chosen by the agent
        score = logit[:, action].squeeze()

        #  It is essential to zero out the gradients before computing the gradients for each minibatch to ensure accurate and timely parameter updates.
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # the gradients of cnn kernels if target_layer is c1 or c2
        activations = self.activations['value'] # the feature maps if target_layer is c1 or c2 shape=(1, 16, 20, 20)
        b, k, u, v = gradients.size() # b=batch, k=kernel=channel, u=height of the kernel, v=width of the kernel

        alpha = gradients.view(b, k, -1).mean(2) # get the mean of the gradients per CNN kernel
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True) # weighted sum of feature maps where weights are averaged gradients of the kernels, shape=(1, 1, 20, 20)
        saliency_map = F.relu(saliency_map) # take positive values, negative values are forced to 0
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False) # up scale saliency map to input size (1, 1, 84, 84)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max() # normalize saliency map
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, action, value, logit, score, gradients, activations, attended_feature_maps

    def __call__(self, input, deterministic=None, retain_graph=False):
        return self.forward(input, deterministic, retain_graph)