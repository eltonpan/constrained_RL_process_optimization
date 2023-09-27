#!/usr/bin/env python
# coding: utf-8

# # Case Study One

# In[309]:


# Authors: 
# Elton Pan, Antonio del Rio Chanona

import pylab
import pandas as pd
import scipy.integrate as scp
import numpy as np
import seaborn as sns
from pylab import *
import csv
import os
import sys
import copy
import torch
from sklearn.preprocessing import StandardScaler
import collections
import numpy.random as rnd
from scipy.spatial.distance import cdist
import sobol_seq
from scipy.optimize import minimize
eps  = np.finfo(float).eps
import random
import time
import pickle
import imageio
# matplotlib.rcParams['font.sans-serif'] = "Helvetica"
# matplotlib.rcParams['font.family'] = "Helvetica"
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from IPython.display import Audio # Import sound alert dependencies
from IPython import display # For live plots
from mpl_toolkits import mplot3d
from ipywidgets import interact
from __future__ import print_function
import sys
import threading
import json
from time import sleep
try:
    import thread
except ImportError:
    import _thread as thread
    
def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@{0} TOOK TOO LONG @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt
def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

def Done():
    display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))
# Insert whatever audio file you want above

############ Defining Environment ##############
class Model_env: 
    
    # --- initializing model --- #
    def __init__(self, parameters, tf):
        
        # Object variable definitions
        self.parameters       = parameters
        self.tf = tf  
        
    # --- dynamic model definition --- #    
    # model takes state and action of previous time step and integrates -- definition of ODE system at time, t
    def model(self, t, state):
        # internal definitions
        params = self.parameters
        u_L  = self.u0[0] # Control for light intensity
        u_Fn = self.u0[1] # Control for nitrate input
                
        # state vector # Added Cq as the third state element
        Cx  = state[0]
        Cn  = state[1]
        Cq  = state[2]
        
        # parameters # Updated with new params
        u_m = params['u_m']; k_s = params['k_s'];
        k_i = params['k_i']; K_N = params['K_N'];
        u_d = params['u_d']; Y_nx = params['Y_nx'];
        k_m = params['k_m']; k_sq = params['k_sq'];
        k_iq = params['k_iq']; k_d = params['k_d'];
        K_Np = params['K_Np'];
        
        # algebraic equations
        
        # variable rate equations
        dev_Cx = (u_m * u_L * Cx * Cn/(Cn+K_N))/(u_L+k_s+u_L**2/k_i) - u_d*Cx
        dev_Cn = (- Y_nx*u_m* u_L * Cx * Cn/(Cn+K_N))/(u_L+k_s+u_L**2/k_i) + u_Fn
        dev_Cq = (k_m * u_L * Cx)/(u_L+k_sq+u_L**2/k_iq) - k_d * Cq/(Cn+K_Np)

        return np.array([dev_Cx, dev_Cn, dev_Cq],dtype='float64') # Added Cq

    def simulation(self, x0, controls):
        # internal definitions
        model, tf     = self.model, self.tf
        self.controls = controls
        
        # initialize simulation
        current_state = x0
        
        # simulation #ONLY ONE STEP unlike the previous code shown above
        self.u0   = controls
        ode       = scp.ode(model)                      # define ode
        ode.set_integrator('lsoda', nsteps=3000)        # define integrator
        ode.set_initial_value(current_state, tf)         # set initial value
        current_state = list(ode.integrate(ode.t + tf)) # integrate system
        xt            = current_state                   # add current state Note: here we can add randomnes as: + RandomNormal noise
        
        return xt

    def MDP_simulation(self, x0, controls): #simulate ONLY ONE STEP
        xt          = self.simulation(x0, controls) #simulate
####         xt_discrete = self.discrete_env(xt) # make output state discrete
####         return xt_discrete
        return xt #remove this if you want to discretize

    def reward(self, state):
        reward = 100*state[-1][0] - state[-1][1]              # objective function 1
        return reward

# Constants
p = {'u_m' : 0.0923*0.62, 
     'k_s' : 178.85, 
     'k_i' : 447.12, 
     'K_N' : 393.10, 
     'u_d' : 0.001, 
     'Y_nx' : 504.49, 
     'k_m' : 2.544*0.62*1e-4,  
     'k_sq' : 23.51,
     'k_iq' : 800.0, 
     'k_d' : 0.281, 
     'K_Np' : 16.89}

tf  = 240./12. # assuming 12 steps, we divide the whole horizon (240 h) over 12 for one step

# Creating the model
MDP_BioEnv = Model_env(p, tf)

def initialize_MDP_BioEnv():
    '''Initialize MDP_BioEnv with parameters with uncertainty'''
    # Constants with uncertainty
    p_uncertain = {'u_m' : 0.0923*0.62, 
                 'k_s' : np.random.normal(178.9, np.sqrt(17.89)),   # 10 % uncertainty # remember np.sqrt takes in std not variance
                 'k_i' : np.random.normal(447.12, np.sqrt(44.712)), # 10 % uncertainty # remember np.sqrt takes in std not variance
                 'K_N' : np.random.normal(393.10, np.sqrt(39.310)), # 10 % uncertainty # remember np.sqrt takes in std not variance
                 'u_d' : 0.001, 
                 'Y_nx' : 504.49, 
                 'k_m' : 2.544*0.62*1e-4,  
                 'k_sq' : 23.51,
                 'k_iq' : 800.0, 
                 'k_d' : 0.281, 
                 'K_Np' : 16.89}
    tf = 240./12.
    # Initialize the model
    MDP_BioEnv = Model_env(p_uncertain, tf)

def generate_initial_state():
    '''Generates initial state with uncertainty'''
    Cx = np.random.normal(1, np.sqrt(1e-3))
    Cn = np.random.normal(150, np.sqrt(22.5))
    Cq = 0
    t  = 0
    initial_state = [Cx, Cn, Cq, t]
    return initial_state

def transition(old_state, action):
    '''Gives the new state given the current state and action
       Arguments
       old state : [Cx, Cn, Cq, t] 
       action    : [u_L, u_Fn]
    
       Output 
       new state : [Cx, Cn, Cq, t]
       reward    : [Cq_Terminal]
       '''
    # If terminal state is reached (episode in final step)
    if abs(old_state[3] - 240.) < 0.1:
        reward    = copy.deepcopy(old_state[2])           # Reward at terminal state is concentration of q
        new_state = MDP_BioEnv.MDP_simulation(old_state[:-1], action)  # Take action and evolve using model
        new_state.append(old_state[-1] + tf)                           # Increment time by tf (20 h)
    
    # Else if past terminal state (episode has ended)
    elif (old_state[3] > 240.):
        reward    = 0         # Zero reward given
        new_state = old_state # Loop back to itself (no evolution)
    
    # Else non-terminal state (episode has not ended)
    else:
        reward    = 0         # Zero reward given
        new_state = MDP_BioEnv.MDP_simulation(old_state[:-1], action) # Take action and evolve using model
        new_state.append(old_state[-1] + tf)                          # Increment time by tf (20 h)

    return new_state, reward

def generate_random_episode(initial_state): 
    '''Generates an episode [
                             [ [Cx, Cn, Cq, t], [u_L, u_Fn], reward ]
                             ...
                             ...
                             ]
    with random policy with
    an initial state
    '''
    # Initial state
    state  = initial_state # initial state
    episode = []
    
    # Simulate 13 steps (not 12 because we want to exit the terminal step)
    for step in range(13):
        old_state = state                  # Save old state
        u_L  = np.random.uniform(120, 400) # Pick random u_L
        u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
        action = [u_L, u_Fn]
        
        state, reward = transition(state, action)
        episode       += [[old_state, action, reward]]
    return episode
# generate_random_episode(initial_state = [1.0,150.0,0,0]) # Test generate_random_episode


def extract_data_from_episode(episode, discount_factor = 0.9):
    '''
    Argument: An episode generated using the generate_random_episode() function
    
    Output: 13 Datapoints in the form of [[Cx, Cn, Cq, t, u_L, u_Fn], Q] for training 
            the Q-network
    '''
    Q_data = []
    for step in reversed(range(13)): # Each episode has 13 entries, and Q table is updated in reversed order
        state, action, reward = episode[step] 

        if step == 12: # If terminal state i.e. t = 240
            G = reward # Return = reward at terminal state
        else:
            G = reward + discount_factor * G  # Return = reward + discounted return of the PREVIOUS state
        
        u_L, u_Fn = action     #  Unpack controls
        state.append(u_L)       # Append u_L
        state.append(u_Fn)      # Append u_Fn
        data_point = [state, G] # Construct datapoint where state is index 0, and return is index 1
        Q_data += [data_point]
    return Q_data
# episode = generate_random_episode(initial_state = [1.0,150.0,0,0])
# extract_data_from_episode(episode, discount_factor = 0.9) # Test function

def standardize_state_Q(state): # For Q-network
    '''Argument: Un-standardized [Cx, Cn, Cq, t, u_L, u_Fn]
    Output: Standardized [Cx, Cn, Cq, t, u_L, u_Fn] using previously determined mean and std values''' 
    for feature in range(len(x_mean_Q)): 
        state[feature] = (state[feature] - x_mean_Q[feature])/x_std_Q[feature]
    return state

def unstandardize_state_Q(state): # For Q-network
    '''Argument: Standardized [Cx, Cn, Cq, t, u_L, u_Fn]
    Output: Un-standardized [Cx, Cn, Cq, t, u_L, u_Fn] using previously determined mean and std values''' 
    for feature in range(len(x_mean_Q)): 
        state[feature] = (state[feature] * x_std_Q[feature]) + x_mean_Q[feature]
    return state

def standardize_state_C(state): # For C-network
    '''Argument: Un-standardized [Cx, Cn, Cq, t_f - t, u_L, u_Fn]
    Output: Standardized [Cx, Cn, Cq, t_f - t, u_L, u_Fn] using previously determined mean and std values''' 
    for feature in range(len(x_mean_C)): 
        state[feature] = (state[feature] - x_mean_C[feature])/x_std_C[feature]
    return state

def unstandardize_state_C(state): # For C-network
    '''Argument: Standardized [Cx, Cn, Cq, t_f - t, u_L, u_Fn]
    Output: Un-standardized [Cx, Cn, Cq, t_f - t, u_L, u_Fn] using previously determined mean and std values''' 
    for feature in range(len(x_mean_C)): 
        state[feature] = (state[feature] * x_std_C[feature]) + x_mean_C[feature]
    return state

def take_random_action(epsilon): # Epsilon represents the probability of taking a random action
    if np.random.uniform(0,1) < epsilon:
        return True
    else:
        return False
    

def max_action(state):
    '''Argument: State   [Cx, Cn, Cq, t]
       Output  : Control [u_L, u_Fn] that maximizes Q_value using stochastic optimization'''
    
    # ROUND ONE: Get a ROUGH estimate of max action
    action_distribution = []
    for i in range(200): # Take 100 actions
        u_L  = np.random.uniform(120, 400) # Pick random u_L
        u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
        action_distribution += [[u_L, u_Fn]]

    inputs = []
    for a in action_distribution:
        s = state.copy()
        s.append(a[0]) # Append u_L
        s.append(a[1]) # Append u_Fn
        s = standardize_state_Q(s) # Standardize the input using function defined
        inputs += [s]
    inputs          = torch.tensor(inputs)
    index_of_highest_Q = np.argmax(Q_net(inputs).detach().numpy()) 

    # Unstandardize controls
    max_u_L  = (inputs[index_of_highest_Q][4] * x_std_Q[4]).item() + x_mean_Q[4]
    max_u_Fn = (inputs[index_of_highest_Q][5] * x_std_Q[5]).item() + x_mean_Q[5]
    max_control = [max_u_L, max_u_Fn]

    # ROUND TWO: Get a PRECISE estimate of max action
    action_distribution = []
    for i in range(200): # Take 100 actions
        u_L  = np.random.uniform(max(max_control[0]-50, 120), min(max_control[0]+50, 400))
        u_Fn = np.random.uniform(max(max_control[1]-5, 0), min(max_control[1]+5, 40))
        action_distribution += [[u_L, u_Fn]]

    inputs = []
    for a in action_distribution:
        s = state.copy()
        s.append(a[0]) # Append u_L
        s.append(a[1]) # Append u_Fn
        s = standardize_state_Q(s) # Standardize the input using function defined
        inputs += [s]

    inputs          = torch.tensor(inputs)
    index_of_highest_Q = np.argmax(Q_net(inputs).detach().numpy()) 

    # Unstandardize controls
    max_u_L  = (inputs[index_of_highest_Q][4] * x_std_Q[4]).item() + x_mean_Q[4]
    max_u_Fn = (inputs[index_of_highest_Q][5] * x_std_Q[5]).item() + x_mean_Q[5]
    max_control = [max_u_L, max_u_Fn] # Predict action with the highest Q value

    return max_control
# max_action([1.0, 150.0, 0, 0]) # Test function

@exit_after(10)
def score_NN_policy(NN, initial_state, num_iterations = 10, get_control = False, g1_threshold = None, g2_threshold = None, g3_threshold = None):
    total_score = 0
    sum_of_policies = []
    if g1_threshold == None: # If no threshold is given (non constrained NN)
        for j in range(num_iterations): 
            state     = initial_state.copy() # Initial state
            x_data    = [state[0]] # Store initial data
            N_data    = [state[1]]
            q_data    = [state[2]]
            t_data    = [state[3]]
            my_policy = []
            for i in range(12): #take ten steps
                action     = NN(state) # Predict action using NN
                state      = transition(state, action)[0]
                my_policy += [action]
                x_data    += [state[0]]
                N_data    += [state[1]]
                q_data    += [state[2]]
                t_data    += [state[3]]
            sum_of_policies += [my_policy] # Create list of lists
            score            = q_data[12]
            total_score     += score
    elif g1_threshold != None: # If threshold given (NN with constrained)
        for j in range(num_iterations): 
            state     = initial_state.copy() # Initial state
            x_data    = [state[0]] # Store initial data
            N_data    = [state[1]]
            q_data    = [state[2]]
            t_data    = [state[3]]
            my_policy = []
            for i in range(12): #take ten steps
#                 print('state after action:', state)
                state_for_NN = state.copy()
                action     = NN(state_for_NN, g1_threshold = g1_threshold, g2_threshold = g2_threshold, g3_threshold = g3_threshold) # Predict action using NN
#                 print('state before action:', state)
                state      = transition(state, action)[0]
                my_policy += [action]
                x_data    += [state[0]]
                N_data    += [state[1]]
                q_data    += [state[2]]
                t_data    += [state[3]]
            sum_of_policies += [my_policy] # Create list of lists
            score            = q_data[12]
            total_score     += score
    if get_control == True:
        return total_score/num_iterations, np.array(sum_of_policies)
    else:
        return total_score/num_iterations
# score_NN_policy(max_action, [1.0, 150.0, 0, 0], num_iterations = 10)

@exit_after(5)
def generate_episode_with_NN(NN, initial_state, epsilon, g1_threshold = None, g2_threshold = None, g3_threshold = None): 
    '''Generates an episode with the chosen action of each step having:
    Probability of epsilon       ---> random action
    Probability of (1 - epsilon) ---> greedy action (according to neural network)
    '''
    episode = []
    state = initial_state # Initial state
    
    if g1_threshold == None: # IF CONSTRAINTS ARE SWITCHED OFF
        for i in range(13): #take (12 + 1) steps
#             old_state = state # Old state for storing into episode
            old_state = copy.deepcopy(state)
    
            if take_random_action(epsilon): # Take random action
                u_L  = np.random.uniform(120, 400) # Pick random u_L
                u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
                action = [u_L, u_Fn]

            else:                           # Else take greedy action
                action = list(NN(old_state))

            state, reward  = transition(state, action)     # Evolve to get new state 
            episode       += [[old_state, action, reward]] # Update step
    
    elif g1_threshold != None: # IF CONSTRAINTS ARE SWITCHED ON
        for i in range(13): #take (12 + 1) steps
#             print('state at the start of each step:', state)
#             old_state_copy = copy.deepcopy(state)  # THIS IS IMPORTANT BCOS THE NN BELOW SOMEHOW DISTORTS OLD_STATE
            if take_random_action(epsilon): # Take random action
                u_L  = np.random.uniform(120, 400) # Pick random u_L
                u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
                action = [u_L, u_Fn]
            else:                           # Else take greedy action
                if i == 12: # If transition out of the terminal state, pick any action
                    action = action
#                     print('13th transition')
                else:
                    state_for_NN = copy.copy(state)
                    
#                     print('state_for_nn:', state_for_NN)
#                     print('state before acion:', state)
                    
                    action = list(NN(state_for_NN, g1_threshold = g1_threshold, g2_threshold = g2_threshold, g3_threshold = g3_threshold)) # APPLY THRESHOLD
#                     print('state after acion:', state)
#                     print(i)
                #                     print('old_state:', old_state)
#                 print('done')

            old_state = state.copy() # Old state for storing into episode
#             print('old_state_1:', old_state)
#             print('state before evolution:', state)
            state, reward  = transition(state, action)     # Evolve to get new state
#             print('state after evolution:', state)
#             print('')
            episode       += [[old_state, action, reward]] # Update step
    return episode
# generate_episode_with_NN(max_action, [1.0, 150.0, 0, 0], epsilon = 0.1)

def update_replay_buffer(replay_buffer, episode, discount_factor = 0.9):
    '''Argument: replay buffer (a collections.deque object) and ONE episode
    Output: Adds standardized datapoints [[Cx, Cn, Cq, t, u_L, u_Fn], Q] for training NN into replay buffer
    '''
    data = extract_data_from_episode(episode, discount_factor = discount_factor) # Extract data points from episode
    for data_point in data:
        X, y = data_point # Unpack datapoint
        X = standardize_state_Q(X) # Standardize X
        y = ((y - y_mean_Q)/y_std_Q)[0] # Standardize y
        data_point = [X, y] # Repack datapoint
        replay_buffer.extend([data_point]) # Add to replay buffer
    
    return replay_buffer
# replay_buffer = collections.deque(maxlen = 3000) # Max capacity of 3000
# episode = generate_episode_with_NN(max_action, [1.0, 150.0, 0, 0], epsilon = 0.1)
# update_replay_buffer(replay_buffer, episode)

def plot_episode(episode):
    '''Plots an episode and corresponding score'''
    x_list = []
    N_list =[]
    q_list =[]
    t_list =[]
    u_L_list= []
    u_Fn_list = []
    reward_list = []
    t = 0
    for step in episode:
        [Cx, Cn, Cq, t], [u_L, u_Fn], reward = step
        x_list += [Cx]
        N_list += [Cn]
        q_list += [Cq]
        t_list += [t]
        u_L_list+= [u_L]
        u_Fn_list += [u_Fn]
        reward_list += [reward]
        t += tf

    plt.figure(figsize=(10,10))

    plt.subplot(3,2,1)
    plt.plot(t_list, x_list, label = 'x trajectory', color = 'b')
    plt.legend(fontsize = 15)

    plt.subplot(3,2,2)
    plt.plot(t_list, N_list, label = 'N trajectory', color = 'b')
    plt.legend(fontsize = 15)

    plt.subplot(3,2,3)
    plt.plot(t_list, q_list, label = 'q trajectory', color = 'b')
    plt.legend(fontsize = 15)

    plt.subplot(3,2,5)
    plt.step(t_list, u_L_list, label = 'u_L trajectory', color = 'r')
    plt.ylim(100,420)
    plt.legend(fontsize = 15)

    plt.subplot(3,2,6)
    plt.step(t_list, u_Fn_list, label = 'u_Fn trajectory', color = 'r')
    plt.ylim(0,43)
    plt.legend(fontsize = 15)

    plt.show()
    print('Score:', q_list[-1])
    # print('x_list:', x_list)
    # print('n_list:', N_list)
    # print('q_list:', q_list)
    # print('reward_list:', reward_list)
    # print('t_list:', t_list)
# episode = generate_episode_with_NN(NN = max_action, initial_state = [1.0, 150.0, 0, 0], epsilon = 0)
# plot_episode(episode)

def violate_g1(Cn):
    if Cn > 800:
        return True
    else:
        return False

def violate_g2(Cq, Cx):
    if Cq - 0.011*Cx > 0:
        return True
    else:
        return False    
    
def violate_g3(Cn):
    if Cn > 150:
        return True
    else:
        return False    

def plot_episode_pool(episode_pool):
    '''Plots an episode pool (n number of episodes) and corresponding score'''
    episodes_g1_violated = 0 # No. of episodes that violate g1
    episodes_g2_violated = 0 # No. of episodes that violate g2
    episodes_g3_violated = 0 # No. of episodes that violate g3
    episodes_g1g2g3_any_violated = 0 # No. of episodes that violate either g1 and g2
    total_score = 0
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
    Cx_pool  = []
    Cn_pool  = []
    Cq_pool  = []
    u_Fn_pool = []
    u_L_pool = []
    
    for idx in range(len(episode_pool)): # Idx refers to nth episode in episode pool
        episode = episode_pool[idx]
        Cx_list  = []
        Cn_list  = []
        Cq_list  = []
        t_list   = []
        u_Fn_list = []
        u_L_list = []
        reward_list = []
        t = 0.
        for step in episode:
            [Cx, Cn, Cq, _], [u_Fn, u_L], reward = step
            Cx_list  += [Cx]
            Cn_list  += [Cn]
            Cq_list  += [Cq]
            t_list   += [t]
            u_Fn_list += [u_Fn]
            u_L_list += [u_L]
            reward_list += [reward]
            t += 20.
    #     fig, ax = plt.figure(figsize=(10,10))
        
        # Add scores to total score
        score        = Cq_list[-1]
        total_score += score
        
        # Count number of EPISODES that have violation
        if True in map(violate_g1, Cn_list):
            episodes_g1_violated += 1
        for j in range(len(Cq_list)):
            if violate_g2(Cq_list[j], Cx_list[j]):
                episodes_g2_violated += 1
                break
        if violate_g3(Cn_list[-1]):
            episodes_g3_violated += 1

        if True in map(violate_g1, Cn_list):
            episodes_g1g2g3_any_violated += 1
        elif violate_g3(Cn_list[-1]):
            episodes_g1g2g3_any_violated += 1
        else:
            for j in range(len(Cq_list)):
                if violate_g2(Cq_list[j], Cx_list[j]):
                    episodes_g1g2g3_any_violated += 1
                    break
        
        Cx_pool  += [Cx_list]
        Cn_pool  += [Cn_list]
        Cq_pool  += [Cq_list]
        u_Fn_pool += [u_Fn_list]
        u_L_pool += [u_L_list]
        
        plt.subplot(3,3,1)
        plt.plot(t_list, Cx_list, color = 'grey', alpha = 0.5)
        plt.legend(fontsize = 15)

        plt.subplot(3,3,2)
        plt.plot(t_list, Cn_list, color = 'grey', alpha = 0.5)

        plt.subplot(3,3,3)
        plt.plot(t_list, Cq_list, color = 'grey', alpha = 0.5)
        
        plt.subplot(3,3,4)
        plt.step(t_list, u_Fn_list, color = 'grey', alpha = 0.5)

        plt.subplot(3,3,5)
        plt.step(t_list, u_L_list, color = 'grey', alpha = 0.5)
    
    Cx_pool  = np.array(Cx_pool)
    Cn_pool  = np.array(Cn_pool)
    Cq_pool  = np.array(Cq_pool)
    u_Fn_pool = np.array(u_Fn_pool)
    u_L_pool = np.array(u_L_pool)
    
    # Take mean of all columns (all time steps)
    Cx_avg_list  = Cx_pool.mean(axis=0)
    Cn_avg_list  = Cn_pool.mean(axis=0)
    Cq_avg_list  = Cq_pool.mean(axis=0)
    u_Fn_avg_list = u_Fn_pool.mean(axis=0)
    u_L_avg_list = u_L_pool.mean(axis=0)
    
    # Find +/- one STD of all columns
#     T_std_upper = np.percentile(T_pool, 84, axis = 0)
#     T_std_lower = np.percentile(T_pool, 16, axis = 0)
    
#     Vol_std_upper = np.percentile(Vol_pool, 84, axis = 0)
#     Vol_std_lower = np.percentile(Vol_pool, 16, axis = 0)
    
    plt.subplot(3,3,1)
    plt.plot(t_list, Cx_avg_list, label = 'x trajectory', color = 'b')
    plt.legend(fontsize = 15)

    plt.subplot(3,3,2)
    plt.plot(t_list, Cn_avg_list, label = 'N trajectory', color = 'b')
    plt.legend(fontsize = 15)

    plt.subplot(3,3,3)
    plt.plot(t_list, Cq_avg_list, label = 'q trajectory', color = 'b')
    plt.legend(fontsize = 15)

    plt.subplot(3,3,4)
    plt.step(t_list, u_Fn_avg_list, label = 'u_Fn trajectory', color = 'r')
    plt.legend(fontsize = 15)

    plt.subplot(3,3,5)
    plt.step(t_list, u_L_avg_list, label = 'u_L trajectory', color = 'r')
    plt.legend(fontsize = 15)
    
    fig.tight_layout()
    fig.delaxes(ax[1][2]) # delete last 6th plot (empty)
    fig.delaxes(ax[2][0]) # delete last 7th plot (empty)
    fig.delaxes(ax[2][1]) # delete last 8th plot (empty)
    fig.delaxes(ax[2][2]) # delete last 9th plot (empty)
    plt.show()
    
    avg_score = total_score/len(episode_pool)
    print('Score:', avg_score)
    print('total no. of episodes:', len(episode_pool))
    print('episodes that violate g1:',episodes_g1_violated) 
    print('episodes that violate g2:',episodes_g2_violated)
    print('episodes that violate g3:',episodes_g3_violated)
#     print('episodes that violate both g1 and g2:',episodes_g1g2_both_violated)
    print('episodes that violate any constraint:',episodes_g1g2g3_any_violated)

def extract_constraint_values_from_episode(episode, N_limit = 800, terminal_N_limit = 150):
    '''Arguments : 1 episode and N_limit
       Output    : Index 0 gives input data in terms of [Cx, Cn, Cq, t-t_f, u_L, u_Fn]
                   Index 1 gives target constraint values'''

    state_action = [] # Initialize inputs
    g1_target   = [] # Initialize target value (constraints)
    g2_target   = [] # Initialize target value (constraints)
    g3_target   = [] # Initialize target value (constraints)
    targets = [] # Initialize target values [[g1, g2 g3], ...]
    
    num_transitions = 12
    for i in range(num_transitions): # Index 0 to 11 instead of 12 because we consider the 1st 12 transitions
        step             = episode[i]       # Choose a specific step
        new_step         = episode[i+1]     # and the corresponding subsequent step
        
        state, action, _ = step             # Unpack state & action
        state            = list(state)
        state[3]         = 240. - state[3]   # IMPORTANT: Modify t to (t_f - t)
        state           += action           # Append u_L and u_Fn
        
        # INPUT for training
        state_action += [state]
        
        new_state, _, _  = new_step         # Unpack subsequent state
        Cx               = new_state[0]     # Biomass of SUBSEQUENT step (for g2)
        Cn               = new_state[1]     # Nitrate of SUBSEQUENT step (for g1)
        Cq               = new_state[2]     # Bioproduct of SUBSEQUENT step (for g2)
        
        if i == (num_transitions-1): # If final transition
            Cn_T         = new_state[1]     # Nitrate of TERMINAL state (for g3)

        # TARGETS for path constraint 1 (g1) where Cn - 800 =< 0
        g1           = Cn - N_limit # Calculate constraint value (+ve for exceeding limit)
        g1_target   += [[g1]]      # TARGET OUTPUT (IMPORTANT TO USE A NESTED LIST hence the DOUBLE square brackets)

        # TARGETS for path constraint 2 (g2) where Cq - 0.011*Cx =< 0
        g2           = Cq - 0.011 * Cx # Calculate constraint value (+ve for exceeding limit)
        g2_target   += [[g2]]      # TARGET OUTPUT (IMPORTANT TO USE A NESTED LIST hence the DOUBLE square brackets)
        
    # TARGETS for terminal constraint 2 (g3) where CnT - 150 =< 0
    g3           = Cn_T - terminal_N_limit # Calculate constraint value (+ve for exceeding limit)
    g3_target    = []
    for k in range(num_transitions):
        g3_target += [[g3]] # Nested list
    # End constraint violation same for all steps in same episode
    
    # Update constraints using "Crystal Ball/Oracle": Highest/Worst constraint value from current and future steps
    for j in range(num_transitions - 1): # Index of constraint value to be checked (11 instead of 12 bcos the 12th value is terminal and has no future)
        for k in range(num_transitions-j): # Number of steps into the future
            # ====Oracle for g1====
            if g1_target[j] < g1_target[j+k]: # If future g1 value is LARGER than current value
                g1_target[j] = g1_target[j+k]  # Replace current value with future value
        
#         # ===== DISCOUNTED ORACLE FOR g2====
#         alpha = (j+1)/12 # Degree of contribution by terminal g2 (low for early states, high for late states)
#         g2_target[j][0] = alpha*g2_target[-1][0] + (1 - alpha)*g2_target[j][0]
            # ====Oracle for g2====
            if g2_target[j] < g2_target[j+k]: # If future g2 value is LARGER than current value
                g2_target[j] = g2_target[j+k]  # Replace current value with future value
            
    
#     for m in range(num_transitions): # Pack into a nested list
#         g1 = g1_target[m]
#         g2 = g2_target[m]
#         g3 = g3_target[m]
#         targets += [[g1]] # g1 only - NEEDS TO BE NESTED LIST
#         targets += [[g1, g2]] # g1 and g2 only - NEEDS TO BE NESTED LIST
#         targets += [[g1, g2, g3]] # g1, g2 and g3 - NEEDS TO BE NESTED LIST
        
#     print('g1:', g1_target)
#     print('')
#     print('g2:', g2_target)
#     print('')
#     print('g3:', g3_target)  
    return state_action, g1_target, g2_target, g3_target
# episode = generate_random_episode(initial_state = [1.0,150.0,0,0]) 
# extract_constraint_values_from_episode(episode)

def extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150):
    '''Arguments : 1 episode and N_limit
       Output    : Index 0 gives input data in terms of [Cx, Cn, Cq, t-t_f, u_L, u_Fn]
                   Index 1 gives target constraint values'''

    state_action = [] # Initialize inputs
    g1_target   = [] # Initialize target value (constraints)
    g2_target   = [] # Initialize target value (constraints)
    g3_target   = [] # Initialize target value (constraints)
    targets = [] # Initialize target values [[g1, g2 g3], ...]
    
    num_transitions = 12
    for i in range(num_transitions+1): # Index 0 to 11 instead of 12 because we consider the 1st 12 transitions
        step             = episode[i]       # Choose a specific step
        
        state, _, _ = step             # Unpack state & action
        Cx = state[0]
        Cn = state[1]
        Cq = state[2]
        
        # TARGETS for path constraint 1 (g1) where Cn - 800 =< 0
        g1           = Cn - N_limit # Calculate constraint value (+ve for exceeding limit)
        g1_target   += [[g1]]      # TARGET OUTPUT (IMPORTANT TO USE A NESTED LIST hence the DOUBLE square brackets)

        # TARGETS for path constraint 2 (g2) where Cq - 0.011*Cx =< 0
        g2           = Cq - 0.011 * Cx # Calculate constraint value (+ve for exceeding limit)
        g2_target   += [[g2]]      # TARGET OUTPUT (IMPORTANT TO USE A NESTED LIST hence the DOUBLE square brackets)
        
    # TARGETS for terminal constraint 2 (g3) where CnT - 150 =< 0
    g3           = Cn - terminal_N_limit # Calculate constraint value (+ve for exceeding limit)
    g3_target    += [[g3]]
    # End constraint violation same for all steps in same episode
            
    return state_action, g1_target, g2_target, g3_target
# episode = generate_random_episode(initial_state = [1.0,150.0,0,0]) 
# extract_constraint_values_from_episode(episode)

def update_g1_buffer(g1_buffer, episode):
    '''Argument: g1 buffer (a collections.deque object) and ONE episode
    Output: Adds standardized datapoints [[Cx, Cn, Cq, t_f-f, u_L, u_Fn], g1] for training g1_net into replay buffer
    '''
    state_action, g1_constraint, g2_constraint, g3_constraint = extract_constraint_values_from_episode(episode) # Extract data points from episode
#     print('state_action:', state_action)

#     print('constraint:', constraint)
    for idx in range(len(state_action)):
        state_action[idx] = standardize_state_C(state_action[idx])
        g1_constraint[idx] = (np.array(g1_constraint[idx]) - y_mean_C[0])/y_std_C[0]
        data_point = [list(state_action[idx]), g1_constraint[idx].item()] # Repack datapoint
        g1_buffer.extend([data_point]) # Add to replay buffer
    
    return g1_buffer
# episode   = generate_random_episode(initial_state = [1.0,150.0,0,0]) 
# g1_buffer = collections.deque(maxlen = 3000) # Max capacity of 3000
# g1_buffer = update_g1_buffer(g1_buffer, episode)

def update_g2_buffer(g2_buffer, episode):
    '''Argument: g2 buffer (a collections.deque object) and ONE episode
    Output: Adds standardized datapoints [[Cx, Cn, Cq, t_f-f, u_L, u_Fn], g1] for training g1_net into replay buffer
    '''
    state_action, g1_constraint, g2_constraint, g3_constraint = extract_constraint_values_from_episode(episode) # Extract data points from episode
#     print('state_action:', state_action)

#     print('constraint:', constraint)
    for idx in range(len(state_action)):
        state_action[idx] = standardize_state_C(state_action[idx])
        g2_constraint[idx] = (np.array(g2_constraint[idx]) - y_mean_C[1])/y_std_C[1]
        data_point = [list(state_action[idx]), g2_constraint[idx].item()] # Repack datapoint
        g2_buffer.extend([data_point]) # Add to replay buffer
    
    return g2_buffer

def update_g3_buffer(g3_buffer, episode):
    '''Argument: g3 buffer (a collections.deque object) and ONE episode
    Output: Adds standardized datapoints [[Cx, Cn, Cq, t_f-f, u_L, u_Fn], g1] for training g1_net into replay buffer
    '''
    state_action, g1_constraint, g2_constraint, g3_constraint = extract_constraint_values_from_episode(episode) # Extract data points from episode
#     print('state_action:', state_action)

#     print('constraint:', constraint)
    for idx in range(len(state_action)):
        state_action[idx] = standardize_state_C(state_action[idx])
        g3_constraint[idx] = (np.array(g3_constraint[idx]) - y_mean_C[2])/y_std_C[2]
        data_point = [list(state_action[idx]), g3_constraint[idx].item()] # Repack datapoint
        g3_buffer.extend([data_point]) # Add to replay buffer
    
    return g3_buffer

def anneal_NN():
    '''Retrains to the neural networks (Q-net and constraint nets) to anneal the policy to
    solve the problem of frozen algorithm'''
    # TRAIN Q-NET
    combined_inputs_Q = []
    combined_targets_Q = []
    samples_Q          = random.sample(replay_buffer, 100) # Draw random samples from replay buffer
    for inputs, target in samples_Q:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_Q  += [inputs]
            combined_targets_Q += [[target]]
    combined_inputs_Q  = torch.tensor(combined_inputs_Q).double()  # Convert list to tensor
    combined_targets_Q = torch.tensor(combined_targets_Q).double() # Convert list to tensor
    Q_optimizer = torch.optim.Adam(Q_net.parameters(), lr=1e-3) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    for epoch in range(20):
        prediction = Q_net(combined_inputs_Q)                    # Input x and predict based on x
        Q_loss     = loss_func(prediction, combined_targets_Q) # Must be (1. nn output, 2. target)
        Q_optimizer.zero_grad()   # Clear gradients for next train
        Q_loss.backward()         # Backpropagation, compute gradients
        Q_optimizer.step() 

    # TRAIN G1-NET
    combined_inputs_g1  = [] # List of lists to be converted into input tensor
    combined_targets_g1 = []
    samples_g1          = random.sample(g1_buffer, 500) # Draw random samples from replay buffer
    for inputs, target in samples_g1:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_g1  += [inputs]
            combined_targets_g1 += [[target]]
    combined_inputs_g1  = torch.tensor(combined_inputs_g1).double()  # Convert list to tensor
    combined_targets_g1 = torch.tensor(combined_targets_g1).double()
    g1_optimizer = torch.optim.Adam(g1_net.parameters(), lr=1e-4) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    for epoch in range(10):
        prediction = g1_net(combined_inputs_g1)                    # Input x and predict based on x
        g1_loss    = loss_func(prediction, combined_targets_g1) # Must be (1. nn output, 2. target)
        g1_optimizer.zero_grad()   # Clear gradients for next train
        g1_loss.backward()         # Backpropagation, compute gradients
        g1_optimizer.step()        # Apply gradients

    # TRAIN G2-NET
    combined_inputs_g2  = [] # List of lists to be converted into input tensor
    combined_targets_g2 = []
    samples_g2          = random.sample(g2_buffer, 1000) # Draw random samples from replay buffer
    for inputs, target in samples_g2:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_g2  += [inputs]
            combined_targets_g2 += [[target]]
    combined_inputs_g2  = torch.tensor(combined_inputs_g2).double()  # Convert list to tensor
    combined_targets_g2 = torch.tensor(combined_targets_g2).double() # Convert list to tensor
    g2_optimizer = torch.optim.Adam(g2_net.parameters(), lr=1e-3) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    for epoch in range(100):
        prediction = g2_net(combined_inputs_g2)                    # Input x and predict based on x
        g2_loss    = loss_func(prediction, combined_targets_g2) # Must be (1. nn output, 2. target)
        g2_optimizer.zero_grad()   # Clear gradients for next train
        g2_loss.backward()         # Backpropagation, compute gradients
        g2_optimizer.step()        # Apply gradients
    
    # TRAIN G3-NET
    combined_inputs_g3  = [] # List of lists to be converted into input tensor
    combined_targets_g3 = [] # List to be converted into target tensor
    samples_g3          = random.sample(g3_buffer, 1000) # Draw random samples from replay buffer
    for inputs, target in samples_g3:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_g3  += [inputs]
            combined_targets_g3 += [[target]]
    combined_inputs_g3  = torch.tensor(combined_inputs_g3).double()  # Convert list to tensor
    combined_targets_g3 = torch.tensor(combined_targets_g3).double() # Convert list to tensor
    g3_optimizer = torch.optim.Adam(g3_net.parameters(), lr=1e-3) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    for epoch in range(200):
        prediction = g3_net(combined_inputs_g3)                    # Input x and predict based on x
        g3_loss    = loss_func(prediction, combined_targets_g3) # Must be (1. nn output, 2. target)
        g3_optimizer.zero_grad()   # Clear gradients for next train
        g3_loss.backward()         # Backpropagation, compute gradients
        g3_optimizer.step()        # Apply gradients
    print('NEURAL NETWORKS ANNEALED')


# In[18]:


def get_violations(thresholds):
    '''Argument: A 1 x 2 array of thresholds [g1_threshold, g2_threshold]
       Output: A 1 x 2 array of 99.5th percentile (worst from each episode) g1, g2 values [g1, g2 violated]'''
    a, b = thresholds # Unpack g1 and g2 thresholds
    max_g1_list = []
    max_g2_list = []
#     max_g3_list = []
    for i in range(1000):
        try:
            episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                     initial_state = generate_initial_state(),
                                                     epsilon = 0, 
                                                     g1_threshold = a,
                                                     g2_threshold = b,
                                                     g3_threshold = 0) # Generate 1 episode
        except KeyboardInterrupt:
            try:
                episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                     initial_state = generate_initial_state(),
                                                     epsilon = 0, 
                                                     g1_threshold = a,
                                                     g2_threshold = b,
                                                     g3_threshold = 0) # Generate 1 episode
            except KeyboardInterrupt:
                try:
                    episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                     initial_state = generate_initial_state(),
                                                     epsilon = 0, 
                                                     g1_threshold = a,
                                                     g2_threshold = b,
                                                     g3_threshold = 0) # Generate 1 episode
                except KeyboardInterrupt:
                    try:
                        episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                     initial_state = generate_initial_state(),
                                                     epsilon = 0, 
                                                     g1_threshold = a,
                                                     g2_threshold = b,
                                                     g3_threshold = 0) # Generate 1 episode
                    except KeyboardInterrupt:
                        try:
                            episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                     initial_state = generate_initial_state(),
                                                     epsilon = 0, 
                                                     g1_threshold = a,
                                                     g2_threshold = b,
                                                     g3_threshold = 0) # Generate 1 episode
                        except KeyboardInterrupt:
                            pass
        # WORST violation of this episode
        max_g1 = max(extract_constraint_values_from_episode(episode, N_limit = 800, terminal_N_limit = 150)[1])[0]
        max_g2 = max(extract_constraint_values_from_episode(episode, N_limit = 800, terminal_N_limit = 150)[2])[0]
#         max_g3 = max(extract_constraint_values_from_episode(episode, N_limit = 800, terminal_N_limit = 150)[3])[0]
        max_g1_list += [max_g1]
        max_g2_list += [max_g2]
#         max_g3_list += [max_g3]

    max_g1_list.sort(reverse = True)
    max_g2_list.sort(reverse = True)    
    
    g1_top_avg = np.mean(max_g1_list[:10])
    g2_top_avg = np.mean(max_g2_list[:10])
    
#     g1_95_pct = np.percentile(max_g1_list, 95) # Take 95th percentile of worst g1 violations
#     g1_96_pct = np.percentile(max_g1_list, 96)
#     g1_97_pct = np.percentile(max_g1_list, 97)
#     g1_98_pct = np.percentile(max_g1_list, 98)
#     g1_99_pct = np.percentile(max_g1_list, 99)
#     g1_worst = max(max_g1_list) # Worst violation of g1 amongst all episodes

#     g2_95_pct = np.percentile(max_g2_list, 95) # Take 95th percentile of worst g2 violations
#     g2_96_pct = np.percentile(max_g2_list, 96)
#     g2_97_pct = np.percentile(max_g2_list, 97)
#     g2_98_pct = np.percentile(max_g2_list, 98)
#     g2_99_pct = np.percentile(max_g2_list, 99)
#     g2_worst = max(max_g2_list) # Worst violation of g2 amongst all episodes
    
#     g1_average = (g1_97_pct + g1_98_pct + g1_99_pct)/3
#     g2_average = (g2_97_pct + g2_98_pct + g2_99_pct)/3
    
    print('g1_threshold:', a)
    print('g2_threshold:', b)
#     print('g1_average:', g1_average)
#     print('g2_average:', g2_average)
    print('g1_top_avg:', g1_top_avg)
    print('g2_top_avg:', g2_top_avg)
#     print('g1_worst:', g1_worst)
#     print('g2_worst:', g2_worst)
    print('')
    return [g1_top_avg, g2_top_avg]
# get_violations([0,0])


# In[34]:


from scipy.optimize import root
root(fun = get_violations, x0 = np.array([-17.75871892492204,-0.0008645125529041619]), method='diagbroyden', options = {'maxiter' : 15})


# ## Generate random MC episodes for standardizing training set

# In[4]:


# Generate data
Q_network_training_data = []
for i in range(10000):
    initialize_MDP_BioEnv() # Initialize system with set of uncertain constants
    episode = generate_random_episode(initial_state = generate_initial_state()) # Generate 1 episode
    data = extract_data_from_episode(episode, discount_factor = 0.9)   # Extract datapoint from episode
    
    Q_network_training_data += data # Add datapoints to training set
Q_network_training_data


# In[187]:


# Random policy score
total_score = 0
for i in range(100):
    total_score += generate_random_episode(initial_state = generate_initial_state())[-1][-1]
print('Objective score of random policy:', total_score/100)


# In[5]:


# Extract data
state_and_action = [] # A list of lists for input
Q_value          = [] # A list of lists for target values 

for datapoint in Q_network_training_data: # Iterate over states
    s_a, Q            = datapoint # Unpack [Cx, Cn, Cq, t, u_L, u_Fn], Q
    state_and_action += [s_a]
    Q_value          += [[Q]]

state_and_action = np.array(state_and_action) # Convert list to tensor
Q_value          = np.array(Q_value)          # Convert list to tensor

# state_and_action[:3], Q_value[:3] # View first three examples
state_and_action.shape, Q_value.shape


# In[6]:


# Standardize data
X = state_and_action
y = Q_value

scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)

# Get mean and std
x_mean_Q = scaler_X.mean_
x_std_Q  = scaler_X.scale_
y_mean_Q = scaler_y.mean_
y_std_Q  = scaler_y.scale_

# Scale x
X_scaled = np.zeros(X.shape)
for feature in range(len(x_mean_Q)): 
    X_scaled[:,feature] = (X[:,feature] - x_mean_Q[feature])/x_std_Q[feature]

# Scale y
y_scaled = (y - y_mean_Q)/y_std_Q

# Convert into tensors
X_scaled = torch.tensor(X_scaled)
y_scaled = torch.tensor(y_scaled)
X_scaled, y_scaled


# In[7]:


X_scaled.shape, y_scaled.shape


# ## Pre-training the Q-network with MC samples

# In[183]:


Q_net = torch.nn.Sequential(
        torch.nn.Linear(6,   200, bias = True), # 6 input nodes
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200,   1, bias = True), # 1 output node
    ).double()


# In[184]:


# Define dataset
torch_dataset = torch.utils.data.TensorDataset(X_scaled, y_scaled)

# def weights_init_uniform(m): # For initializing uniform weights
#     classname = m.__class__.__name__
#     # for every Linear layer in a model..
#     if classname.find('Linear') != -1:
#         # apply a uniform distribution to the weights and a bias=0
#         m.weight.data.uniform_(0.0, 1.0)
#         m.bias.data.fill_(0)

#Define a network
Q_net = torch.nn.Sequential(
        torch.nn.Linear(6,   200, bias = True), # 6 input nodes
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200,   1, bias = True), # 1 output node
    ).double()

# Q_net.apply(weights_init_uniform) # Initialize uniform weights

optimizer = torch.optim.Adam(Q_net.parameters(), lr = 1e-3) # Initialize Adam optimizer
loss_func = torch.nn.SmoothL1Loss()  # this is for regression mean squared loss

BATCH_SIZE = 1000 # Batch size
EPOCH      = 50   # No. of epochs

# Splitting dataset into batches
loader = torch.utils.data.DataLoader(dataset     = torch_dataset, 
                         batch_size  = BATCH_SIZE, 
                         shuffle     = True, 
                         num_workers = 0)

# my_images = []  # For saving into gif
fig, ax = plt.subplots(figsize=(10,7))

epoch_list = [] # Save epoch for plotting
loss_list  = [] # Save loss for plotting

# Start training
for epoch in range(EPOCH):
    epoch_list += [epoch]
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step

        prediction = Q_net(batch_x.double())          # input x and predict based on x

        loss       = loss_func(prediction, batch_y) # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if step == 1:
            # plot and show learning process
            plt.cla()
            loss_list += [loss.data.numpy()]
            plt.plot(epoch_list, loss_list, label = 'Latest Loss = %.4f' % loss.data.numpy(), color = 'r')
            plt.xlabel('Epoch', fontsize = 20)
            plt.ylabel('Loss', fontsize = 20)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.legend(fontsize = 20, frameon= 0)
            
            display.clear_output(wait = True) #these two lines plots the data in real time
            display.display(fig)
            print('Current Epoch =', epoch)
            print('Latest Loss =', loss.data.numpy())


# In[185]:


predictions = (Q_net(X_scaled).detach().numpy() * y_std_Q) + y_mean_Q
predictions[:10], y[:10]


# In[186]:


# Plot histogram of errors
residual = predictions - y
fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(residual)
ax.set(xlabel= 'Residual', ylabel = 'Number of samples')
plt.show()


# In[2751]:


score_NN_policy(GA_optimize, [1.0, 150.0, 0, 0], num_iterations = 20)


# ## Training the UNCONSTRAINED Q-network

# In[1427]:


# Initialize Q-network
Q_net = torch.nn.Sequential(
        torch.nn.Linear(6,   200, bias = True), # 6 input nodes
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200,   1, bias = True), # 1 output node
    ).double()


# In[1429]:


# ========== 1) GENERATING INITIAL BATCH OF SAMPLES FOR REPLAY BUFFER ==========
epsilon       = 0.99
replay_buffer = collections.deque(maxlen = 3000) # Max capacity of 3000

for i in range(100): # Generate 100 episodes
    episode       = generate_episode_with_NN(NN = GA_optimize,
                                             initial_state = [1.0, 150.0, 0, 0],
                                             epsilon = epsilon)
    replay_buffer = update_replay_buffer(replay_buffer, episode, discount_factor = 0.9) # Add to replay buffer
print(len(replay_buffer))

# ================================= 2) TRAINING THE Q-NETWORK =================================
epsilon            = 0.99 # Initial epsilon
score_list         = [] # For plotting score of policy visited over time
iteration_list     = []
policy_mean_list   = [] # For plotting evolution of policy visited over time
policy_std_list    = [] # For plotting evolution of policy visited over time
training_loss_list = [] # For plotting loss of NN (after 20 epochs) over time
episode_bank       = [] # A list of list of lists - Tertiary list  -> Iterations
                        #                         - Secondary list -> Episodes
                        #                         - Primary list   -> Steps
                        #                         For plotting evolution of states visited over time            

for iteration in range(1000):
    # SCORE EXISTING POLICY
    print('=========== TRAINING ITERATION %.0f ===========' % iteration)
    print('Current epsilon = ', epsilon)
    current_score = score_NN_policy(GA_optimize, [1.0, 150.0, 0, 0], num_iterations = 10)
    print('AVERAGE SCORE = '  , current_score)
    print('')
    
    # GENERATE EPISODES & ADD TO REPLAY BUFFER
    for j in range(100):
        episode       = generate_episode_with_NN(NN = GA_optimize,
                                             initial_state = [1.0, 150.0, 0, 0],
                                             epsilon = epsilon)
        replay_buffer = update_replay_buffer(replay_buffer, episode, discount_factor = 0.9) # Add to replay buffer
    
    # SAMPLE EPISODES FROM REPLAY BUFFER
    combined_inputs  = [] # List of lists to be converted into input tensor
    combined_targets = [] # List to be converted into target tensor
    samples          = random.sample(replay_buffer, 100) # Draw random samples from replay buffer
    
    for inputs, target in samples:
        combined_inputs  += [inputs]
        combined_targets += [[target]]
    combined_inputs  = torch.tensor(combined_inputs)  # Convert list to tensor
    combined_targets = torch.tensor(combined_targets) # Convert list to tensor
    
    # TRAIN THE NEURAL NETWORK
    optimizer = torch.optim.Adam(Q_net.parameters(), lr=1e-3) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    

    for epoch in range(20):
        prediction = Q_net(combined_inputs)                    # Input x and predict based on x
        loss       = loss_func(prediction, combined_targets) # Must be (1. nn output, 2. target)

        optimizer.zero_grad()   # Clear gradients for next train
        loss.backward()         # Backpropagation, compute gradients
        optimizer.step()        # Apply gradients
#         print('Epoch = ', epoch, 'Loss = %.4f' % loss.data.numpy())

    score_list         += [current_score] # Store score for each iteration
    training_loss_list += [loss.data.numpy()]                                # Store training loss after 20 epochs for each iteration
    iteration_list     += [iteration]                                        # Store iteration
    
    # INVESTIGATING EVOLUTION OF POLICY OVER TRAINING ITERATIONS
    if iteration in np.arange(0,5000,10): # Every 10 iterations
        list_of_policies  = score_NN_policy(GA_optimize, [1.0, 150.0, 0, 0], num_iterations = 100, get_control = True)[1] # add policies
        policy_mean       = np.mean(list_of_policies, axis = 0) # Calculate mean of policies
        policy_std        = np.std(list_of_policies, axis = 0)  # Calculate std of policies
        policy_mean_list += [policy_mean] # Save mean policy for plotting
        policy_std_list  += [policy_std]  # Save std of policy for plotting
    
    # INVESTIGATING EVOLUTION OF STATES VISITED OVER TRAINING ITERATIONS
    if iteration in np.arange(0, 5000, 10): # Every 10 iterations
        pool_of_episodes = [] # A *list of lists* of episodes
        for i in range(10):   # Generate 10 sample episodes and store them
            pool_of_episodes += [generate_episode_with_NN(NN = GA_optimize,
                                             initial_state = [1.0, 150.0, 0, 0],
                                             epsilon = 0)]
        episode_bank         += [pool_of_episodes]
    
    epsilon *= 0.99 # Decay epsilon


# In[2975]:


# # Save neural network
# torch.save(Q_net, './NN_models/CS1_Q_net_unconstrained')

# # Load pre-trained neural network
# Q_net = torch.load('./NN_models/CS1_Q_net_unconstrained')
# Q_net.eval()


# In[2753]:


# Save data
# with open('./Data/CS1_score_GA_unconstrained', 'wb') as f:
#     pickle.dump(score_list, f, pickle.HIGHEST_PROTOCOL)
# with open('./Data/CS1_loss_GA_unconstrained', 'wb') as f:
#     pickle.dump(training_loss_list, f, pickle.HIGHEST_PROTOCOL)

# Load pre-run data
# with open('./Data/CS1_score_GA_unconstrained', 'rb') as f:
#     score_list = pickle.load(f)
# with open('./Data/CS1_loss_GA_unconstrained', 'rb') as f:
#     training_loss_list = pickle.load(f)

# Plot score vs. training iteration
fig, ax = plt.subplots(figsize=(8,6))
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(which='major', length=10, width=1, direction='out')
ax.tick_params(which='minor', length=4, width=1, direction='out')
plt.plot(iteration_list, score_list, color = 'b')
plt.legend(frameon =  0)
plt.ylabel('Average score of policy', fontsize = 20)
plt.xlabel('Training iteration', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()

# Plot training loss vs. training iteration
fig, ax = plt.subplots(figsize=(8,6))
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(which='major', length=10, width=1, direction='out')
ax.tick_params(which='minor', length=4, width=1, direction='out')
plt.plot(iteration_list, training_loss_list, color = 'r')
plt.legend(frameon =  0)
plt.ylabel('Training loss (after 20 epochs)', fontsize = 20)
plt.xlabel('Training iteration', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()


# In[3089]:


episode = generate_episode_with_NN(NN = max_action, initial_state = [1.0, 150.0, 0, 0], epsilon = 0)
plot_episode(episode) 


# In[1298]:


score_NN_policy(GA_optimize, [1.0, 150.0, 0, 0], num_iterations = 100)


# ## Genetic algorithm

# In[8]:


# Source: Ahmed Gad - https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/master/Tutorial%20Project
def cal_pop_fitness(population, initial_state):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    state = []
    population = population.tolist() # convert 2D array to list of lists
    for i in range(len(population)):
        s = initial_state + population[i] # Append action to state 
        s = standardize_state_Q(s) # Standardize
        state += [s]
    state = torch.tensor(state).double()
    fitness = Q_net(state).detach().numpy()
    return fitness

def cal_pop_fitness_constrained(population, initial_state, g1_threshold, g2_threshold, g3_threshold):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    
    # CALCULATE Q-FITNESS
    state = []
    population = population.tolist() # convert 2D array to list of lists
    for i in range(len(population)):
        s = initial_state + population[i] # Append action to state 
        s = standardize_state_Q(s) # Standardize using Q-network mean & std
        state += [s]
    state = torch.tensor(state).double()
    fitness_Q = (Q_net(state).detach().numpy() * y_std_Q) + y_mean_Q # Unstandardized Q-network fitness
#     print('fitness_Q:', fitness_Q)
    # CALCULATE C-FITNESS 
    state = []
    
    initial_state_for_C = copy.deepcopy(initial_state) 
    initial_state_for_C[3] = 240. - initial_state_for_C[3] #### NEED TO TRANSFORM T TO T_F- T #####
#     print('population:', population)
    for i in range(len(population)):
        s = initial_state_for_C + population[i] # Append action to state 
#         print('non-standardized state:',s)
        s = standardize_state_C(s) # Standardize using C-network mean & std
#         print('standardized state:',s)
        state += [s]
    state = torch.tensor(state).double()
    state_copy = copy.deepcopy(state) # Just to be sure
    state_copy2 = copy.deepcopy(state) # Just to be sure
    
    fitness_g1 = -1*((g1_net(state).detach().numpy() * y_std_C[0]) + y_mean_C[0])  # Unstandardized C-network fitness - NEGATIVE SIGN because -ve g1_net values denote higher fitness!
    fitness_g2 = -1*((g2_net(state_copy).detach().numpy() * y_std_C[1]) + y_mean_C[1])
    fitness_g3 = -1*((g3_net(state_copy2).detach().numpy() * y_std_C[2]) + y_mean_C[2])
#     print('fitness_g1:', fitness_g1)
#     print('fitness_g2:', fitness_g2)
#     print('fitness_g3:', fitness_g3)
    # COMBINE Q-FITNESS AND C-FITNESS
    fitness_overall = []
    for i in range(len(population)):
        
        # include gl and g2 only for now
        f = max(0, fitness_Q[i].item()) + 100*min(0, fitness_g1[i].item() + g1_threshold) + 10000*min(0, fitness_g2[i].item() + g2_threshold) + 100*min(0, fitness_g3[i].item() + g3_threshold)
        fitness_overall += [f]
    
    return fitness_overall

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999     # Make the current parent the most unfit
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2) # Returns 1 in this case (because 2 values)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, mut_percen):

    for idx in range(offspring_crossover.shape[0]): # Iterate over individual offspring
        for feature in range(offspring_crossover.shape[1]): # Iterate over its features 
            r = np.random.uniform(0, 1)
            
            if r < mut_percen: # If mutation occurs
#                 print('Mutated')
                if feature == 0: # If u_L 
                    offspring_crossover[idx, feature] = np.random.uniform(120, 400) # Pick random u_L
                else:            # Else u_Fn
                    offspring_crossover[idx, feature] = np.random.uniform(0, 40) # Pick random u_Fn
    return offspring_crossover

def generate_random_episode_constrained(initial_state, g1_threshold = 0): 
    '''Generates an episode using RANDOM policy
    threshold is the tolerance of the limit, where 
    0            : Limit itself (as per defined in extract_constraint_values_from_episode() function)
    +ve threshold: Looser limit   
    -ve threshold: Stricter limit 
    '''
    episode = []
    state = initial_state  # Initial state
    for i in range(13):   # Take (12 + 1) steps
        predicted_c = np.inf # Initialize constraint
        while predicted_c > g1_threshold: # Keep choosing a different action this until constraint is LESS THAN zero
            old_state = state # Old state for storing into episode
            old_state_without_action = old_state
            old_state = list(old_state)
            old_state[3] = 240.0 - old_state[3] # Convert to t to (t - t_f)
            u_L  = np.random.uniform(120, 400) # Pick random u_L
            u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
            action = [u_L, u_Fn]
            old_state += action     # Append action to state

            old_state_standardized = old_state.copy()
            old_state_standardized = standardize_state_C(old_state_standardized)

            old_state_standardized = torch.tensor(old_state_standardized)
            predicted_c = ((g1_net(old_state_standardized).detach().numpy())*y_std_C) + y_mean_C # Update predicted constraint
#             print('predicted_c:', predicted_c)
        state, reward  = transition(state, action)     # Evolve to get new state 
        episode       += [[list(old_state_without_action), action, reward]] # Update step
    return episode
# generate_random_episode_constrained(initial_state = [1.0,150.0,0.,0.], g1_threshold = 0)

def GA_optimize(state, mut_percen = 0.1):
    '''Argument: State [Cx, Cn, Cq, t]
       Output: Control [u_L, u_Fn] that maximizes Q 
    '''
    num_features       = 2  # Number of features (u_L and u_Fn)
    sol_per_pop        = 10 # Number of individuals per population
    num_parents_mating = 5  # Numbers of parents selected from each generation
    num_generations    = 10 # Number of generations
    
    mut_percen = mut_percen # Initial probability of mutation

    #Creating the initial population.
    new_pop = []
    for i in range(200): # Initial pop
        u_L  = np.random.uniform(120, 400) # Pick random u_L
        u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
        new_pop += [[u_L, u_Fn]]
    new_pop = np.array(new_pop)
#     print('new_pop',new_pop)

    decay_factor = (0.01/mut_percen)**(1/num_generations) # Define decay factor for mut_percen

    state = state
    avg_fitness_list = [] # Store average fitness scores
    max_fitness_list = [] # Store max fitness scores
    generation_list = []  # Store generation index

    for generation in range(num_generations):
        
        # Measuring the fitness of each chromosome in the population.
        fitness = cal_pop_fitness(new_pop, state)

        avg_fit = np.mean(fitness) # Average fitness
        max_fit = np.max(fitness)  # Max fitness of the generation
    #     print('==== Generation %.0f ====' % generation)
    #     print('Average fitness:', avg_fit)
    #     print('Max fitness:', max_fit)
    #     print('')
        avg_fitness_list += [avg_fit]    # Save average fitness score for plotting
        max_fitness_list += [max_fit]    # Save average fitness score for plotting
        generation_list  += [generation] # Save generation index

        # Selecting the best parents in the population for mating
        parents = select_mating_pool(new_pop, fitness, num_parents_mating)
#         print('parents:',parents)

        # Generating next generation using crossover
        offspring_crossover = crossover(parents, offspring_size=(sol_per_pop-parents.shape[0], num_features))
#         print('offspring_crossover',offspring_crossover)

        offspring_mutation = mutation(offspring_crossover, mut_percen = mut_percen)
#         print('offspring_mutation',offspring_mutation)

        # Creating the new population based on the parents and offspring
        new_pop[0:parents.shape[0], :] = parents
        new_pop[parents.shape[0]:sol_per_pop, :] = offspring_mutation

        # Cull excess initial population
        new_pop = new_pop[0:sol_per_pop,:]

    #     mut_percen = 0.5*e**(-1*generation/(num_generations/3)) # Decay
        mut_percen *= decay_factor
    
    max_action = new_pop[0] # Select the fittest action from final population
    return max_action
# GA_optimize([1.0, 150.0, 0, 0], mut_percen = 0.25) # Test function

def GA_optimize_constrained(state, g1_threshold, g2_threshold, g3_threshold, init_mut_percen = 0.01):
    '''Argument: State [Cx, Cn, Cq, t]
       Output: Control [u_L, u_Fn] that maximizes Q and avoids violation of constraints
    '''
    num_features       = 2  # Number of features (u_L and u_Fn)
    sol_per_pop        = 10 # Number of individuals per population
    num_parents_mating = 5  # Numbers of parents selected from each generation
    num_generations    = 10 # Number of generations
    
    max_fitness = -np.inf # Initialize fitness
    
    while max_fitness < 0: # Keep running this loop until no violations (no -ve fitness) occurs
    
        mut_percen = init_mut_percen # Initial probability of mutation

        #Creating the initial population.
        new_pop = []
        for i in range(200): # Initial pop
            u_L  = np.random.uniform(120, 400) # Pick random u_L
            u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
            new_pop += [[u_L, u_Fn]]
        new_pop = np.array(new_pop)
    #     print('new_pop',new_pop)

        decay_factor = (0.01/mut_percen)**(1/num_generations) # Define decay factor for mut_percen

        state = state
        avg_fitness_list = [] # Store average fitness scores
        max_fitness_list = [] # Store max fitness scores
        generation_list = []  # Store generation index

        for generation in range(num_generations):

            # Measuring the fitness of each chromosome in the population.
            fitness = cal_pop_fitness_constrained(new_pop, state, g1_threshold = g1_threshold, g2_threshold = g2_threshold, g3_threshold = g3_threshold)

            avg_fit = np.mean(fitness) # Average fitness
            max_fit = np.max(fitness)  # Max fitness of the generation
        #     print('==== Generation %.0f ====' % generation)
        #     print('Average fitness:', avg_fit)
        #     print('Max fitness:', max_fit)
        #     print('')
            avg_fitness_list += [avg_fit]    # Save average fitness score for plotting
            max_fitness_list += [max_fit]    # Save average fitness score for plotting
            generation_list  += [generation] # Save generation index

            # Selecting the best parents in the population for mating
            parents = select_mating_pool(new_pop, fitness, num_parents_mating)
    #         print('parents:',parents)

            # Generating next generation using crossover
            offspring_crossover = crossover(parents, offspring_size=(sol_per_pop-parents.shape[0], num_features))
    #         print('offspring_crossover',offspring_crossover)

            offspring_mutation = mutation(offspring_crossover, mut_percen = mut_percen)
    #         print('offspring_mutation',offspring_mutation)

            # Creating the new population based on the parents and offspring
            new_pop[0:parents.shape[0], :] = parents
            new_pop[parents.shape[0]:sol_per_pop, :] = offspring_mutation

            # Cull excess initial population
            new_pop = new_pop[0:sol_per_pop,:]

        #     mut_percen = 0.5*e**(-1*generation/(num_generations/3)) # Decay
    #         mut_percen *= decay_factor

        max_action = new_pop[0] # Select the fittest action from final population
        max_fitness = max_fitness_list[-1] # Update max_fitness
#     plt.figure()
#     plt.plot(generation_list, avg_fitness_list, label = 'Avg (%.3f)'%avg_fitness_list[-1])
#     plt.plot(generation_list, max_fitness_list, label = 'Max (%.3f)'%max_fitness_list[-1])
#     plt.legend()
#     plt.show()
#     print(max_fitness)
    
    return max_action
# GA_optimize_constrained([1.0, 150.0, 0, 0], g1_threshold = g1_threshold, g2_threshold = g2_threshold, g3_threshold = g3_threshold ) # Test function

# def GA_optimize_constrained(state, g1_threshold, g2_threshold, g3_threshold, init_mut_percen = 0.01):
#     '''Argument: State [Cx, Cn, Cq, t]
#        Output: Control [u_L, u_Fn] that maximizes Q and avoids violation of constraints
#     '''
#     max_fitness = -np.inf # Initialize fitness
    
#     while max_fitness < 0: # Keep running this loop until no violations (no -ve fitness) occurs
    
#         #Creating the initial population.
#         new_pop = []
#         for i in range(1000): # Initial pop
#             u_L  = np.random.uniform(120, 400) # Pick random u_L
#             u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
#             new_pop += [[u_L, u_Fn]]
            
#         new_pop = np.array(new_pop)

#         fitness = cal_pop_fitness_constrained(new_pop, state, g1_threshold = g1_threshold, g2_threshold = g2_threshold, g3_threshold = g3_threshold)
#         max_idx = np.argmax(fitness)
#         max_action  = new_pop[max_idx]
#         max_fitness = fitness[max_idx]
            
#     return max_action
# GA_optimize_constrained([1.0, 150.0, 0, 0], g1_threshold = 1000, g2_threshold = 100, g3_threshold = 0) # Test function


# In[4]:


episode = generate_episode_with_NN(GA_optimize_constrained, [1.0, 150.0, 0, 0], epsilon = 0, 
                                   g1_threshold = g1_threshold,
                                   g2_threshold = g2_threshold,
                                   g3_threshold = g3_threshold)
episode


# In[198]:


#test constrained
sol_per_pop = 200 # Number of individuals per population
num_features = 2 # Number of features (u_L and u_Fn)
num_parents_mating = 2
num_generations = 10
mut_percen = 0.1
state = [1.0, 150.0, 0, 0]
new_pop = []
for i in range(sol_per_pop): # Initial pop
    u_L  = np.random.uniform(120, 400) # Pick random u_L
    u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
    new_pop += [[u_L, u_Fn]]
new_pop = np.array(new_pop)
print(new_pop)


# In[199]:


fitness = cal_pop_fitness_constrained(new_pop, [1.0, 150.0, 0, 0], 
                                      g1_threshold = 500, 
                                      g2_threshold = 0.004,
                                      g3_threshold = 0)
fitness


# In[440]:


GA_optimize_constrained([1.0, 150.0, 0, 0], g1_threshold = 0, g2_threshold = 0, g3_threshold = 0,, init_mut_percen = 0.01)


# In[1352]:


sol_per_pop = 10 # Number of individuals per population
num_features = 2 # Number of features (u_L and u_Fn)
mut_percen = 0.1

#Creating the initial population.
new_pop = []
for i in range(200): # Initial pop
    u_L  = np.random.uniform(120, 400) # Pick random u_L
    u_Fn = np.random.uniform(0, 40)    # Pick random u_Fn
    new_pop += [[u_L, u_Fn]]
new_pop = np.array(new_pop)
print('new_pop',new_pop)

num_generations = 10
decay_factor = (0.01/mut_percen)**(1/num_generations) # Define decay factor for mut_percen
num_parents_mating = 5
state = [1.0, 150.0, 0, 0]
avg_fitness_list = []
max_fitness_list = []
generation_list = []

for generation in range(num_generations):
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(new_pop, state)
    
    avg_fit = np.mean(fitness)
    max_fit = np.max(fitness)
#     print('==== Generation %.0f ====' % generation)
#     print('Average fitness:', avg_fit)
#     print('Max fitness:', max_fit)
#     print('')
    avg_fitness_list += [avg_fit]    # Save average fitness score for plotting
    max_fitness_list += [max_fit]    # Save average fitness score for plotting
    generation_list  += [generation] # Save generation index
    
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_pop, fitness, num_parents_mating)
    print('parents:',parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size=(sol_per_pop-parents.shape[0], num_features))
    print('offspring_crossover',offspring_crossover)

    offspring_mutation = mutation(offspring_crossover, mut_percen = mut_percen)
    print('offspring_mutation',offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_pop[0:parents.shape[0], :] = parents
    new_pop[parents.shape[0]:sol_per_pop, :] = offspring_mutation

    # Cull excess
    new_pop = new_pop[0:sol_per_pop,:]
    
#     mut_percen = 0.5*e**(-1*generation/(num_generations/3)) # Decay
#     mut_percen *= decay_factor
    print('Mut_percen:', mut_percen)
    print('new_pop',new_pop)
plt.figure()
plt.plot(generation_list, avg_fitness_list, label = 'Avg (%.3f)'%avg_fitness_list[-1])
plt.plot(generation_list, max_fitness_list, label = 'Max (%.3f)'%avg_fitness_list[-1])
plt.legend()
plt.show()


# In[26]:


# Inspecting the spread of actions predicted by GA
start = time.time()  # For timing purposes
u_L_pool  = []
u_Fn_pool = []
for i in range(500):
    u_L, u_Fn = GA_optimize_constrained([1.0, 150.0, 0, 0], g1_threshold = 0, g2_threshold = 0, g3_threshold = 0,init_mut_percen = 0.01)
    u_L_pool  += [u_L]
    u_Fn_pool += [u_Fn]
end = time.time()  # For timing purposes
print(end - start)
plt.figure()
plt.scatter(u_L_pool, u_Fn_pool, s = 0.5)
plt.scatter(np.mean(u_L_pool), np.mean(u_Fn_pool), color = 'r')
plt.xlim(120, 400)
plt.ylim(0,40)
plt.xlabel('u_L')
plt.ylabel('u_Fn')


# In[2513]:


# Inspecting the spread of predicted max_actions
start = time.time()  # For timing purposes
u_L_pool  = []
u_Fn_pool = []
for i in range(1000):
    u_L, u_Fn = max_action([1.0, 150.0, 0, 0])
    u_L_pool += [u_L]
    u_Fn_pool += [u_Fn]
end = time.time()  # For timing purposes
print(end - start)
plt.figure()
plt.scatter(u_L_pool, u_Fn_pool, s = 0.5)
plt.scatter(np.mean(u_L_pool), np.mean(u_Fn_pool), color = 'r')
# plt.xlim(120, 250)
# plt.ylim(35,41)
plt.xlabel('u_L')
plt.ylabel('u_Fn')


# # Introducing constraints
# The goal is to train a 2nd neural network to predict a constraint value given a state-action pair. The data used to train this NN will generated using a random policy in the cells below.

# ### Method 1: Training Q-network and C-network separately

# In[9]:


# Generate UNCONSTRAINED data for training the constraint neural network

start = time.time()  # For timing purposes
state_action = []    # To store state and action [Cx, Cn, Cq, t-t_f, u_L, u_Fn]
g1_constraint   = []    # To store g1 constraint 
g2_constraint   = []    # To store g2 constraint 
g3_constraint   = []    # To store g3 constraint 
num_episodes = 10000 # Number of episodes to generate

for j in range(num_episodes):
    initialize_MDP_BioEnv() # Initialize system
    generated_episode = generate_random_episode(initial_state = generate_initial_state())  # Generate one episode
    s_a, g1, g2, g3   = extract_constraint_values_from_episode(generated_episode)            # Extract data from episode
    
    state_action += s_a             # Append state and action 
    g1_constraint   += g1           # Append g1 constraint
    g2_constraint   += g2           # Append g2 constraint
    g3_constraint   += g3           # Append g2 constraint

end = time.time() # For timing purposes

state_action = np.array(state_action)
g1_constraint   = np.array(g1_constraint)
g2_constraint   = np.array(g2_constraint)
g3_constraint   = np.array(g3_constraint)
print('')
print('Time taken:', end - start)
print('==========INPUT [Cx, Cn, Cq, t_f-t, u_L, u_Fn]==========')
print(state_action)
print('')
print('==========TARGET VALUES [g1]==========')
print(g1_constraint)
print('==========TARGET VALUES [g2]==========')
print(g2_constraint)
print('==========TARGET VALUES [g3]==========')
print(g3_constraint)


# In[10]:


fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(g1_constraint)
ax.set(xlabel= '$g_1$', ylabel = 'Fraction of samples')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(g2_constraint)
ax.set(xlabel= '$g_2$', ylabel = 'Fraction of samples')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(g3_constraint)
ax.set(xlabel= '$g_3$', ylabel = 'Fraction of samples')
plt.legend()
plt.show()


# In[11]:


# Combine g1_constraint,g2_constraint and g3_constraint into a (n x 3) array for standardization purposes
g1g2g3_constraint = []
for i in range(len(g1_constraint)):
    g1g2g3_constraint += [[g1_constraint[i][0], g2_constraint[i][0], g3_constraint[i][0]]]
g1g2g3_constraint = np.array(g1g2g3_constraint)
g1g2g3_constraint


# In[12]:


# Standardize data for C-network
X = state_action
y = g1g2g3_constraint

scaler_X = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)

# Get mean and std
x_mean_C = scaler_X.mean_
x_std_C  = scaler_X.scale_
y_mean_C = scaler_y.mean_
y_std_C  = scaler_y.scale_

# Scale x
X_scaled = np.zeros(X.shape)
for feature in range(len(x_mean_C)): 
    X_scaled[:,feature] = (X[:,feature] - x_mean_C[feature])/x_std_C[feature]

# Scale g1
g1_scaled = np.zeros([y.shape[0],1])
g1_scaled[:] = (g1_constraint[:] - y_mean_C[0])/y_std_C[0] # Use 0th index of y_mean and y_std

# Scale g2
g2_scaled = np.zeros([y.shape[0],1])
g2_scaled[:] = (g2_constraint[:] - y_mean_C[1])/y_std_C[1] # Use 1st index of y_mean and y_std

# Scale g3
g3_scaled = np.zeros([y.shape[0],1])
g3_scaled[:] = (g3_constraint[:] - y_mean_C[2])/y_std_C[2] # Use 1st index of y_mean and y_std


# Convert into tensors
X_scaled = torch.tensor(X_scaled)
g1_scaled = torch.tensor(g1_scaled)
g2_scaled = torch.tensor(g2_scaled)
g3_scaled = torch.tensor(g3_scaled)
X_scaled, g1_scaled, g2_scaled, g3_scaled


# In[13]:


len(X_scaled), len(g1_scaled), len(g2_scaled), len(g3_scaled)


# ### Training g1-net

# In[195]:


g1_net = torch.nn.Sequential(
        torch.nn.Linear(6, 200, bias = True), # 6 input nodes (Cx, Cn, Cq, t_f - t, u_L, u_Fn)
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 100, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 50, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 1, bias = True), # 1 output node
    ).double()


# In[196]:


# Define dataset
torch_dataset = torch.utils.data.TensorDataset(X_scaled, g1_scaled)

optimizer = torch.optim.Adam(g1_net.parameters(), lr = 1e-4, amsgrad= True) # Initialize Adam optimizer
loss_func = torch.nn.SmoothL1Loss()  # this is for regression mean squared loss

BATCH_SIZE = 200 # Batch size
EPOCH      = 500   # No. of epochs

# Splitting dataset into batches
loader = torch.utils.data.DataLoader(dataset     = torch_dataset, 
                         batch_size  = BATCH_SIZE, 
                         shuffle     = True, 
                         num_workers = 0)

# my_images = []  # For saving into gif
fig, ax = plt.subplots(figsize=(10,7))

epoch_list = [] # Save epoch for plotting
loss_list  = [] # Save loss for plotting

# Start training
for epoch in range(EPOCH):
    epoch_list += [epoch]
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step

        prediction = g1_net(batch_x.double())          # input x and predict based on x

        loss       = loss_func(prediction, batch_y) # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if step == 1:
            # plot and show learning process
            plt.cla()
            loss_list += [loss.data.numpy()]
            plt.plot(epoch_list, loss_list, label = 'Latest Loss = %.4f' % loss.data.numpy(), color = 'r')
            plt.xlabel('Epoch', fontsize = 20)
            plt.ylabel('Loss', fontsize = 20)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.legend(fontsize = 20, frameon= 0)
            
            display.clear_output(wait = True) #these two lines plots the data in real time
            display.display(fig)
            print('Current Epoch =', epoch)
            print('Latest Loss =', loss.data.numpy())


# Very poor prediction of first transition [1.0, 150.0, 0, 240, u_L, u_Fn] --> cannot predict -ve values of g1

# In[220]:


print('g1_net predictions of g1:',  (g1_net(X_scaled[0:200:12]).detach().numpy()*y_std_C[0])+y_mean_C[0]   )
print('Actual labels:',  g1_constraint[0:200:12]   )


# ### Training g2-net

# In[206]:


g2_net = torch.nn.Sequential(
        torch.nn.Linear(6, 200, bias = True), # 6 input nodes (Cx, Cn, Cq, t_f - t, u_L, u_Fn)
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 100, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 50, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 1, bias = True), # 1 output node
    ).double()


# In[207]:


# for overfitting
# Define dataset
torch_dataset = torch.utils.data.TensorDataset(X_scaled[:1000], g2_scaled[:1000])

optimizer = torch.optim.Adam(g2_net.parameters(), lr = 1e-3, amsgrad= True) # Initialize Adam optimizer
loss_func = torch.nn.SmoothL1Loss()  # this is for regression mean squared loss


EPOCH      = 1000   # No. of epochs

# my_images = []  # For saving into gif
fig, ax = plt.subplots(figsize=(10,7))

epoch_list = [] # Save epoch for plotting
loss_list  = [] # Save loss for plotting

# Start training
for epoch in range(EPOCH):
    epoch_list += [epoch]
    

    prediction = g2_net(X_scaled[:1000])          # input x and predict based on x

    loss       = loss_func(prediction, g2_scaled[:1000]) # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients


            # plot and show learning process
    plt.cla()
    loss_list += [loss.data.numpy()]
    plt.plot(epoch_list, loss_list, label = 'Latest Loss = %.4f' % loss.data.numpy(), color = 'r')
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20, frameon= 0)
            
    display.clear_output(wait = True) #these two lines plots the data in real time
    display.display(fig)
    print('Current Epoch =', epoch)
    print('Latest Loss =', loss.data.numpy())


# In[488]:


# Define dataset
torch_dataset = torch.utils.data.TensorDataset(X_scaled, g2_scaled)

optimizer = torch.optim.Adam(g2_net.parameters(), lr = 1e-4, amsgrad= True) # Initialize Adam optimizer
loss_func = torch.nn.SmoothL1Loss()  # this is for regression mean squared loss

BATCH_SIZE = 200 # Batch size
EPOCH      = 500   # No. of epochs

# Splitting dataset into batches
loader = torch.utils.data.DataLoader(dataset     = torch_dataset, 
                         batch_size  = BATCH_SIZE, 
                         shuffle     = True, 
                         num_workers = 0)

# my_images = []  # For saving into gif
fig, ax = plt.subplots(figsize=(10,7))

epoch_list = [] # Save epoch for plotting
loss_list  = [] # Save loss for plotting

# Start training
for epoch in range(EPOCH):
    epoch_list += [epoch]
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step

        prediction = g2_net(batch_x.double())          # input x and predict based on x

        loss       = loss_func(prediction, batch_y) # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if step == 1:
            # plot and show learning process
            plt.cla()
            loss_list += [loss.data.numpy()]
            plt.plot(epoch_list, loss_list, label = 'Latest Loss = %.4f' % loss.data.numpy(), color = 'r')
            plt.xlabel('Epoch', fontsize = 20)
            plt.ylabel('Loss', fontsize = 20)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.legend(fontsize = 20, frameon= 0)
            
            display.clear_output(wait = True) #these two lines plots the data in real time
            display.display(fig)
            print('Current Epoch =', epoch)
            print('Latest Loss =', loss.data.numpy())


# Poor prediction by g2_net in general (as shown by the high losses above)

# In[228]:


print('g2_net predictions of g2:',  (g2_net(X_scaled[20:40]).detach().numpy()*y_std_C[1])+y_mean_C[1]   )
print('Actual labels:',  g2_constraint[20:40]   )


# ### Training g3-net

# In[204]:


g3_net = torch.nn.Sequential(
        torch.nn.Linear(6, 200, bias = True), # 6 input nodes (Cx, Cn, Cq, t_f - t, u_L, u_Fn)
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 100, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 50, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 1, bias = True), # 1 output node
    ).double()


# In[205]:


# for overfitting
# Define dataset
torch_dataset = torch.utils.data.TensorDataset(X_scaled[:1000], g3_scaled[:1000])

optimizer = torch.optim.Adam(g3_net.parameters(), lr = 1e-3, amsgrad= True) # Initialize Adam optimizer
loss_func = torch.nn.SmoothL1Loss()  # this is for regression mean squared loss

EPOCH      = 1000   # No. of epochs

# my_images = []  # For saving into gif
fig, ax = plt.subplots(figsize=(10,7))

epoch_list = [] # Save epoch for plotting
loss_list  = [] # Save loss for plotting

# Start training
for epoch in range(EPOCH):
    epoch_list += [epoch]
    

    prediction = g3_net(X_scaled[:1000])          # input x and predict based on x

    loss       = loss_func(prediction, g3_scaled[:1000]) # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients


            # plot and show learning process
    plt.cla()
    loss_list += [loss.data.numpy()]
    plt.plot(epoch_list, loss_list, label = 'Latest Loss = %.4f' % loss.data.numpy(), color = 'r')
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(fontsize = 20, frameon= 0)
            
    display.clear_output(wait = True) #these two lines plots the data in real time
    display.display(fig)
    print('Current Epoch =', epoch)
    print('Latest Loss =', loss.data.numpy())


# Generally, g3_net has good predictions

# In[186]:


print('g3_net predictions of g3:',  (g3_net(X_scaled[0:20]).detach().numpy()*y_std_C[2])+y_mean_C[2]   )
print('Actual labels:',  g3_constraint[0:20])


# ## Plotting state space and visualize violation of constraints

# ### Random **unconstrained** policy

# In[208]:


# Generate data using random policy
plotting_state_data = []
for i in range(1000):
    initialize_MDP_BioEnv() # Initialize system
    episode = generate_random_episode(initial_state = generate_initial_state()) # Generate 1 episode
    data = extract_data_from_episode(episode, discount_factor = 0.9)   # Extract datapoint from episode
    for j in data:
        j = j[:-1]
        plotting_state_data += j  # Add datapoints to training set
# plotting_state_data
plotting_state_data = pd.DataFrame(plotting_state_data)
plotting_state_data_g1_violated = plotting_state_data[plotting_state_data[1] - 800 > 0] # Conpath 1
plotting_state_data_g2_violated = plotting_state_data[plotting_state_data[2] - 0.011*plotting_state_data[0] > 0] # Conpath 2
plotting_state_data_g3_violated = plotting_state_data[(plotting_state_data[3] == 240.0) & (plotting_state_data[1] - 150 > 0)] # Terminal constraint

print('No. of states visited:',len(plotting_state_data))
print('No. of states where g1 violated:',len(plotting_state_data_g1_violated))
print('No. of states where g2 violated:',len(plotting_state_data_g2_violated))
print('No. of states where g3 violated:',len(plotting_state_data_g3_violated))

Cx_all = plotting_state_data[0]
Cn_all = plotting_state_data[1]
Cq_all = plotting_state_data[2]

Cx_g1 = plotting_state_data_g1_violated[0]
Cn_g1 = plotting_state_data_g1_violated[1]
Cq_g1 = plotting_state_data_g1_violated[2]

Cx_g2 = plotting_state_data_g2_violated[0]
Cn_g2 = plotting_state_data_g2_violated[1]
Cq_g2 = plotting_state_data_g2_violated[2]

Cx_g3 = plotting_state_data_g3_violated[0]
Cn_g3 = plotting_state_data_g3_violated[1]
Cq_g3 = plotting_state_data_g3_violated[2]

@interact(ns=(-90,90,0.1),ew=(-90,90,0.1))
def interact_poly(ns,ew):
    fig = plt.figure(figsize = (10,7))
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.plot3D(Cx_all, Cn_all, Cq_all, 'g', label = 'no violation', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g1, Cn_g1, Cq_g1, 'r', label = 'g1', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g2, Cn_g2, Cq_g2, 'b', label = 'g2', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g3, Cn_g3, Cq_g3, 'black', label = 'g3', linestyle = '', marker = 'o', markersize = 1)
    plt.xlabel('$C_x$', labelpad=30)
    plt.ylabel('$C_n$', labelpad=50)
    ax.set_zlabel('$C_q$', fontsize=20, labelpad=20)
    plt.legend()
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlim(0,16)
    plt.ylim(0,2200)
    ax.set_zlim(0, 0.25)
    ax.view_init(ns,ew)
    plt.show()


# ### Trained **unconstrained** policy

# In[2746]:


# Generate data using random policy
plotting_state_data = []
for i in range(1000):
    episode = generate_episode_with_NN(GA_optimize, [1.0, 150.0, 0, 0], epsilon = 0) # Generate 1 episode
    data = extract_data_from_episode(episode, discount_factor = 0.9)   # Extract datapoint from episode
    for j in data:
        j = j[:-1]
        plotting_state_data += j  # Add datapoints to training set
# plotting_state_data
plotting_state_data = pd.DataFrame(plotting_state_data)
plotting_state_data_g1_violated = plotting_state_data[plotting_state_data[1] - 800 > 0] # Conpath 1
plotting_state_data_g2_violated = plotting_state_data[plotting_state_data[2] - 0.011*plotting_state_data[0] > 0] # Conpath 2
plotting_state_data_g3_violated = plotting_state_data[(plotting_state_data[3] == 240.0) & (plotting_state_data[1] - 150 > 0)] # Terminal constraint

print('No. of states visited:',len(plotting_state_data))
print('No. of states where g1 violated:',len(plotting_state_data_g1_violated))
print('No. of states where g2 violated:',len(plotting_state_data_g2_violated))
print('No. of states where g3 violated:',len(plotting_state_data_g3_violated))

Cx_all = plotting_state_data[0]
Cn_all = plotting_state_data[1]
Cq_all = plotting_state_data[2]

Cx_g1 = plotting_state_data_g1_violated[0]
Cn_g1 = plotting_state_data_g1_violated[1]
Cq_g1 = plotting_state_data_g1_violated[2]

Cx_g2 = plotting_state_data_g2_violated[0]
Cn_g2 = plotting_state_data_g2_violated[1]
Cq_g2 = plotting_state_data_g2_violated[2]

Cx_g3 = plotting_state_data_g3_violated[0]
Cn_g3 = plotting_state_data_g3_violated[1]
Cq_g3 = plotting_state_data_g3_violated[2]

@interact(ns=(-90,90,0.1),ew=(-90,90,0.1))
def interact_poly(ns,ew):
    fig = plt.figure(figsize = (10,7))
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.plot3D(Cx_all, Cn_all, Cq_all, 'g', label = 'no violation', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g1, Cn_g1, Cq_g1, 'r', label = 'g1', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g2, Cn_g2, Cq_g2, 'b', label = 'g2', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g3, Cn_g3, Cq_g3, 'black', label = 'g3', linestyle = '', marker = 'o', markersize = 1)
    plt.xlabel('$C_x$', labelpad=20)
    plt.ylabel('$C_n$', labelpad=20)
    ax.set_zlabel('$C_q$', fontsize=20, labelpad=20)
    plt.legend()
    # plt.zlabel('Cq')
    ax.view_init(ns,ew)
    plt.show()


# ### Method 2: Training Q-network and C-network simultaneously

# In[31]:


# Initialize Q-network
Q_net = torch.nn.Sequential(
        torch.nn.Linear(6,   200, bias = True), # 6 input nodes
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),    # apply ReLU activation function
        torch.nn.Linear(200,   1, bias = True), # 1 output node
    ).double()

# Initialize g1_network
g1_net = torch.nn.Sequential(
        torch.nn.Linear(6, 200, bias = True), # 4 input nodes (Cx, Cn, Cq, t_f - t, u_L, u_Fn)
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 100, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 50, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 1, bias = True), # 1 output node
    ).double()

# Initialize g2_network
g2_net = torch.nn.Sequential(
        torch.nn.Linear(6, 200, bias = True), # 4 input nodes (Cx, Cn, Cq, t_f - t, u_L, u_Fn)
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 100, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 50, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 1, bias = True), # 1 output node
    ).double()

g3_net = torch.nn.Sequential(
        torch.nn.Linear(6, 200, bias = True), # 6 input nodes (Cx, Cn, Cq, t_f - t, u_L, u_Fn)
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 200, bias = True), 
        torch.nn.LeakyReLU(),                
        torch.nn.Linear(200, 100, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 50, bias = True), 
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 1, bias = True), # 1 output node
    ).double()


# In[34]:


# # ========== 1) GENERATING INITIAL BATCH OF SAMPLES FOR REPLAY BUFFER ==========
# epsilon       = 0.99   # Initial epsilon
# g1_threshold  = 500    # Initial g1_threshold
# g2_threshold  = 0.050  # Initial g2_threshold
# g3_threshold  = 150    # Initial g3_threshold
# replay_buffer = collections.deque(maxlen = 3000)  # Max capacity of 3k
# g1_buffer     = collections.deque(maxlen = 30000) # Max capacity of 30k
# g2_buffer     = collections.deque(maxlen = 30000) # Max capacity of 30k
# g3_buffer     = collections.deque(maxlen = 30000) # Max capacity of 30k

# for i in range(100): # Generate 100 initial episodes
#     initialize_MDP_BioEnv() # Initialize system
#     episode       = generate_episode_with_NN(NN = GA_optimize_constrained,
#                                              initial_state = generate_initial_state(),
#                                              epsilon = epsilon, 
#                                              g1_threshold = g1_threshold,
#                                              g2_threshold = g2_threshold,
#                                              g3_threshold = g3_threshold)
#     episode_copy  = copy.deepcopy(episode) # Copied to ensure same episodes are fed into Q and C
#     episode_copy2 = copy.deepcopy(episode) # Copied to ensure same episodes are fed into Q and C
#     episode_copy3 = copy.deepcopy(episode) # Copied to ensure same episodes are fed into Q and C
#     replay_buffer = update_replay_buffer(replay_buffer, episode, discount_factor = 0.9) # Add to replay buffer
#     g1_buffer     = update_g1_buffer(g1_buffer, episode_copy)  # Add to g1 buffer
#     g2_buffer     = update_g2_buffer(g2_buffer, episode_copy2) # Add to g2 buffer
#     g3_buffer     = update_g3_buffer(g3_buffer, episode_copy3) # Add to g2 buffer
# print('Initial size of replay buffer:', len(replay_buffer))
# print('Initial size of g1 buffer:', len(g1_buffer))
# print('Initial size of g2 buffer:', len(g2_buffer))
# print('Initial size of g3 buffer:', len(g3_buffer))

# # ================================= 2) TRAINING THE Q-NETWORK =================================
# g1_threshold_list  = [] # Store g1_threshold
# g2_threshold_list  = [] # Store g2_threshold
# g3_threshold_list  = [] # Store g3_threshold
# epsilon_list       = [] # Store epsilon
# score_list         = [] # For plotting score of policy visited over time
# iteration_list     = [] # Store iteration
# policy_mean_list   = [] # For plotting evolution of policy visited over time
# policy_std_list    = [] # For plotting evolution of policy visited over time
# training_loss_list = [] # For plotting loss of NN (after 20 epochs) over time
# episode_bank       = [] # A list of list of lists - Tertiary list  -> Iterations
#                         #                         - Secondary list -> Episodes
#                         #                         - Primary list   -> Steps
#                         #                         For plotting evolution of states visited over time            


for iteration in range(2000):
    
    # SCORE EXISTING POLICY
    print('=========== TRAINING ITERATION %.0f ===========' % iteration)
    print('Current epsilon = ', epsilon)
    print('Current g1_threshold = ', g1_threshold)
    print('Current g2_threshold = ', g2_threshold)
    print('Current g3_threshold = ', g3_threshold)
    MDP_BioEnv = Model_env(p, tf) # Initialize with fixed constants for scoring
    try:
        current_score = score_NN_policy(GA_optimize_constrained, 
                                        initial_state = [1.0, 150.0, 0, 0], # Use fixed initial state
                                        num_iterations = 1, 
                                        get_control = False, 
                                        g1_threshold = g1_threshold,
                                        g2_threshold = g2_threshold,
                                        g3_threshold = g3_threshold)
    except KeyboardInterrupt: 
        try:
            current_score = score_NN_policy(GA_optimize_constrained, 
                                        initial_state = [1.0, 150.0, 0, 0], 
                                        num_iterations = 1, 
                                        get_control = False, 
                                        g1_threshold = g1_threshold,
                                        g2_threshold = g2_threshold,
                                        g3_threshold = g3_threshold)
        except KeyboardInterrupt:
            anneal_NN()
            
    print('AVERAGE SCORE = ', current_score)
    print('')
    
    # GENERATE EPISODES & ADD TO REPLAY BUFFER
    for j in range(100):
        if j in range(0,1000,10): # Reinitialize system every 10 episodes
            initialize_MDP_BioEnv() # Initialize system
            print('MDP_BioEnv reinitialized')
        try: # generate_episode_with_NN has timeout of 5 seconds
            episode      = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = epsilon, 
                                                 g1_threshold = g1_threshold,
                                                 g2_threshold = g2_threshold,
                                                 g3_threshold = g3_threshold)
        except KeyboardInterrupt: # 2nd chance if first try fails
            try: # generate_episode_with_NN has timeout of 5 seconds
                episode      = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = epsilon, 
                                                 g1_threshold = g1_threshold,
                                                 g2_threshold = g2_threshold,
                                                 g3_threshold = g3_threshold)
            except KeyboardInterrupt: # 2nd chance if first try fails
                try: # generate_episode_with_NN has timeout of 5 seconds
                    episode      = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = epsilon, 
                                                 g1_threshold = g1_threshold,
                                                 g2_threshold = g2_threshold,
                                                 g3_threshold = g3_threshold)
                except KeyboardInterrupt:
                    anneal_NN()
                    
        if episode != None:                
            episode_copy  = copy.deepcopy(episode) # Copied to ensure same episodes are fed into Q and C
            episode_copy2 = copy.deepcopy(episode)
            episode_copy3 = copy.deepcopy(episode)
            replay_buffer = update_replay_buffer(replay_buffer, episode, discount_factor = 0.9) # Add to replay buffer
            g1_buffer     = update_g1_buffer(g1_buffer, episode_copy) # Add to g1 buffer
            g2_buffer     = update_g2_buffer(g2_buffer, episode_copy2) # Add to g2 buffer
            g3_buffer     = update_g3_buffer(g3_buffer, episode_copy3) # Add to g2 buffer
            if j in np.arange(0, 100, 10):
                print('%.0f th episode generated' % j)
            
    # === TRAIN Q_NETWORK NETWORK ===
    combined_inputs_Q  = [] # List of lists to be converted into input tensor
    combined_targets_Q = [] # List to be converted into target tensor
    samples_Q          = random.sample(replay_buffer, 100) # Draw random samples from replay buffer
    
    for inputs, target in samples_Q:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_Q  += [inputs]
            combined_targets_Q += [[target]]
    combined_inputs_Q  = torch.tensor(combined_inputs_Q).double()  # Convert list to tensor
    combined_targets_Q = torch.tensor(combined_targets_Q).double() # Convert list to tensor

    Q_optimizer = torch.optim.Adam(Q_net.parameters(), lr=1e-3) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    
    for epoch in range(20):
        prediction = Q_net(combined_inputs_Q)                    # Input x and predict based on x
        Q_loss     = loss_func(prediction, combined_targets_Q) # Must be (1. nn output, 2. target)

        Q_optimizer.zero_grad()   # Clear gradients for next train
        Q_loss.backward()         # Backpropagation, compute gradients
        Q_optimizer.step()        # Apply gradients
#         print('Epoch = ', epoch, 'Loss = %.4f' % Q_loss.data.numpy())
    print('Q-network trained')
    # === TRAIN g1_NETWORK ===
    combined_inputs_g1  = [] # List of lists to be converted into input tensor
    combined_targets_g1 = [] # List to be converted into target tensor
    samples_g1          = random.sample(g1_buffer, 500) # Draw random samples from replay buffer
    
    for inputs, target in samples_g1:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_g1  += [inputs]
            combined_targets_g1 += [[target]]

    combined_inputs_g1  = torch.tensor(combined_inputs_g1).double()  # Convert list to tensor
    combined_targets_g1 = torch.tensor(combined_targets_g1).double() # Convert list to tensor

    g1_optimizer = torch.optim.Adam(g1_net.parameters(), lr=1e-4) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    
    for epoch in range(10):
        prediction = g1_net(combined_inputs_g1)                    # Input x and predict based on x
        g1_loss    = loss_func(prediction, combined_targets_g1) # Must be (1. nn output, 2. target)

        g1_optimizer.zero_grad()   # Clear gradients for next train
        g1_loss.backward()         # Backpropagation, compute gradients
        g1_optimizer.step()        # Apply gradients
#         print('Epoch = ', epoch, 'Loss = %.4f' % g1_loss.data.numpy())
    print('g1-network trained')
    # === TRAIN g2_NETWORK ===
    combined_inputs_g2  = [] # List of lists to be converted into input tensor
    combined_targets_g2 = [] # List to be converted into target tensor
    samples_g2          = random.sample(g2_buffer, 1000) # Draw random samples from replay buffer
    
    for inputs, target in samples_g2:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_g2  += [inputs]
            combined_targets_g2 += [[target]]

    combined_inputs_g2  = torch.tensor(combined_inputs_g2).double()  # Convert list to tensor
    combined_targets_g2 = torch.tensor(combined_targets_g2).double() # Convert list to tensor

    g2_optimizer = torch.optim.Adam(g2_net.parameters(), lr=1e-3) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    
    for epoch in range(100):
        prediction = g2_net(combined_inputs_g2)                    # Input x and predict based on x
        g2_loss    = loss_func(prediction, combined_targets_g2) # Must be (1. nn output, 2. target)

        g2_optimizer.zero_grad()   # Clear gradients for next train
        g2_loss.backward()         # Backpropagation, compute gradients
        g2_optimizer.step()        # Apply gradients
#         print('Epoch = ', epoch, 'Loss = %.4f' % g2_loss.data.numpy())
    print('g2-network trained')
    # === TRAIN g3_NETWORK ===
    combined_inputs_g3  = [] # List of lists to be converted into input tensor
    combined_targets_g3 = [] # List to be converted into target tensor
    samples_g3          = random.sample(g3_buffer, 1000) # Draw random samples from replay buffer
    
    for inputs, target in samples_g3:
        if len(inputs) == 6: # To fix the changing len problem
            combined_inputs_g3  += [inputs]
            combined_targets_g3 += [[target]]

    combined_inputs_g3  = torch.tensor(combined_inputs_g3).double()  # Convert list to tensor
    combined_targets_g3 = torch.tensor(combined_targets_g3).double() # Convert list to tensor

    g3_optimizer = torch.optim.Adam(g3_net.parameters(), lr=1e-3) # Initialize Adam optimizer
    loss_func = torch.nn.SmoothL1Loss()                     # Define loss function
    
    for epoch in range(200):
        prediction = g3_net(combined_inputs_g3)                    # Input x and predict based on x
        g3_loss    = loss_func(prediction, combined_targets_g3) # Must be (1. nn output, 2. target)

        g3_optimizer.zero_grad()   # Clear gradients for next train
        g3_loss.backward()         # Backpropagation, compute gradients
        g3_optimizer.step()        # Apply gradients
#         print('Epoch = ', epoch, 'Loss = %.4f' % g2_loss.data.numpy())
    print('g3-network trained')
    # Save information
    g1_threshold_list  += [g1_threshold]
    g2_threshold_list  += [g2_threshold]
    g3_threshold_list  += [g3_threshold]
    epsilon_list       += [epsilon]
    score_list         += [current_score] # Store score for each iteration
    training_loss_list += [Q_loss.data.numpy()]                                # Store training loss after 20 epochs for each iteration
    iteration_list     += [iteration]                                        # Store iteration
    

    # INVESTIGATING EVOLUTION OF STATES VISITED OVER TRAINING ITERATIONS
    if iteration in np.arange(0, 5000, 10): # Every 10 iterations
        pool_of_episodes = [] # A *list of lists* of episodes
        for i in range(10):   # Generate 10 sample episodes and store them
            initialize_MDP_BioEnv()
            try:
                pool_of_episodes += [generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = g1_threshold,
                                                 g2_threshold = g2_threshold,
                                                 g3_threshold = g3_threshold)]
            except KeyboardInterrupt:
                try:
                    pool_of_episodes += [generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = g1_threshold,
                                                 g2_threshold = g2_threshold,
                                                 g3_threshold = g3_threshold)]
                except KeyboardInterrupt:
                    try:
                        pool_of_episodes += [generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = g1_threshold,
                                                 g2_threshold = g2_threshold,
                                                 g3_threshold = g3_threshold)]
                    except KeyboardInterrupt:
                        anneal_NN()
        episode_bank         += [pool_of_episodes]
        print('Episode pool added to episode bank')
    print('')
    epsilon      *= 0.99  # Decay epsilon
    g1_threshold *= 0.995 # Decay threshold
    g2_threshold *= 0.995 # Decay threshold
    g3_threshold *= 0.995 # Decay threshold
    if epsilon < 1.8637566e-9: # after 2000 iterations
        break


# ## Objective score policy

# In[362]:


a = -5.97850052e+00
b = -4.47116078e-04
c = 0
# a = 0
# b = 0
# c = 0


# In[397]:


# Score policy using 1000 MC episodes
scores = []
for i in range(1000):
    initialize_MDP_BioEnv()
    try:
        score = score_NN_policy(GA_optimize_constrained, 
                                            initial_state = generate_initial_state(), 
                                            num_iterations = 1, 
                                            get_control = False, 
                                            g1_threshold = a,
                                            g2_threshold = b, # Backoff needed for g2
                                            g3_threshold = c)
    except KeyboardInterrupt:
        score = score_NN_policy(GA_optimize_constrained, 
                                            initial_state = generate_initial_state(), 
                                            num_iterations = 1, 
                                            get_control = False, 
                                            g1_threshold = a,
                                            g2_threshold = b, # Backoff needed for g2
                                            g3_threshold = c)
    scores += [score] # append score


# In[408]:


# Save data
with open('./Data/CS1_score_dist_g1g2g3_noise_2000_ep', 'wb') as f:
    pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL)

# Load pre-run data
# with open('./Data/CS1_score_dist_g1g2g3_noise_2000_ep', 'rb') as f:
#     scores = pickle.load(f)


# In[376]:


# Plot distribution of objectives
fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(scores, norm_hist = True)
ax.set(xlabel= 'Objective score', ylabel = 'Fraction of samples')
# plt.plot([0,0],[0,0.0023], linestyle = '--', color = 'r')
plt.legend()
plt.show()


# ## Plot 1 episode

# In[68]:


# Run 1 episode
episode = generate_episode_with_NN(GA_optimize_constrained, [1.0, 150.0, 0, 0], epsilon = 0, 
                                   g1_threshold = a,
                                   g2_threshold = b,
                                   g3_threshold = c)
plot_episode(episode)


# ## Save neural networks

# In[14]:


# Save neural network
# torch.save(Q_net, './NN_models/CS1_Q_net_g1g2g3_noise_2000_ep')
# # Save neural network
# torch.save(g1_net, './NN_models/CS1_g1_net_g1g2g3_noise_2000_ep')
# # Save neural network
# torch.save(g2_net, './NN_models/CS1_g2_net_g1g2g3_noise_2000_ep')
# # Save neural network
# torch.save(g3_net, './NN_models/CS1_g3_net_g1g2g3_noise_2000_ep')


# Load pre-trained neural network
# Q_net = torch.load('./NN_models/CS1_Q_net_g1g2g3_noise_2000_ep')
# Q_net.eval()
# # Load pre-trained neural network
# g1_net = torch.load('./NN_models/CS1_g1_net_g1g2g3_noise_2000_ep')
# g1_net.eval()
# # Load pre-trained neural network
# g2_net = torch.load('./NN_models/CS1_g2_net_g1g2g3_noise_2000_ep')
# g2_net.eval()
# # Load pre-trained neural network
# g3_net = torch.load('./NN_models/CS1_g3_net_g1g2g3_noise_2000_ep')
# g3_net.eval()


# ## Plot objective score and Q-network loss

# In[ ]:


iteration_list = range(2000)
# Save data
with open('./Data/CS1_score_g1g2g3_noise_2000_ep', 'wb') as f:
    pickle.dump(score_list, f, pickle.HIGHEST_PROTOCOL)
with open('./Data/CS1_loss_g1g2g3_noise_2000_ep', 'wb') as f:
    pickle.dump(training_loss_list, f, pickle.HIGHEST_PROTOCOL)

# Load pre-run data
# with open('./Data/CS1_score_g1g2g3_noise_2000_ep', 'rb') as f:
#     score_list = pickle.load(f)
# with open('./Data/CS1_loss_g1g2g3_noise_2000_ep', 'rb') as f:
#     training_loss_list = pickle.load(f)

# Plot score vs. training iteration
fig, ax = plt.subplots(figsize=(8,6))
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(which='major', length=10, width=1, direction='out')
ax.tick_params(which='minor', length=4, width=1, direction='out')
plt.plot(iteration_list[:len(score_list)], score_list, color = 'b')
plt.legend(frameon =  0)
plt.ylabel('Average score of policy', fontsize = 20)
plt.xlabel('Training iteration', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()

# Plot training loss vs. training iteration
fig, ax = plt.subplots(figsize=(8,6))
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.tick_params(which='major', length=10, width=1, direction='out')
ax.tick_params(which='minor', length=4, width=1, direction='out')
plt.plot(iteration_list[:len(score_list)], training_loss_list, color = 'r')
plt.legend(frameon =  0)
plt.ylabel('Training loss (after 20 epochs)', fontsize = 20)
plt.xlabel('Training iteration', fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()


# ## Plot MC episodes and explore state space

# In[1017]:


# # Generate g1_threshold
# g1_threshold_list = []
# g1_threshold = 500
# g2_threshold_list = []
# g2_threshold = 0.05
# for i in range(2000):
#     g1_threshold_list += [g1_threshold]
#     g1_threshold *= 0.995
#     g2_threshold_list += [g2_threshold]
#     g2_threshold *= 0.995


# In[ ]:


# Save data to csv
plotting_state_data.to_csv(r'./Data/CS1_state_space_g1g2g3_noise_2000_ep.csv', index = False)
# # Load data as pd.DataFrame
# plotting_state_data = pd.read_csv('./Data/CS1_state_space_g1g2g3_noise_2000_ep.csv')


# In[25]:


# Generate data using random policy
plotting_state_data = []
for i in range(1000):
    initialize_MDP_BioEnv() # Initialize system
    try:
        episode      = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = a,
                                                 g2_threshold = b, # Backoff needed for g2
                                                 g3_threshold = c)
    except KeyboardInterrupt:
        episode      = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = a,
                                                 g2_threshold = b,
                                                 g3_threshold = c)
    data = extract_data_from_episode(episode, discount_factor = 0.9)   # Extract datapoint from episode
    for j in data:
        j = j[:-1]
        plotting_state_data += j  # Add datapoints to training set

# plotting_state_data
plotting_state_data = pd.DataFrame(plotting_state_data)

try:
    plotting_state_data_g1_violated = plotting_state_data[plotting_state_data[1] - 800 > 0] # Conpath 1
    plotting_state_data_g2_violated = plotting_state_data[plotting_state_data[2] - 0.011*plotting_state_data[0] > 0] # Conpath 2
    plotting_state_data_g3_violated = plotting_state_data[(plotting_state_data[3] == 240.0) & (plotting_state_data[1] - 150 > 0)] # Terminal constraint
except:
    plotting_state_data_g1_violated = plotting_state_data[plotting_state_data['1'] - 800 > 0] # Conpath 1
    plotting_state_data_g2_violated = plotting_state_data[plotting_state_data['2'] - 0.011*plotting_state_data['0'] > 0] # Conpath 2
    plotting_state_data_g3_violated = plotting_state_data[(plotting_state_data['3'] == 240.0) & (plotting_state_data['1'] - 150 > 0)] # Terminal constraint
    
print('No. of states visited:',len(plotting_state_data))
print('No. of states where g1 violated:',len(plotting_state_data_g1_violated))
print('No. of states where g2 violated:',len(plotting_state_data_g2_violated))
print('No. of states where g3 violated:',len(plotting_state_data_g3_violated))

try: 
    Cx_all = plotting_state_data[0]
    Cn_all = plotting_state_data[1]
    Cq_all = plotting_state_data[2]
    Cx_g1 = plotting_state_data_g1_violated[0]
    Cn_g1 = plotting_state_data_g1_violated[1]
    Cq_g1 = plotting_state_data_g1_violated[2]
    Cx_g2 = plotting_state_data_g2_violated[0]
    Cn_g2 = plotting_state_data_g2_violated[1]
    Cq_g2 = plotting_state_data_g2_violated[2]
    Cx_g3 = plotting_state_data_g3_violated[0]
    Cn_g3 = plotting_state_data_g3_violated[1]
    Cq_g3 = plotting_state_data_g3_violated[2]
except:
    Cx_all = plotting_state_data['0']
    Cn_all = plotting_state_data['1']
    Cq_all = plotting_state_data['2']
    Cx_g1 = plotting_state_data_g1_violated['0']
    Cn_g1 = plotting_state_data_g1_violated['1']
    Cq_g1 = plotting_state_data_g1_violated['2']
    Cx_g2 = plotting_state_data_g2_violated['0']
    Cn_g2 = plotting_state_data_g2_violated['1']
    Cq_g2 = plotting_state_data_g2_violated['2']
    Cx_g3 = plotting_state_data_g3_violated['0']
    Cn_g3 = plotting_state_data_g3_violated['1']
    Cq_g3 = plotting_state_data_g3_violated['2']

@interact(ns=(-90,90,0.1),ew=(-90,90,0.1))
def interact_poly(ns,ew):
    fig = plt.figure(figsize = (10,7))
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.plot3D(Cx_all, Cn_all, Cq_all, 'g', label = 'no violation', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g1, Cn_g1, Cq_g1, 'r', label = 'g1', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g2, Cn_g2, Cq_g2, 'b', label = 'g2', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g3, Cn_g3, Cq_g3, 'black', label = 'g3', linestyle = '', marker = 'o', markersize = 1)
    plt.xlabel('$C_x$', labelpad=30)
    plt.ylabel('$C_n$', labelpad=50)
    ax.set_zlabel('$C_q$', fontsize=20, labelpad=20)
    plt.legend()
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlim(0,16)
    plt.ylim(0,2200)
    ax.set_zlim(0, 0.25)
    ax.view_init(ns,ew)
    plt.show()


# ## Plot evolution of state space and save to gif

# In[55]:


init_g1_threshold = 500
init_g2_threshold = 0.05
g1_threshold_list = [500]
g2_threshold_list = [0.05]
for i in range(2000):
    init_g1_threshold *= 0.995
    init_g2_threshold *= 0.995
    g1_threshold_list += [init_g1_threshold]
    g2_threshold_list += [init_g2_threshold]
g1_threshold_list


# In[251]:


# SAVING EPISODE BANK - JSON WORKS BUT NOT PICKLE BCOS NESTED LIST
# # open output file for writing
# with open('./Data/CS1_episode_bank_g1g2g3_noise_1000_ep', 'w') as f:
#     json.dump(episode_bank, f)
# # open output file for reading
# with open('./Data/CS1_episode_bank_g1g2g3_noise_1000_ep', 'r') as f:
#     episode_bank = json.load(f)


# In[56]:


# Plot episode bank 
my_images = []  # For saving into gif

for i in range(len(episode_bank)): # i is the index for iteration
    plotting_state_data = []
    
    for j in range(len(episode_bank[i])): # j is the index for different episodes
        episode = episode_bank[i][j]

        data = extract_data_from_episode(episode, discount_factor = 0.9)   # Extract datapoint from episode
        for z in data:
            z = z[:-1]
            plotting_state_data += z  # Add datapoints to training set
    iteration = iteration_list[::10][i] # Record training iteration
    g1_threshold = g1_threshold_list[::10][i] # Record g1_threshold
    g2_threshold = g2_threshold_list[::10][i] # Record g1_threshold
    # plotting_state_data
    plotting_state_data = pd.DataFrame(plotting_state_data)
    plotting_state_data_g1_violated = plotting_state_data[plotting_state_data[1] - 800 > 0] # Conpath 1
    plotting_state_data_g2_violated = plotting_state_data[plotting_state_data[2] - 0.011*plotting_state_data[0] > 0] # Conpath 2
    plotting_state_data_g3_violated = plotting_state_data[(plotting_state_data[3] == 240.0) & (plotting_state_data[1] - 150 > 0)] # Terminal constraint

    print('No. of states visited:',len(plotting_state_data))
    print('No. of states where g1 violated:',len(plotting_state_data_g1_violated))
    print('No. of states where g2 violated:',len(plotting_state_data_g2_violated))
    print('No. of states where g3 violated:',len(plotting_state_data_g3_violated))

    Cx_all = plotting_state_data[0]
    Cn_all = plotting_state_data[1]
    Cq_all = plotting_state_data[2]

    Cx_g1 = plotting_state_data_g1_violated[0]
    Cn_g1 = plotting_state_data_g1_violated[1]
    Cq_g1 = plotting_state_data_g1_violated[2]

    Cx_g2 = plotting_state_data_g2_violated[0]
    Cn_g2 = plotting_state_data_g2_violated[1]
    Cq_g2 = plotting_state_data_g2_violated[2]

    Cx_g3 = plotting_state_data_g3_violated[0]
    Cn_g3 = plotting_state_data_g3_violated[1]
    Cq_g3 = plotting_state_data_g3_violated[2]

    
    fig = plt.figure(figsize = (10,7))
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.plot3D(Cx_all, Cn_all, Cq_all, 'g', label = 'no violation', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g1, Cn_g1, Cq_g1, 'r', label = 'g1', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g2, Cn_g2, Cq_g2, 'b', label = 'g2', linestyle = '', marker = 'o', markersize = 1)
    ax.plot3D(Cx_g3, Cn_g3, Cq_g3, 'black', label = 'g3', linestyle = '', marker = 'o', markersize = 1)
    plt.xlabel('$C_x$', labelpad=30)
    plt.ylabel('$C_n$', labelpad=50)
    ax.set_zlabel('$C_q$', fontsize=20, labelpad=20)
    plt.legend()
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.xlim(0,16)
    plt.ylim(0,2200)
    ax.set_zlim(0, 0.25)
    plt.legend()
    ax.text(0,1000,0.11,'Training iteration {} \ng1_Threshold = {} \ng2_Threshold = {}'.format(iteration, round(g1_threshold,0), round(g2_threshold,4)),None)
    ax.view_init(0,0)
    plt.show()
        
    # Used to return the plot as an image array 
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    my_images.append(image)

# save images as a gif    
# imageio.mimsave('./Data/CS1_training_g1g2g3_1100_ep.gif', my_images, fps=6)


# ## Plot trajectory with MC episodes

# In[301]:


# SAVING EPISODE POOL - JSON WORKS BUT NOT PICKLE BCOS NESTED LIST
import json
# open output file for writing
# with open('./Data/CS1_episode_pool_g1g2g3_noise_2000_ep', 'w') as f:
#     json.dump(episode_pool, f)
# open output file for reading
# with open('./Data/CS1_episode_pool_g1g2g3_noise_2000_ep', 'r') as f:
#     episode_pool = json.load(f)


# In[92]:


# a = -5.97850052e+00
# b = -4.47116078e-04
# c = 0
a = 0
b = 0
c = 0


# In[93]:


# Plot average of 1000 episodes
episode_pool = []
for i in range(400):
    initialize_MDP_BioEnv() # Initialize system
    try:
        episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = a,
                                                 g2_threshold = b,
                                                 g3_threshold = c)
    except KeyboardInterrupt:
        episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = a,
                                                 g2_threshold = b,
                                                 g3_threshold = c) # Generate 1 episode
    episode_pool += [episode]


# In[87]:


plot_episode_pool(episode_pool)


# ## Plot distribution of g values

# In[26]:


# Save data
# with open('./Data/CS1_max_g1_g1g2g3_noise_2000_ep', 'wb') as f:
#     pickle.dump(max_g1_list, f, pickle.HIGHEST_PROTOCOL)
# with open('./Data/CS1_max_g2_g1g2g3_noise_2000_ep', 'wb') as f:
#     pickle.dump(max_g2_list, f, pickle.HIGHEST_PROTOCOL)
# with open('./Data/CS1_max_g3_g1g2g3_noise_2000_ep', 'wb') as f:
#     pickle.dump(max_g3_list, f, pickle.HIGHEST_PROTOCOL)
    
# Load pre-run data
# with open('./Data/CS1_max_g1_g1g2g3_noise_2000_ep', 'rb') as f:
#     max_g1_list = pickle.load(f)
# with open('./Data/CS1_max_g2_g1g2g3_noise_2000_ep', 'rb') as f:
#     max_g2_list = pickle.load(f)
# with open('./Data/CS1_max_g3_g1g2g3_noise_2000_ep', 'rb') as f:
#     max_g3_list = pickle.load(f)


# In[68]:


# Backoffs obtained using Broyden's method
# array([-5.97850052e+00, -4.47116078e-04])
a = -5.97850052e+00
b = -4.47116078e-04
c = 0


# In[22]:


max_g1_list = []
max_g2_list = []
max_g3_list = []
for i in range(400):
    initialize_MDP_BioEnv()
    try:
        episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = a,
                                                 g2_threshold = b,
                                                 g3_threshold = c) # Generate 1 episode
    except KeyboardInterrupt:
        episode = generate_episode_with_NN(NN = GA_optimize_constrained,
                                                 initial_state = generate_initial_state(),
                                                 epsilon = 0, 
                                                 g1_threshold = a,
                                                 g2_threshold = b,
                                                 g3_threshold = c) # Generate 1 episode
    # WORST violation of this episode
    max_g1 = max(extract_constraint_values_from_episode(episode, N_limit = 800, terminal_N_limit = 150)[1])[0]
    max_g2 = max(extract_constraint_values_from_episode(episode, N_limit = 800, terminal_N_limit = 150)[2])[0]
    max_g3 = max(extract_constraint_values_from_episode(episode, N_limit = 800, terminal_N_limit = 150)[3])[0]
    max_g1_list += [max_g1]
    max_g2_list += [max_g2]
    max_g3_list += [max_g3]


# In[23]:


# Plot distribution of g1
fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(max_g1_list, norm_hist = True, color = 'green')
ax.set(xlabel= '$g_1$', ylabel = 'Proportion of samples')
plt.plot([0,0],[0,0.13], linestyle = '--', color = 'r')
# plt.xlim(-1000,1000)
plt.show()

# Plot distribution of g2
fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(max_g2_list, norm_hist = True, color = 'green')
ax.set(xlabel= '$g_2$', ylabel = 'Proportion of samples')
plt.plot([0,0],[0,2000], linestyle = '--', color = 'r')
plt.xticks(rotation=90)
# plt.xlim(-0.055,0.022)
plt.show()

# Plot distribution of g1
fig, ax = plt.subplots(figsize=(10,6))
ax = sns.distplot(max_g3_list, norm_hist = True, color = 'green')
ax.set(xlabel= '$g_3$', ylabel = 'Proportion of samples')
plt.plot([0,0],[0,0.55], linestyle = '--', color = 'r')
# plt.xlim(-160,330)
plt.show()


# ## Plot g1 and g2 trajectories

# In[366]:


# LOAD EPISODE POOLS WITH AND WITHOUT BACKOFFS
# open output file for reading
# with open('./Data/CS1_episode_pool_g1g2g3_noise_2000_ep', 'r') as f:
#     episode_pool_backoffs = json.load(f)
# with open('./Data/CS1_episode_pool_g1g2g3_noise_2000_ep_no_backoffs', 'r') as f:
#     episode_pool_no_backoffs = json.load(f)
# with open('./Data/CS1_episode_pool_MPC_10%_noise_noisy_init', 'r') as f:
#     episode_pool_MPC = json.load(f)


# In[368]:


# PROCESS EPISODE POOLS WITH AND WITHOUT BACKOFFS
g1_pool_backoffs    = [] # List of lists of g1 trajectories
g2_pool_backoffs    = []
g3_pool_backoffs    = []
g1_pool_no_backoffs = [] # List of lists of g1 trajectories
g2_pool_no_backoffs = []
g3_pool_no_backoffs = []
g1_pool_MPC         = [] # List of lists of g1 trajectories
g2_pool_MPC         = []
g3_pool_MPC         = []

for i in range(len(episode_pool_backoffs)):
    episode = episode_pool_backoffs[i]
    g1_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[1]
    g2_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[2]
    g3_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[3]
    g1_pool_backoffs += [list(g1_list)]
    g2_pool_backoffs += [list(g2_list)]
    g3_pool_backoffs += [list(g3_list)]

for i in range(len(episode_pool_no_backoffs)):
    episode = episode_pool_no_backoffs[i]
    g1_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[1]
    g2_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[2]
    g3_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[3]
    g1_pool_no_backoffs += [list(g1_list)]
    g2_pool_no_backoffs += [list(g2_list)]
    g3_pool_no_backoffs += [list(g3_list)]

for i in range(len(episode_pool_MPC)):
    episode = episode_pool_MPC[i]
    g1_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[1]
    g2_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[2]
    g3_list = extract_constraint_values_from_episode_NO_ORACLE(episode, N_limit = 800, terminal_N_limit = 150)[3]
    g1_pool_MPC += [list(g1_list)]
    g2_pool_MPC += [list(g2_list)]
    g3_pool_MPC += [list(g3_list)]
# g1_pool_backoffs = g1_pool_backoffs.tolist()
# g2_pool_backoffs = g2_pool_backoffs.tolist()
# g3_pool_backoffs = g3_pool_backoffs.tolist()
# g1_pool_no_backoffs = g1_pool_no_backoffs.tolist()
# g2_pool_no_backoffs = g2_pool_no_backoffs.tolist()
# g3_pool_no_backoffs = g3_pool_no_backoffs.tolist()


# In[380]:


# Plot g1 trajectory with/without backoffs
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(figsize = (10, 10))
g1_pool_backoffs       = np.array(g1_pool_backoffs)
g1_max_backoffs        = np.percentile(g1_pool_backoffs, 99, axis = 0) # Take 99p of trajectory
g1_min_backoffs        = np.percentile(g1_pool_backoffs, 1, axis = 0) # Take 1p of trajectory
g1_average_backoffs    = g1_pool_backoffs.mean(axis = 0) # Take average of trajectory
g1_pool_no_backoffs    = np.array(g1_pool_no_backoffs)
g1_max_no_backoffs        = np.percentile(g1_pool_no_backoffs, 99, axis = 0) # Take 99p of trajectory
g1_min_no_backoffs        = np.percentile(g1_pool_no_backoffs, 1, axis = 0) # Take 1p of trajectory
g1_average_no_backoffs = g1_pool_no_backoffs.mean(axis = 0) # Take average of trajectory
g1_max_backoffs = np.reshape(g1_max_backoffs, 13)
g1_min_backoffs = np.reshape(g1_min_backoffs, 13)
g1_max_no_backoffs = np.reshape(g1_max_no_backoffs, 13)
g1_min_no_backoffs = np.reshape(g1_min_no_backoffs, 13)
ax1.fill_between(t_list, g1_max_no_backoffs, g1_min_no_backoffs, facecolor='red', alpha=0.3)
ax1.fill_between(t_list, g1_max_backoffs, g1_min_backoffs, facecolor='green', alpha=0.4)
plt.plot(t_list, g1_average_backoffs, color = 'green', label = '$g_1$ with backoffs ($P_v = 0$)')
plt.plot(t_list, g1_average_no_backoffs, color = 'red', label = '$g_1$ without backoffs ($P_v = 0.77$)')
plt.plot([-100,2500], [0,0], color = 'black')
plt.xlabel('Time (h)', fontsize = 30)
plt.ylabel('$g_1$ (mgL$^{-1}$)', fontsize = 30)
plt.xlim(0,240)
plt.ylim(-780, 50)
plt.legend(frameon = 0, fontsize = 30, loc = 'lower left')
ax1.tick_params(axis='x', labelsize= 30)
ax1.tick_params(axis='y', labelsize= 30)

# Inset
axins = plt.axes([0.21,0.33,0.35,0.25]) # First two are location, last two are size of inset
axins.plot(t_list, g1_average_backoffs, color = 'green', label = '$g_1$ with backoffs')
axins.plot(t_list, g1_average_no_backoffs, color = 'red', label = '$g_1$ without backoffs')
axins.fill_between(t_list, g1_max_no_backoffs, g1_min_no_backoffs, facecolor='red', alpha=0.3)
axins.fill_between(t_list, g1_max_backoffs, g1_min_backoffs, facecolor='green', alpha=0.4)
axins.plot([-100,2500], [0,0], color = 'black')
x1, x2, y1, y2 = 10, 140, -100, 20 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
axins.tick_params(axis='x', labelsize= 20)
axins.tick_params(axis='y', labelsize= 20)

plt.show()


# In[381]:


# Plot g1 trajectory with backoffs vs MPC
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(figsize = (10, 10))
g1_pool_backoffs       = np.array(g1_pool_backoffs)
g1_max_backoffs        = np.percentile(g1_pool_backoffs, 99, axis = 0) # Take 99p of trajectory
g1_min_backoffs        = np.percentile(g1_pool_backoffs, 1, axis = 0) # Take 1p of trajectory
g1_average_backoffs    = g1_pool_backoffs.mean(axis = 0) # Take average of trajectory
g1_pool_MPC    = np.array(g1_pool_MPC)
g1_max_MPC        = np.percentile(g1_pool_MPC, 99, axis = 0) # Take 99p of trajectory
g1_min_MPC        = np.percentile(g1_pool_MPC, 1, axis = 0) # Take 1p of trajectory
g1_average_MPC = g1_pool_MPC.mean(axis = 0) # Take average of trajectory
g1_max_backoffs = np.reshape(g1_max_backoffs, 13)
g1_min_backoffs = np.reshape(g1_min_backoffs, 13)
g1_max_MPC = np.reshape(g1_max_MPC, 13)
g1_min_MPC = np.reshape(g1_min_MPC, 13)
ax1.fill_between(t_list, g1_max_MPC, g1_min_MPC, facecolor='blue', alpha=0.3)
ax1.fill_between(t_list, g1_max_backoffs, g1_min_backoffs, facecolor='green', alpha=0.4)
plt.plot(t_list, g1_average_backoffs, color = 'green', label = '$g_1$ with backoffs ($P_v = 0$)')
plt.plot(t_list, g1_average_MPC, color = 'blue', label = '$g_1$-MPC ($P_v = 0.12$)')
plt.plot([-100,2500], [0,0], color = 'black')
plt.xlabel('Time (h)', fontsize = 30)
plt.ylabel('$g_1$ (mgL$^{-1}$)', fontsize = 30)
plt.xlim(0,240)
plt.ylim(-780, 50)
plt.legend(frameon = 0, fontsize = 30)
ax1.tick_params(axis='x', labelsize= 30)
ax1.tick_params(axis='y', labelsize= 30)

# Inset
axins = plt.axes([0.21,0.33,0.35,0.25]) # First two are location, last two are size of inset
axins.plot(t_list, g1_average_backoffs, color = 'green', label = '$g_1$ with backoffs')
axins.plot(t_list, g1_average_MPC, color = 'blue', label = '$g_1$-MPC')
axins.fill_between(t_list, g1_max_MPC, g1_min_MPC, facecolor='blue', alpha=0.3)
axins.fill_between(t_list, g1_max_backoffs, g1_min_backoffs, facecolor='green', alpha=0.4)
axins.plot([-100,2500], [0,0], color = 'black')
x1, x2, y1, y2 = 10, 140, -100, 20 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
axins.tick_params(axis='x', labelsize= 20)
axins.tick_params(axis='y', labelsize= 20)

plt.show()


# In[415]:


# Plot g2 trajectory with/without backoffs
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(figsize = (10, 10))
g2_pool_backoffs       = np.array(g2_pool_backoffs)
g2_max_backoffs        = np.percentile(g2_pool_backoffs, 99, axis = 0) # Take 99p of trajectory
g2_min_backoffs        = np.percentile(g2_pool_backoffs, 1, axis = 0) # Take 1p of trajectory
g2_average_backoffs    = g2_pool_backoffs.mean(axis = 0) # Take average of trajectory
g2_pool_no_backoffs    = np.array(g2_pool_no_backoffs)
g2_max_no_backoffs        = np.percentile(g2_pool_no_backoffs, 99, axis = 0) # Take 99p of trajectory
g2_min_no_backoffs        = np.percentile(g2_pool_no_backoffs, 1, axis = 0) # Take 1p of trajector
g2_average_no_backoffs = g2_pool_no_backoffs.mean(axis = 0) # Take average of trajectory
g2_max_backoffs = np.reshape(g2_max_backoffs, 13)
g2_min_backoffs = np.reshape(g2_min_backoffs, 13)
g2_max_no_backoffs = np.reshape(g2_max_no_backoffs, 13)
g2_min_no_backoffs = np.reshape(g2_min_no_backoffs, 13)
ax1.fill_between(t_list, g2_max_no_backoffs, g2_min_no_backoffs, facecolor='red', alpha=0.3)
ax1.fill_between(t_list, g2_max_backoffs, g2_min_backoffs, facecolor='green', alpha=0.4)
plt.plot(t_list, g2_average_backoffs, color = 'green', label = '$g_2$ with backoffs ($P_v = 0.01$)')
plt.plot(t_list, g2_average_no_backoffs, color = 'red', label = '$g_2$ without backoffs ($P_v = 0.24$)')
plt.plot([-100,2500], [0,0], color = 'black')
plt.xlabel('Time (h)', fontsize= 30)
plt.ylabel('$g_2$ (mgL$^{-1}$)', fontsize= 30)
plt.xlim(0,240)
plt.ylim(-0.053,0.004)
plt.legend(frameon = 0, fontsize = 30, loc = 'lower left')
ax1.tick_params(axis='x', labelsize= 30)
ax1.tick_params(axis='y', labelsize= 30)

# Inset
axins = plt.axes([0.47,0.57,0.28,0.25]) # First two are location, last two are size of inset
axins.plot(t_list, g2_average_backoffs, color = 'green', label = '$g_2$ with backoffs')
axins.plot(t_list, g2_average_no_backoffs, color = 'red', label = '$g_2$ without backoffs')
axins.fill_between(t_list, g2_max_no_backoffs, g2_min_no_backoffs, facecolor='red', alpha=0.3)
axins.fill_between(t_list, g2_max_backoffs, g2_min_backoffs, facecolor='green', alpha=0.4)
axins.plot([-100,2500], [0,0], color = 'black')
x1, x2, y1, y2 = 235, 240, -0.004, +0.001 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
axins.tick_params(axis='x', labelsize= 20)
axins.tick_params(axis='y', labelsize= 20)

plt.show()


# In[416]:


# Plot g2 trajectory with/without backoffs
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

fig, ax1 = plt.subplots(figsize = (10, 10))
g2_pool_backoffs       = np.array(g2_pool_backoffs)
g2_max_backoffs        = np.percentile(g2_pool_backoffs, 99, axis = 0) # Take 99p of trajectory
g2_min_backoffs        = np.percentile(g2_pool_backoffs, 1, axis = 0) # Take 1p of trajectory
g2_average_backoffs    = g2_pool_backoffs.mean(axis = 0) # Take average of trajectory
g2_pool_MPC    = np.array(g2_pool_MPC)
g2_max_MPC        = np.percentile(g2_pool_MPC, 99, axis = 0) # Take 99p of trajectory
g2_min_MPC        = np.percentile(g2_pool_MPC, 1, axis = 0) # Take 1p of trajector
g2_average_MPC = g2_pool_MPC.mean(axis = 0) # Take average of trajectory
g2_max_backoffs = np.reshape(g2_max_backoffs, 13)
g2_min_backoffs = np.reshape(g2_min_backoffs, 13)
g2_max_MPC = np.reshape(g2_max_MPC, 13)
g2_min_MPC = np.reshape(g2_min_MPC, 13)
ax1.fill_between(t_list, g2_max_MPC, g2_min_MPC, facecolor='blue', alpha=0.3)
ax1.fill_between(t_list, g2_max_backoffs, g2_min_backoffs, facecolor='green', alpha=0.4)
plt.plot(t_list, g2_average_backoffs, color = 'green', label = '$g_2$ with backoffs ($P_v = 0.01$)')
plt.plot(t_list, g2_average_MPC, color = 'blue', label = '$g_2$-MPC ($P_v = 0.53$)')
plt.plot([-100,2500], [0,0], color = 'black')
plt.xlabel('Time (h)', fontsize= 30)
plt.ylabel('$g_2$ (mgL$^{-1}$)', fontsize= 30)
plt.xlim(0,240)
plt.ylim(-0.053,0.004)
plt.legend(frameon = 0, fontsize = 30, loc = 'lower left')
ax1.tick_params(axis='x', labelsize= 30)
ax1.tick_params(axis='y', labelsize= 30)

# Inset
axins = plt.axes([0.47,0.57,0.28,0.25]) # First two are location, last two are size of inset
axins.plot(t_list, g2_average_backoffs, color = 'green', label = '$g_2$ with backoffs')
axins.plot(t_list, g2_average_MPC, color = 'blue', label = '$g_2$-MPC')
axins.fill_between(t_list, g2_max_MPC, g2_min_MPC, facecolor='blue', alpha=0.3)
axins.fill_between(t_list, g2_max_backoffs, g2_min_backoffs, facecolor='green', alpha=0.4)
axins.plot([-100,2500], [0,0], color = 'black')
x1, x2, y1, y2 = 235, 240, -0.004, +0.001 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
axins.tick_params(axis='x', labelsize= 20)
axins.tick_params(axis='y', labelsize= 20)

plt.show()


# ## Compare distribution of scores

# In[407]:


scores_MPC = []
for i in range(len(episode_pool_MPC)):
    score = episode_pool_MPC[i][-1][-1]
    scores_MPC += [score]
scores_MPC


# In[488]:


# Load pre-run data
# with open('./Data/CS1_score_dist_g1g2g3_noise_2000_ep', 'rb') as f:
#     scores_10_noise = pickle.load(f)
# with open('./Data/CS1_score_dist_g1g2g3_30%_noise_2000_ep', 'rb') as f:
#     scores_30_noise = pickle.load(f)
# with open('./Data/CS1_score_dist_g1g2g3_50%_noise_2000_ep', 'rb') as f:
#     scores_50_noise = pickle.load(f)
# with open('./Data/CS1_score_dist_MPC', 'rb') as f:
#     scores_MPC = pickle.load(f)


# In[489]:


# Remove outliers
scores_10_noise = pd.DataFrame(scores_10_noise)
scores_10_noise = scores_10_noise[scores_10_noise > 0.16]
scores_10_noise = scores_10_noise.dropna()
scores_10_noise = scores_10_noise.values.tolist()
scores_30_noise = pd.DataFrame(scores_30_noise)
scores_30_noise = scores_30_noise[scores_30_noise > 0.16]
scores_30_noise = scores_30_noise.dropna()
scores_30_noise = scores_30_noise.values.tolist()
scores_50_noise = pd.DataFrame(scores_50_noise)
scores_50_noise = scores_50_noise[scores_50_noise > 0.16]
scores_50_noise = scores_50_noise.dropna()
scores_50_noise = scores_50_noise.values.tolist()


# In[497]:


# Plot distribution of objective values
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.distplot(scores_10_noise, norm_hist = True, label = 'Objective-10%-noise ($\sigma$ = {})'.format(round(np.std(scores_10_noise),4)), color = 'green', bins = 50)
ax = sns.distplot(scores_30_noise, norm_hist = True, label = 'Objective-30%-noise ($\sigma$ = {})'.format(round(np.std(scores_30_noise),4)), color = 'orange', bins = 50)
ax = sns.distplot(scores_50_noise, norm_hist = True, label = 'Objective-50%-noise ($\sigma$ = {})'.format(round(np.std(scores_50_noise),4)), color = 'red', bins = 50)
# ax = sns.distplot(scores_MPC, norm_hist = True,  label = 'Objective (MPC)', color = 'blue', bins = 75)
# ax.set(xlabel= 'Objective score', ylabel = 'Fraction of samples')
plt.xlabel('Objective value', fontsize = 30)
plt.ylabel('No. of samples', fontsize = 30)
plt.xlim(0.155,0.1775)
plt.ylim(0, 600)
plt.legend(frameon = 0, fontsize = 30, loc = 'upper left')
ax.tick_params(axis='x', labelsize= 30)
ax.tick_params(axis='y', labelsize= 30)
plt.show()


# ## Make gif for nitrate trajectory

# In[132]:


my_images = []

for i in range(len(episode_bank)):
    if i <= 20:
        for j in range(4):
            N_lim = 800 + 500*(0.995**(i*10))
            episode_pool = episode_bank[i]   

            fig, ax = plt.subplots(figsize = (7,5))
            Cn_pool = []
            for episode in episode_pool: # Idx refers to nth episode in episode pool
                Cn_list = []
                t_list = []
                t = 0
                for step in episode:
                    Cn_list += [step[0][1]]
                    t_list  += [t]
                    t       += tf
                plt.plot(t_list, Cn_list, color = 'grey', alpha = 0.5)
                Cn_pool += [Cn_list]
            Cn_pool = np.array(Cn_pool)
            Cn_average = Cn_pool.mean(axis=0)
            plt.plot(t_list, Cn_average, color = 'b', label = 'Nitrate conc. \nIteration %.0f' % (i*10))
            if i < 90:
                plt.plot([-50, 300], [N_lim, N_lim], linestyle = '--', color = 'r')
            else:
                plt.plot([-50, 300], [N_lim, N_lim], linestyle = '--', linewidth = 5, color = 'r')
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(which='major', length=10, width=1, direction='out')
            ax.tick_params(which='minor', length=4, width=1, direction='out')
            plt.xlim(-5,250)
            plt.ylim(0,2100)
            plt.legend(frameon = 0)
            plt.show()
            # For saving into gif
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image  = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            my_images.append(image)
    
    else:
        N_lim = 800 + 500*(0.995**(i*10))
        episode_pool = episode_bank[i]   

        fig, ax = plt.subplots(figsize = (7,5))
        Cn_pool = []
        for episode in episode_pool: # Idx refers to nth episode in episode pool
            Cn_list = []
            t_list = []
            t = 0
            for step in episode:
                Cn_list += [step[0][1]]
                t_list  += [t]
                t       += tf
            plt.plot(t_list, Cn_list, color = 'grey', alpha = 0.5)
            Cn_pool += [Cn_list]
        Cn_pool = np.array(Cn_pool)
        Cn_average = Cn_pool.mean(axis=0)
        plt.plot(t_list, Cn_average, color = 'b', label = 'Nitrate conc. \nIteration %.0f' % (i*10))
        if i < 90:
            plt.plot([-50, 300], [N_lim, N_lim], linestyle = '--', color = 'r')
        else:
            plt.plot([-50, 300], [N_lim, N_lim], linestyle = '--', linewidth = 5, color = 'r')
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='major', length=10, width=1, direction='out')
        ax.tick_params(which='minor', length=4, width=1, direction='out')
        plt.xlim(-5,250)
        plt.ylim(0,2100)
        plt.legend(frameon = 0)
        plt.show()
        # For saving into gif
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image  = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        my_images.append(image)
        

# save images as a gif    
# imageio.mimsave('./Data/CS1_Nitrate_modified.gif', my_images, fps=10)


# ## Make gif for biomass trajectory

# In[124]:


my_images = []

for i in range(len(episode_bank)):
# for i in range(1):
    episode_pool = episode_bank[i]   

    fig, ax = plt.subplots(figsize = (7,5))
    Cx_pool = []
    for episode in episode_pool: # Idx refers to nth episode in episode pool
        Cx_list = []
        t_list = []
        t = 0
        for step in episode:
            Cx_list += [step[0][0]]
            t_list  += [t]
            t       += tf
        plt.plot(t_list, Cx_list, color = 'grey', alpha = 0.5)
        Cx_pool += [Cx_list]
    Cx_pool = np.array(Cx_pool)
    Cx_average = Cx_pool.mean(axis=0)
    plt.plot(t_list, Cx_average, color = 'b', label = 'Biomass conc. \nIteration %.0f' % (i*10))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='major', length=10, width=1, direction='out')
    ax.tick_params(which='minor', length=4, width=1, direction='out')
    plt.xlim(-5,250)
    plt.ylim(0,20)
    plt.legend(frameon = 0, loc = 'upper left')
    plt.show()

    # For saving into gif
    # Used to return the plot as an image array 
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image  = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    my_images.append(image)

# save images as a gif    
# imageio.mimsave('./Data/CS1_Biomass.gif', my_images, fps=10)


# ## Make gif for product trajectory

# In[131]:


my_images = []

for i in range(len(episode_bank)):
# for i in range(1):
    episode_pool = episode_bank[i]   

    fig, ax = plt.subplots(figsize = (7,5))
    Cq_pool = []
    for episode in episode_pool: # Idx refers to nth episode in episode pool
        Cq_list = []
        t_list = []
        t = 0
        for step in episode:
            Cq_list += [step[0][2]]
            t_list  += [t]
            t       += tf
        plt.plot(t_list, Cq_list, color = 'grey', alpha = 0.5)
        Cq_pool += [Cq_list]
    Cq_pool = np.array(Cq_pool)
    Cq_average = Cq_pool.mean(axis=0)
    plt.plot(t_list, Cq_average, color = 'b', label = 'Product conc. \nIteration %.0f' % (i*10))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='major', length=10, width=1, direction='out')
    ax.tick_params(which='minor', length=4, width=1, direction='out')
    plt.xlim(-10,250)
    plt.ylim(-0.01,0.21)
    plt.legend(frameon = 0, loc = 'upper left')
    plt.show()

    # For saving into gif
    # Used to return the plot as an image array 
    # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image  = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    my_images.append(image)

# save images as a gif    
# imageio.mimsave('./Data/CS1_Product.gif', my_images, fps=10)

