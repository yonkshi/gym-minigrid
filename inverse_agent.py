#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import ipdb
import random

import matplotlib.pyplot as plt

# -----------------------------
# ---MaxEnt Inverse RL agent---
# -----------------------------
class InverseAgentClass():
    
    def __init__(self, env):

        self.tau_num = 100; # number of trajectories
        self.tau_len = 25; # length of each trajectory
        
        self.gridSize = env.gridSize
        self.num_states = self.gridSize*self.gridSize # number of states
        self.num_actions = env.action_space.n # number of actions

        self.TAU = np.random.randint(0,4,size=(self.tau_num, self.tau_len)) # matrix with all trajectories

        self.theta = np.random

    def policy(self,env,s):
        return env.action_space.sample()
        
    # compute P(s | pi_theta, T) 
    def get_state_visitation_frequency(self,env):

        # mu[state, time] is the prob of visiting state s at time t
        mu = np.zeros([self.num_states, self.tau_len]) 
        
        for tau in self.TAU:
            mu[tau[0]] += 1
        mu[:,0] = mu[:,0]/self.tau_len
            
        for time in range(self.tau_len-1):
            for state in range(self.num_states):
                for state_previous in range(self.num_states):

                    # assuming (for now) that T(s,a,s') is equiprobable if states are nearby
                    T_sas = 1.0;#1.0/self.num_actions if abs(state-state_previous) <1 else 0
                
                    mu[state, time+1] += mu[state_previous, time] * T_sas * self.policy(env,state)
                    
        p = np.sum(mu, 1)
        print(mu)
        return p
        
    def update(self,env):
        
        print("Done!")        
