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
    
    def __init__(self, env, tau_num, tau_len):

        self.tau_num = tau_num; # number of trajectories
        self.tau_len = tau_len; # length of each trajectory

        self.gridSize = env.gridSize
        self.num_states = self.gridSize*self.gridSize # number of states
        self.num_actions = env.action_space.n # number of actions

        self.theta = np.random.rand(self.num_states,self.num_actions) # policy parameter theta

        self.psi = np.random.rand(self.num_states); # reward function parameter

    # store all trajectories data (states and actions seperately)
    def store_trajectories(self, TAU):
        self.TAU_S = TAU[0];
        self.TAU_A = TAU[1];

    # get an action given state:  a ~ pi(.|s;theta)
    #def policy(self,env,s):
    #    return env.action_space.sample() # now returning random action

    # get policy: pi(a|s,theta)
    def policy(self,env,s,a):
        return np.exp(self.theta[s,a])/ np.sum([np.exp(self.theta[s,b]) for b in range(self.num_actions)])
    
    
    # compute P(s | pi_theta, T) 
    def get_state_visitation_frequency(self,env):

        # mu[state, time] is the prob of visiting state s at time t
        mu = np.zeros([self.num_states, self.tau_len]) 

        for tau_t0 in self.TAU_S[0,:]: # look at t=0 for each trajectory
            mu[int(tau_t0),0] += 1 # initialize mu(.,t=0)
            
        mu[:,0] = mu[:,0]/self.tau_num

        for time in range(self.tau_len-1):
            for state in range(self.num_states):
                for action in range(self.num_actions):
                    for state_next in range(self.num_states):
                        T_sas = 1 if state_next==state else 0.0  # TODO: get proper transition matrix                
                        mu[state_next, time+1] += mu[state, time] * self.policy(env,state,action) * T_sas
                             
        return np.sum(mu, 1) # squeeze throughout time and return
        
    def update_psi(self,env):
        self.get_state_visitation_frequency(env) 
        print("updating..")    
