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
        self.gamma = 0.9;
        
        self.gridSize = env.gridSize
        self.num_states = self.gridSize*self.gridSize # number of states
        self.num_actions = env.action_space.n # number of actions

        ## POLICY / value-FUNCTIONS
        self.theta = np.random.rand(self.num_states,self.num_actions) # policy parameter theta # should we parametrize pi??
        ##self.pi = np.random.rand(self.num_states,self.num_actions)
        self.value = np.zeros((self.num_states))

        ## REWARD
        self.psi = np.random.rand(self.num_states); # reward function parameter

    # store all trajectories data (states and actions seperately)
    def store_trajectories(self, TAU):
        self.TAU_S = TAU[0];
        self.TAU_A = TAU[1];

    # get an action given state:  a ~ pi(.|s;theta)
    #def policy(self,env,s):
    #    return env.action_space.sample() # now returning random action

    # value-iteration
    # perform value iteration with the current R(s;psi) and update pi(a|s;theta)
    def value_iteration(self,env):

        T_sas=0.01; update_difference=float("inf");

        while update_difference < 0.01:
            for s in range(self.num_states):
                old_value = self.value[s]
                self.value[s] = np.max( [np.sum([ T_sas*(self.reward(s_prime)+self.gamma*self.value[s_prime]) for s_prime in range(self.num_states)]) for a in range(self.num_actions)])
                update_difference = max(update_difference, abs(old_value-self.value[s]))                

    # get reward: r(s;psi)
    def reward(self,s):
        return self.psi[s];
                    
    # get policy: pi(a|s,theta)
    def policy(self,env,s,a):
        
        return np.exp(self.theta[s,a])/ np.sum([np.exp(self.theta[s,b]) for b in range(self.num_actions)])
        #return self.pi[s,a]
          
    # compute P(s | pi_theta, T) 
    def get_state_visitation_frequency(self,env):

        # mu[state, time] is the prob of visiting state s at time t
        mu = np.zeros([self.num_states, self.tau_len]) 

        for tau_t0 in self.TAU_S[0,:]: # look at t=0 for each trajectory
            mu[int(tau_t0),0] += 1 # initialize mu(.,t=0)
            
        mu[:,0] = mu[:,0]/self.tau_num

        #for time in range(self.tau_len-1):
        for state in range(self.num_states):
            for action in range(self.num_actions):
                for state_next in range(self.num_states):
                    print("(state:",state,",action:",action,",state_p",state_next,")  mu(s):",mu[state,0]," pi(a|s):", self.policy(env,state,action), " T(s'|s,a):", env.T_sas(state,action,state_next))
                    mu[state_next, 1] += mu[state, 0] * self.policy(env,state,action) * env.T_sas(state,action,state_next)

        #print(np.sum([self.policy(env,4,action) for action in range(self.num_actions)]))
        #print(np.sum([env.T_sas(4,3,state_prime) for state_prime in range(self.num_states)]))
        
        print(np.sum(mu[:,0]))
        print(np.sum(mu[:,1]))
        print(np.sum(mu[:,2]))
        print(np.sum(mu[:,3]))
        
        return np.mean(mu, 1) # squeeze throughout time and return

    def get_grad(self):
        '''
        computer grad_psi L(psi) = - 1/D * sum_{tau \in TAU}
        '''
        
        grad = 1; ## TODO: write gradient
    
    
    def update(self,env):

        # STEP:1
        # TODO: solve for optimal policy: do policy iteration on r(s;psi)
        self.value_iteration(env);
        ## pi = get_policy();

        # STEP:2
        # working: but must add transition function from grid world
        self.get_state_visitation_frequency(env)

        # STEP:3
        self.get_grad();

        # STEP:4
        # TODO: update psi of r(s;psi)

        # REPEAT
        
        print("updating..")    
