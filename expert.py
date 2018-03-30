#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import ipdb

class ExpertClass():
    def __init__(self,env):

        # obviously a bad way to find #states. TODO: find alternative
        num_states = 24*24; #env.observation_space.sample()['image'].shape[0]*env.observation_space.sample()['image'].shape[1]*env.observation_space.sample()['image'].shape[2]

        self.q = np.empty((num_states,env.action_space.n))

        
    def update_q(self,a,agentPos,reward):
        q_old = self.q
        
        ## obs['image'].flatten() # THIS IS ALL OBSERVATIONS OF WORLD!
        s = agentPos[0] + 24*agentPos[1];

        self.q[s,a] = self.q[s,a] + 0.01*(reward + 0.9*q_old[s,a] - self.q[s,a])              
        
    def update(self,env):
        
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        self.update_q(action,env.agentPos,reward)
        
        #print('step=%s, reward=%s' % (env.stepCount, reward))

