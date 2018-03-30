#!/usr/bin/env python3

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import ipdb
import random

import matplotlib.pyplot as plt

class ExpertClass():
    def __init__(self,env):

        # obviously a bad way to find #states. TODO: find alternative
        self.gridSize = env.gridSize
        self.num_states = self.gridSize*self.gridSize

        self.q = np.random.rand(self.num_states,env.action_space.n)/10
        #self.init_value_plot()
        
    def reset(self,env):
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
        
    def get_action(self, env):
        s = env.agentPos[0] + self.gridSize*env.agentPos[1];

        if random.uniform(0, 1) <= 0.8:
            return np.argmax(self.q[s,:])
        else:
            return env.action_space.sample()

    def update_q(self,a,agentPos,reward):
        q_old = self.q
        
        ## obs['image'].flatten() # THIS IS ALL OBSERVATIONS OF WORLD!
        s = agentPos[0] + self.gridSize*agentPos[1];

        self.q[s,a] = self.q[s,a] + 0.01*(reward + 0.9*q_old[s,a] - self.q[s,a])

    def init_value_plot(self):
        q_max = np.max(self.q,1)
        v = np.reshape(q_max,(self.gridSize,self.gridSize))
        self.v_plotter = plt.imshow(v,interpolation='none', cmap='viridis')        
        plt.vmin=0; plt.vmax=1;
        plt.ion();
        plt.show();
        
    def see_value_plot(self):
        q_max = np.max(self.q,1)
        v = np.reshape(q_max,(self.gridSize,self.gridSize))
        self.v_plotter.set_data(v)        
        plt.draw(); plt.show()
        plt.pause(0.0001)
        
    def update(self,env):
        
        action = self.get_action(env)
        obs, reward, done, info = env.step(action)
        
        self.update_q(action,env.agentPos,reward)           
        if(reward):
            print('step=%s, reward=%s' % (env.stepCount, reward))

        if done:
            print("done!")
            self.reset(env)
            #self.see_value_plot()
            

