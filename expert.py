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

        self.epsilon = 0.5;
        
        self.q = np.zeros((self.num_states, env.action_space.n)); 
        self.init_value_plot()
        
    def reset(self,env):
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)
        
    def get_action(self, env):
        s = env.agentPos[0] + self.gridSize*env.agentPos[1];

        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q[s,:])

    def update_q(self,s,a,r,s_prime):

        ## obs['image'].flatten() # THIS IS ALL OBSERVATIONS OF WORLD!

        self.q[s,a] = self.q[s,a] + 0.01*(r + 0.7*np.argmax(self.q[s_prime,:]) - self.q[s,a])

    def init_value_plot(self):

        # get initial plot config
        fig = plt.figure(figsize=(3,3))
        self.axes = fig.add_subplot(111)
        self.axes.set_autoscale_on(True)

        # get value from q-function
        q_max = np.max(self.q,1)
        v = np.reshape(q_max,(self.gridSize,self.gridSize))

        # plot value function
        self.v_plotter = plt.imshow(v,interpolation='none', cmap='viridis', vmin=v.min(), vmax=v.max());
        plt.xticks([]); plt.yticks([]); self.axes.grid(False);
        plt.ion();
        plt.show();
        
    def see_value_plot(self):
        q_max = np.max(self.q,1)        
        v = np.reshape(q_max,(self.gridSize,self.gridSize))        
        self.v_plotter.set_data(v)
        plt.clim(v.min(),v.max()) 
        plt.draw(); plt.show()
        plt.pause(0.0001)
        
    def update(self,env,DEBUG):
        
        if(DEBUG):
            self.epsilon = 0.9

        s = env.agentPos[0] + self.gridSize*env.agentPos[1];
        a = self.get_action(env)
        
        obs, r, done, info = env.step(a)

        s_prime = env.agentPos[0] + self.gridSize*env.agentPos[1];

        self.update_q(s,a,r,s_prime)
        
        if(r):
            print('step=%s, reward=%s' % (env.stepCount, r))

        if done:
            print("done!")
            self.reset(env)
            self.see_value_plot()

        if(DEBUG):
            print(s,a,"->",end='')
        
            

