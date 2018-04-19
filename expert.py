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
    def __init__(self,env,tau_num,tau_len):

        self.HEIRARCHICAL = True;
        self.PLOT_V = False
        self.EGREEDY = False
        
        # states
        self.gridSize = env.gridSize
        if(self.HEIRARCHICAL):
            self.num_states = self.gridSize*self.gridSize*2
        else:
            self.num_states = self.gridSize*self.gridSize

        # meta-parameters
        self.epsilon = 1.0; # e-greedy
        self.alpha = 0.01; # learning rate
        self.gamma = 0.7; # discount factor
        self.carried_flag = None;
        
        self.q = np.zeros((self.num_states, env.action_space.n)); 

        if(self.PLOT_V):
            self.init_value_plot()

        ## initialize trajectory details ##
        self.tau_num = tau_num;
        self.tau_len = tau_len;

        self.TAU_S = np.zeros((self.tau_len, self.tau_num))-1 # matrix of states with all trajectories
        self.TAU_A = np.zeros((self.tau_len, self.tau_num))-1 # matrix of actions with all trajectories
        
    def reset(self,env):
        env.reset()
        
    def get_action(self, env):
        if(self.HEIRARCHICAL):
            s = env.agentPos[0] + self.gridSize*env.agentPos[1] + int((env.carrying!=None)*self.num_states/2);
        else:
            s = env.agentPos[0] + self.gridSize*env.agentPos[1];
            
        if(self.EGREEDY):
            if random.uniform(0, 1) <= self.epsilon:
                return env.action_space.sample()
            else:
                return np.argmax(self.q[s,:])
        else:
            pi = np.exp(self.q[s,:]-np.max(self.q))
            pi = pi/np.sum(pi)
            return np.random.choice(range(env.action_space.n), p=pi)

    def store_tau(self,episode,time,state,action):
        if(self.HEIRARCHICAL):
            self.TAU_S[time][episode] = int(state%(self.gridSize*self.gridSize));
        else:
            self.TAU_S[time][episode] = int(state);            
        self.TAU_A[time][episode] = int(action);

    def get_tau(self,PRINT):
        if(PRINT):
            print(self.TAU_S.T,self.TAU_A.T)
        return (self.TAU_S,self.TAU_A)
        
    def update_q(self,s,a,r,s_prime):
        self.q[s,a] = self.q[s,a] + self.alpha*(r + self.gamma*np.max(self.q[s_prime,:]) - self.q[s,a])

    def init_value_plot(self):

        # get initial plot config
        fig = plt.figure(figsize=(5,5))
        self.axes = fig.add_subplot(111)
        self.axes.set_autoscale_on(True)

        # get value from q-function
        q_max = np.max(self.q,1)
        v = np.reshape(q_max[:self.gridSize*self.gridSize],(self.gridSize,self.gridSize))

        # plot value function
        self.v_plotter = plt.imshow(v,interpolation='none', cmap='viridis', vmin=v.min(), vmax=v.max());
        plt.colorbar(); plt.xticks([]); plt.yticks([]); self.axes.grid(False);
        plt.title('true value function'); plt.ion(); plt.show();
        
    def see_value_plot(self):
        q_max = np.max(self.q,1)        
        v = np.reshape(q_max[:self.gridSize*self.gridSize],(self.gridSize,self.gridSize))        
        self.v_plotter.set_data(v)
        plt.clim(v.min(),v.max()) 
        plt.draw(); plt.show()
        plt.pause(0.0001)
        
    def update(self,env,episode,STORE):
        
        if(STORE):
            self.epsilon = 0.2
        else:
            self.epsilon = 0.3        

        if(self.HEIRARCHICAL):
            s = env.agentPos[0]+self.gridSize*env.agentPos[1]+int((env.carrying!=None)*self.num_states/2);
        else:
            s = env.agentPos[0]+self.gridSize*env.agentPos[1];            
        a = self.get_action(env)

        obs, r, done, info = env.step(a)

        # sub-goal #experimental
        #r = r or (env.carrying!=None and self.carried_flag==None)
        #self.carried_flag = env.carrying        
        
        if(self.HEIRARCHICAL):
            s_prime = env.agentPos[0] + self.gridSize*env.agentPos[1]+int((env.carrying!=None)*self.num_states/2);
        else:
            s_prime = env.agentPos[0] + self.gridSize*env.agentPos[1];

        self.update_q(s,a,r,s_prime)

        if done and self.PLOT_V and episode%100==0:
            self.see_value_plot()

        if(STORE):
            self.store_tau(episode,env.stepCount-1,s,a);
            #if(done):
            #    self.store_tau(episode,env.stepCount,s_prime,-1);         
        return done
        
            

