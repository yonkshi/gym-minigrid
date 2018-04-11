#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid
import expert
import inverse_agent

def main():
    
    basic_mode = True
    expert_mode = True
    inverse_mode = False
    
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-CaptureTheFlag-Static-v0',
        #default='MiniGrid-LockedRoom-v0',
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    # Load expert agent / inverse learner
    if(expert_mode):
        q_expert = expert.ExpertClass(env)
    else:
        maxent_learner = inverse_agent.InverseAgentClass(env)        


    if(expert_mode):

        renderer = env.render('human')
        
        for train in range(1000):
            action=q_expert.update(env,False)
            
            env.render('human')
            time.sleep(0.01)
        
            # If the window was closed
            if renderer.window == None:
                break
            
        for test in range(10):
            action=q_expert.update(env,True)
            
            env.render('human')
            time.sleep(0.01)
        
            # If the window was closed
            if renderer.window == None:
                break        
    else:

       maxent_learner.get_state_visitation_frequency(env) 

if __name__ == "__main__":
    main()
