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
    q_expert = expert.ExpertClass(env)
    maxent_learner = inverse_agent.InverseAgentClass(env)        

    ## expert_mode: get expert trajectories

    renderer = env.render('human')
        
    for episode in range(10):
        for t in range(15):
            
            if(q_expert.update(env,episode,False)):
                q_expert.reset(env)
                break
            
            env.render('human')
            time.sleep(0.01)
            
            # If the window was closed
            if renderer.window == None:
                break
        
    q_expert.reset(env)
        
    for episode in range(10):
        for t in range(15):
            if(q_expert.update(env,episode,True)):
                q_expert.reset(env)
                break
            
            env.render('human')
            time.sleep(0.01)
            
            # If the window was closed
            if renderer.window == None:
                break
    ## get traj    
    TAU = q_expert.get_tau();
    print("TAU_S:",TAU[0])
    print("TAU_A:",TAU[1])
    
    ## inverse RL mode: learn MaxEnt IRL from trajectories

    #maxent_learner.store_trajectories(TAU);
    #maxent_learner.get_state_visitation_frequency(env) 

if __name__ == "__main__":
    main()
