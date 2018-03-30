#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid

def main():
    basic_mode = True
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

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)


        action = 0
        if basic_mode:
            if keyName == 'LEFT':
                action = env.actions.move_left
            elif keyName == 'RIGHT':
                action = env.actions.move_right
            elif keyName == 'UP':
                action = env.actions.move_up
            elif keyName == 'DOWN':
                action = env.actions.move_down
        else:
            if keyName == 'LEFT':
                action = env.actions.left
            elif keyName == 'RIGHT':
                action = env.actions.right
            elif keyName == 'UP':
                action = env.actions.forward
            elif keyName == 'SPACE':
                action = env.actions.toggle
            elif keyName == 'CTRL':
                action = env.actions.wait
            else:
                print("unknown key %s" % keyName)
                return

        obs, reward, done, info = env.step(action)

        print('step=%s, reward=%s' % (env.stepCount, reward))
        print(env.observation_space.spaces['image'].shape)

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
