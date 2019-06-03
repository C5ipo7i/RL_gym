from train import train
from Agents import DQN,Priority_DQN
from plot.py import plot
import matplotlib.pyplot as plt
import gym
import numpy as np

# Parameters
BUFFER_SIZE = 10000
MIN_BUFFER_SIZE = 200
BATCH_SIZE = 50
ALPHA = 0.6 # 0.7 or 0.6
START_BETA = 0.5 # from 0.5-1
END_BETA = 1
LR = 0.00025
EPSILON = 1
MIN_EPSILON = 0.01
GAMMA = 0.99
TAU = 0.01
UPDATE_EVERY = 4
CLIP_NORM = 10

def main():
    env = gym.make('LunarLander-v2')
    env.seed(0)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    print('Observation Space {}, Action Space {}'.format(nS,nA))
    seed = 7
    agent = Priority_DQN(nS,nA,seed,UPDATE_EVERY,BATCH_SIZE,BUFFER_SIZE,MIN_BUFFER_SIZE,LR,GAMMA,TAU,CLIP_NORM,ALPHA)

    scores = train(agent,env)

    # plot the scores
    plot(scores)
    

if __name__ == "__main__":
    main()