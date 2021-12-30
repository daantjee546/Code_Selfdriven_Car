# AI for Autonomous Vehicles - Build a Self-Driving Car

# Importing the libraries

import numpy as np
import random 
import os # use the save and load option
import torch # using pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #optimizer
import torch.autograd as autograd
from torch.autograd import Variable

hidden_neurons = 30


# creating the architecture of a neural network

class Network(nn.Module) : #inheritance (overerven)
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, hidden_neurons) #fc1 is full connection 1, between input layer (INIT) and hidden layer
        self.fc2 = nn.Linear(hidden_neurons, nb_action) 
        
    def forward(self, state): # state is exactly the input of our neural networks and activate the neurons
        x = F.relu(self.fc1(state)) # x represent the hidden neurons, relu activates the hidden neurons
        q_values = self.fc2(x) # we will get the output neurons of our neural network
        return q_values # return the q_values for each possible actions, left, straight or right
    
# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event) # adds a single event to the existing list
        
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) # Get random samples from the memory
        return map(lambda x: Variable(torch.cat(x, 0)), samples) # will do the mapping from the samples to torch variables, that will contain a tensor integrated
    
# Implementing Deep Q-Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma): # input_size = number of dimensions and the vectors that are including your state, your input states ; nb_action = a number of actions, which is a number of possible actions a car can take
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) *50) # T = 7 # the closer to zero the less sure the neural network will be when playing in action and how higher how better
        action = probs.multinomial(len(probs))
        return action.data[0, 0]

    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]
        batch_targets = batch_rewards + self.gamma * batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, new_state, new_reward):
        new_state = torch.Tensor(new_state).float().unsqueeze(0)
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state))
        new_action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(100)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        return new_action
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")








































