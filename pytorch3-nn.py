import os
import sys
import warnings
import torch
from torchvision import datasets
from torchvision import io
from torchvision import models
from torchvision import transforms
from torchvision import ops
from torchvision import utils
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#print(sys.path)

train = datasets.MNIST('', train=True, download=True,
					transform = transforms.Compose([
						transforms.ToTensor()
						]))

test = datasets.MNIST('', train=False, download=True,
					transform = transforms.Compose([
						transforms.ToTensor()
						]))

#out of sample testing data

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)



class Net(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 64) #input 784 total, output 64
		self.fc2 = nn.Linear(64, 64) #input 64 total, output 64
		self.fc3 = nn.Linear(64, 64) #input 64 total, output 64
		self.fc4 = nn.Linear(64, 10) #input 64 total, output 10 classes

	def forward(self, x):	
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		
		return F.log_softmax(x, dim=1) #dim =  dimension


		#return x


net = Net()
print(net)

X = torch.rand((28,28))

X = X.view(-1,28*28)

optimiser = optim.Adam(net.parametre(), 1r=0.001)

