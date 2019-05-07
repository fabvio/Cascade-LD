# LCNet Definition
#######################

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

class Block(nn.Module):
	def __init__(self, channels_in, channels_out):
		super().__init__()
		self.layers = nn.ModuleList()
		conv_1 = nn.Conv2d(channels_in, channels_out, kernel_size=(3,3), padding=(1,1))
		self.layers.append(conv_1)
		self.layers.append(nn.BatchNorm2d(channels_out, eps=1e-03))
		self.layers.append(nn.ReLU())

		conv_2 = nn.Conv2d(channels_out, channels_out, kernel_size=(3,3), padding=(1,1))
		self.layers.append(conv_2)
		self.layers.append(nn.BatchNorm2d(channels_out, eps=1e-03))
		self.layers.append(nn.ReLU())

		self.layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

	def forward(self, input):
		output = input
		for layer in self.layers:
			output = layer(output)
		return output

class Net(nn.Module):
	def __init__(self, num_classes, w, h):
		super().__init__()
		
		self.linw = int(w/8)
		self.linh = int(h/8)

		self.layers = nn.ModuleList()
		self.linear_layers = nn.ModuleList()

		self.layers.append(Block(3,16))
		self.layers.append(Block(16,32))
		self.layers.append(Block(32,64))
		
		linear_1 = nn.Linear(64 * self.linw * self.linh,1024)

		self.linear_layers.append(linear_1)
		self.linear_layers.append(nn.BatchNorm1d(1024, eps=1e-03))
		self.linear_layers.append(nn.ReLU())
		
		linear_2 = nn.Linear(1024,num_classes)
		self.linear_layers.append(linear_2)

	def forward(self, input):
		output = input

		for layer in self.layers:
			output = layer(output)
		output = output.view(-1, 64 * self.linw * self.linh)
		
		for layer in self.linear_layers:
			output = layer(output)

		return output

