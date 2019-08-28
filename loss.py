import torch
import torch.nn as nn
from torch.nn import functional as f

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable

class InstanceLoss(_Loss):
	def __init__(self, hm=2.0, sampling_size=400, size_average=True):
		super().__init__(size_average)
		# Needed to avoid logarithm of zero
		self.epsilon = 1e-9

		# Margin for Hinge Loss
		self.hm = hm
		self.sampling_size = sampling_size

	def forward(self, input, target):
		# Loss initialization
		loss = Variable(torch.zeros(1)).cuda()

		# If the target has 3 dimensions let's add one
		if len(target.size()) == 3:
			target.unsqueeze(1)

		# First of all extract relevant information from the input
		[batchsize, classes, height, width] = input.size()

		# get input probabilities
		p = f.softmax(input, dim = 1)

		# Iterate on each sample
		for i in range(0, batchsize):

			# It's not feasible to evaluate all the pixels in the image because of quadratic comparisons
			# We use a sampling strategy

			# Find the indices of the not zero elements in target
			target_nonzero_pixels = torch.nonzero(target[i].view(-1)).squeeze()
			if(target_nonzero_pixels.numel() > 0):
				# Get the uniform sampling indices, the number is fixed regardless of the dimension of the instance
				sampling_indices = Variable(torch.zeros(self.sampling_size).uniform_(0, target_nonzero_pixels.size()[0] - 0.01)).long().cuda()
				indices_to_keep = torch.index_select(target_nonzero_pixels, 0, sampling_indices)

				# Get the pixels both in target and in input
				p_sampled = torch.index_select(p[i][1:].view(classes - 1, -1), 1, indices_to_keep)
				t_sampled = torch.index_select(target[i].view(-1), 0, indices_to_keep)

				# Building the R matrix as described in the paper
				R = Variable(torch.eq(t_sampled.unsqueeze(1).expand(-1, self.sampling_size), t_sampled.unsqueeze(0).expand(self.sampling_size, -1)).data).float().cuda()
				# We are not intrested in the diagonal, it is obvious that one pixel belongs to the same instance of itself
				diag = Variable(1 - torch.eye(self.sampling_size)).cuda()

				# Calculation of the loss term for the pairwise loss
				# self.sampling_size * (self.sampling_size - 1) is the number of pairs
				tik = Variable(p_sampled.unsqueeze(2).expand(-1,-1,self.sampling_size).data).cuda()
				tjk = p_sampled.unsqueeze(1).expand(-1, self.sampling_size, -1)
				dkl = torch.sum(tik * (torch.log(tik + self.epsilon) - torch.log(tjk + self.epsilon)), 0)
				loss += torch.sum(diag * ((R * dkl) + (1 - R)*torch.clamp(self.hm - dkl, min=0.0))) / ((self.sampling_size)*(self.sampling_size - 1))

			# Calculation of the loss term for bg/fg segmentation
			bg_mask = torch.eq(target[i],0).float()
			bg_term = torch.sum(bg_mask * torch.log(p[i][0]))
			fg_term = torch.sum((1 - bg_mask) * torch.log(torch.sum(p[i][1:], 0)))
			loss += - (1 / (width * height)) * (fg_term + bg_term)

		return (loss / batchsize)

