#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import arrow

class NonNegativeClipper(object):
	def __init__(self):
		pass

	def __call__(self, module):
		"""enforce non-negative constraints"""
		if hasattr(module, 'B_nonzero'):
			B0 = module.B_nonzero[0].data
			B1 = module.B_nonzero[1].data
			module.B_nonzero[0].data = torch.clamp(B0, min=0.)
			module.B_nonzero[1].data = torch.clamp(B1, min=0.)
		if hasattr(module, 'A_nonzero'):
			A0 = module.A_nonzero[0].data
			A1 = module.A_nonzero[1].data
			module.A_nonzero[0].data = torch.clamp(A0, min=0.)
			module.A_nonzero[1].data = torch.clamp(A1, min=0.)
		if hasattr(module, 'H_nonzero'):
			H0 = module.H_nonzero[0].data
			H1 = module.H_nonzero[1].data
			module.H_nonzero[0].data = torch.clamp(H0, min=0.)
			module.H_nonzero[1].data = torch.clamp(H1, min=0.)

class COVID19linear(nn.Module):
	'''
	Class to learn the parameters in a state space model
	via MLE. We then make predictions via the Kalman filter
	'''
	def __init__(self, p, adj, dist, n_counties, n_mobility, n_covariates):
		'''
		Initialize State Space model

		Args:
		- p              : number of lags
		- adj            : adjacent matrix                                       [ n_counties, n_counties ]
		- dist           : distance matrix holding distances between each county [ n_counties, n_counties ]
		- n_counties     : number of counties
		- n_mobility     : number of mobility types
		- n_covariates   : number of covariates included in the model

		'''
		super().__init__()

		self.l1ratio    = 1e1
		self.l2ratio    = 1e3
		self.p          = p # number of lags
		self.n_counties = n_counties
		self.n_mobility = n_mobility
		self.dist       = torch.FloatTensor(dist) # distance matrix

		# non-zero entries of matrices Lambda (spatio-temporal dependences)
		n_nonzero       = len(np.where(adj == 1)[0])
		self.coords     = torch.LongTensor(np.where(adj == 1))
		self.B_nonzero  = nn.ParameterList([])
		self.A_nonzero  = nn.ParameterList([])
		self.H_nonzero  = nn.ParameterList([])
		for tau in range(self.p):
			self.B_nonzero.append(torch.nn.Parameter(torch.randn((n_nonzero), requires_grad=True)))
			self.A_nonzero.append(torch.nn.Parameter(torch.randn((n_nonzero), requires_grad=True)))
			self.H_nonzero.append(torch.nn.Parameter(torch.randn((n_nonzero), requires_grad=True)))
		
		# community mobility
		self.mu         = torch.nn.Parameter(torch.randn(n_mobility, self.p), requires_grad=True)
		self.nu         = torch.nn.Parameter(torch.randn(n_mobility, self.p), requires_grad=True)
		# demographic census
		self.upsilon    = torch.nn.Parameter(torch.randn(n_covariates), requires_grad=True)
		self.zeta       = torch.nn.Parameter(torch.randn(n_covariates), requires_grad=True)

		# exponential decaying factor
		self.theta      = 1e3 # torch.nn.Parameter(torch.ones(1).float(), requires_grad=True)
		# covariance matrix
		self.Sigma      = self.theta * torch.exp(-self.theta * self.dist) # [ n_counties, n_counties ]

	def loss(self, C, D, C_hat, D_hat):
		'''
		Calculate the reqularized l2 loss of the model prediction over T with exponentially decreasing weights

		Args: 
		C     : The observed confirmed cases  [ T, n_counties ]
		D     : The observed death counts     [ T, n_counties ]
		C_hat : The predicted confirmed cases [ T, n_counties ]
		D_hat : The predicted death counts    [ T, n_counties ]

		Returns:
		The regularized l2 loss
		'''
		# take log for poisson regression
		# C = torch.log(C + 1e-2)
		# D = torch.log(D + 1e-2)

		T, n_counties = C.shape
		inv           = torch.inverse(self.Sigma)
		# Add up the loss for each week with exponentially decreasing weights
		D_loss = torch.stack([ self.l2loss(D[i], D_hat[i], inv) for i in range(T) ]).sum()
		C_loss = torch.stack([ self.l2loss(C[i], C_hat[i], inv) for i in range(T) ]).sum()

		# D_loss = torch.stack([ 0.99 ** (T - i) * self.l2loss(D[i], D_hat[i], inv) for i in range(T) ]).sum()
		# C_loss = torch.stack([ 0.99 ** (T - i) * self.l2loss(C[i], C_hat[i], inv) for i in range(T) ]).sum()

		B_nonzeros = torch.stack([ self.B_nonzero[tau] for tau in range(self.p) ], 1)
		A_nonzeros = torch.stack([ self.A_nonzero[tau] for tau in range(self.p) ], 1)
		H_nonzeros = torch.stack([ self.H_nonzero[tau] for tau in range(self.p) ], 1)
		# # Calculate the l1 norm
		# l1_norm = torch.norm(B_nonzeros, p=1) + torch.norm(A_nonzeros, p=1) + torch.norm(H_nonzeros, p=1)
		# # Calculate the l2 norm
		# l2_norm = torch.norm(B_nonzeros, p=2) + torch.norm(A_nonzeros, p=2) + torch.norm(H_nonzeros, p=2)
		# print("obj: %.5e\tl1 norm: %.5e\tl2 norm: %.5e." % ((0.9 * D_loss + 0.1 * C_loss) / (T * n_counties), self.l1ratio * l1_norm, self.l2ratio * l2_norm))
		return (0.9 * D_loss + 0.1 * C_loss) / (T * n_counties) # + self.l1ratio * l1_norm + self.l2ratio * l2_norm

	def l2loss(self, y, yhat, sigmaInv):
		'''
		The l2 loss of two inputs y and yhat, with a covariance matrix Sigma

		Args:
		- y         : The observed values                        [ n_counties ]
		- yhat      : The predicted values                       [ n_counties ]
		- sigmaInv  : The inverse of the covariance matrix Sigma [ n_counties, n_counties ]

		Returns:
		loss        : scalar of l2 loss from the inputs
		'''
		loss = torch.matmul((y - yhat), sigmaInv) # [ n_counties ]
		loss = torch.matmul(loss, (y - yhat).T)
		return loss

	def forward(self, C, D, M, cov):
		'''
		Customized forward function
		'''
		# construct spatial coefficents.
		self.B = [ torch.sparse.FloatTensor(self.coords, self.B_nonzero[tau], torch.Size([self.n_counties, self.n_counties])).to_dense() for tau in range(self.p) ]
		self.A = [ torch.sparse.FloatTensor(self.coords, self.A_nonzero[tau], torch.Size([self.n_counties, self.n_counties])).to_dense() for tau in range(self.p) ]
		self.H = [ torch.sparse.FloatTensor(self.coords, self.H_nonzero[tau], torch.Size([self.n_counties, self.n_counties])).to_dense() for tau in range(self.p) ]

		C_hat, D_hat = [], []
		T = C.shape[0]
		for t in range(self.p, T):
			yc, yd = self.predict(t, C, D, M, cov)
			C_hat.append(yc)
			D_hat.append(yd)
		return C_hat, D_hat # [ T - p, n_counties ]

	def predict(self, t, C, D, M, cov):
		'''
		Forward of the state space model.

		Args:
		- t   : predicted week
		- C   : Tensor of the spatially structured confirmed cases [ T, n_counties ]
		- D   : Tensor of the spatially structured deaths          [ T, n_counties ]
		- M   : Tensor of mobility data                            [ n_mobility, T, n_counties ]
		- cov : Tensor of census data                              [ n_covariates, n_counties ]

		Returns:
		c_hat : predicted confirmed cases (list of tensors) [ T, n_counties ]
		d_hat : predicted deaths (list of tensors)          [ T, n_counties ]
		'''
		n_counties = C.shape[1]
		# predict dt and ct for the last
		ct = C[t-self.p:t, :].clone()    # [ p, n_counties ]
		dt = D[t-self.p:t, :].clone()    # [ p, n_counties ]
		mt = M[:, t-self.p:t, :].clone() # [ n_mobility, p, n_counties ]

		c2c = torch.stack([ torch.matmul(ct[tau].clone(), self.B[tau]) for tau in range(self.p) ], dim=1).sum(1)
		c2d = torch.stack([ torch.matmul(ct[tau].clone(), self.H[tau]) for tau in range(self.p) ], dim=1).sum(1)
		d2d = torch.stack([ torch.matmul(dt[tau].clone(), self.A[tau]) for tau in range(self.p) ], dim=1).sum(1)

		mu = self.mu.unsqueeze(-1).repeat(1, 1, n_counties) 
		nu = self.nu.unsqueeze(-1).repeat(1, 1, n_counties)
		
		# Make predictions
		c_hat = c2c + (mt * mu).sum(0).sum(0) + torch.matmul(self.upsilon, cov)    # [ n_counties ]
		d_hat = c2d + d2d + (mt * nu).sum(0).sum(0) + torch.matmul(self.zeta, cov) # [ n_counties ]

		c_hat = torch.nn.functional.softplus(c_hat)
		d_hat = torch.nn.functional.softplus(d_hat)

		return c_hat, d_hat # [ n_counties ]
