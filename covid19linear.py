import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import arrow

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

		self.p          = p # number of lags
		self.n_counties = n_counties
		self.n_mobility = n_mobility
		self.dist       = torch.FloatTensor(dist) # distance matrix

		# non-zero entries of matrices Lambda (spatio-temporal dependences)
		n_nonzero       = len(np.where(adj == 1)[0])
		self.B_nonzero  = torch.nn.Parameter(torch.randn((n_nonzero), requires_grad=True))
		self.A_nonzero  = torch.nn.Parameter(torch.randn((n_nonzero), requires_grad=True))
		self.H_nonzero  = torch.nn.Parameter(torch.randn((n_nonzero), requires_grad=True))
		# matrices Lambda
		coords          = torch.LongTensor(np.where(adj == 1))
		self.B          = [ torch.sparse.FloatTensor(coords, self.B_nonzero, torch.Size([n_counties, n_counties])).to_dense() for tau in range(p) ]
		self.A          = [ torch.sparse.FloatTensor(coords, self.A_nonzero, torch.Size([n_counties, n_counties])).to_dense() for tau in range(p) ]
		self.H          = [ torch.sparse.FloatTensor(coords, self.H_nonzero, torch.Size([n_counties, n_counties])).to_dense() for tau in range(p) ]
		# self.B          = torch.sparse.FloatTensor(coords, self.B_nonzero, torch.Size([n_counties, n_counties])).to_dense()
		# self.A          = torch.sparse.FloatTensor(coords, self.A_nonzero, torch.Size([n_counties, n_counties])).to_dense()
		# self.H          = torch.sparse.FloatTensor(coords, self.H_nonzero, torch.Size([n_counties, n_counties])).to_dense()
		# community mobility
		self.mu         = torch.nn.Parameter(torch.randn(n_mobility, self.p), requires_grad=True)
		self.nu         = torch.nn.Parameter(torch.randn(n_mobility, self.p), requires_grad=True)
		# demographic census
		self.upsilon    = torch.nn.Parameter(torch.randn(n_covariates), requires_grad=True)
		self.zeta       = torch.nn.Parameter(torch.randn(n_covariates), requires_grad=True)
		# exponential decaying factor
		self.theta      = torch.nn.Parameter(torch.ones(1).float(), requires_grad=True)
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
		C = torch.log(C + 1e-2)
		D = torch.log(D + 1e-2)

		T, n_counties = C.shape
		inv           = torch.inverse(self.Sigma)
		# Add up the loss for each week with exponentially decreasing weights
		D_loss = sum([ 0.85 ** self.l2loss(D[i], D_hat[i], inv) for i in range(T) ])
		C_loss = sum([ 0.85 ** self.l2loss(C[i], C_hat[i], inv) for i in range(T) ])

		# Calculate the l1 norm
		l1_norm = torch.norm(torch.stack(self.B, 1), p=1) + torch.norm(torch.stack(self.A, 1), p=1) + torch.norm(torch.stack(self.H, 1), p=1)
		# Calculate the l2 norm
		l2_norm = torch.norm(torch.stack(self.B, 1), p=2) + torch.norm(torch.stack(self.A, 1), p=2) + torch.norm(torch.stack(self.H, 1), p=2)
		print("obj", (0.9 * D_loss + 0.1 * C_loss) / (T * n_counties), "norm", 1e2 * l1_norm + 1e3 * l2_norm)
		return (0.9 * D_loss + 0.1 * C_loss) / (T * n_counties) + 1e2 * l1_norm + 1e3 * l2_norm

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

		mu = self.mu.unsqueeze(-1).repeat(1, 1, n_counties) 
		nu = self.nu.unsqueeze(-1).repeat(1, 1, n_counties)

		c2c = torch.stack([ torch.matmul(ct[tau].clone(), self.B[tau]) for tau in range(self.p) ], dim=1).sum(1)
		c2d = torch.stack([ torch.matmul(ct[tau].clone(), self.H[tau]) for tau in range(self.p) ], dim=1).sum(1)
		d2d = torch.stack([ torch.matmul(dt[tau].clone(), self.A[tau]) for tau in range(self.p) ], dim=1).sum(1)

		# Make predictions
		c_hat = c2c + (mt * mu).sum(0).sum(0) + torch.matmul(self.upsilon, cov)    # [ n_counties ]
		d_hat = c2d + d2d + (mt * nu).sum(0).sum(0) + torch.matmul(self.zeta, cov) # [ n_counties ]
		# c_hat = torch.matmul(ct, self.B).sum(0) + (mt * mu).sum(0).sum(0) + torch.matmul(self.upsilon, cov)                                # [ n_counties ]
		# d_hat = torch.matmul(ct, self.H).sum(0) + torch.matmul(dt, self.A).sum(0) + (mt * nu).sum(0).sum(0) + torch.matmul(self.zeta, cov) # [ n_counties ]

		return c_hat, d_hat # [ n_counties ]
