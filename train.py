#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrow
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from covid19linear import COVID19linear, NonNegativeClipper


#--------------------------------------------------------------------------
#
# DATA MATRICES
#
#--------------------------------------------------------------------------

# confirmed cases and deaths
C = torch.FloatTensor(np.load("mat/ConfirmedCases_1-17.npy")) # [ T, counties ]
D = torch.FloatTensor(np.load("mat/death_1-17.npy"))          # [ T, counties ]
print("Case matrix shape", C.shape)
print("Death matrix shape", D.shape)

# Load covariates
M      = torch.FloatTensor(np.load("mat/mobility_1-17.npy").transpose([2,0,1])) # [ n_mobility, T, counties ]
pop    = np.load("mat/population.npy")
over60 = np.load("mat/over60.npy")
cov    = torch.FloatTensor(np.array([pop, over60]))                             # [ n_covariates, T, counties ]
print("Mobility matrix shape", M.shape)
print("Census matrix shape", cov.shape)

T, n_counties = C.shape
n_mobility    = M.shape[0]
n_covariates  = cov.shape[0]

#--------------------------------------------------------------------------
#
# META DATA AND CONFIGURATIONS
#
#--------------------------------------------------------------------------

# Distance matrix for counties
distance = np.sqrt(np.load("mat/distance.npy")) # [ 3144, 3144 ]
adj      = np.load("mat/adjacency_matrix.npy")  # [ 3144, 3144 ]

# Number of lags
p = 2
# other configurations
I = np.load("mat/counties.npy").tolist()

#---------------------------------------------------------------
#
# Data normalization
#
#---------------------------------------------------------------

# TODO: Uncomment this line when new mobility data is updated.
M   = (M - M.min()) / (M.max() - M.min())
cov = (cov - cov.min()) / (cov.max() - cov.min())

#--------------------------------------------------------------------------
#
# Train a model for each time window
#
#--------------------------------------------------------------------------

torch.manual_seed(0)

print("[%s] start fitting model..." % arrow.now())
model = COVID19linear(
    p=p, adj=adj, dist=distance,
    n_counties=n_counties, n_mobility=n_mobility, n_covariates=n_covariates)

# Use Adam optimizer for optimization with exponential learning rate
optimizer = optim.Adam(model.parameters(), lr=1e+3)
# decayRate = 0.9999
# my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# # Complete training
# for i in range(5000):
#     # clipper = NonNegativeClipper()
#     model.train()
#     optimizer.zero_grad()
#     C_hat, D_hat = model(C=C, D=D, M=M, cov=cov) # [ T - p, n_counties ], [ T - p, n_counties ]
#     loss         = model.loss(C=C[p+1:], D=D[p+1:], C_hat=C_hat[:-1], D_hat=D_hat[:-1])
#     loss.backward(retain_graph=True)
#     optimizer.step()
#     # model.apply(clipper)
#     # my_lr_scheduler.step()
#     if i % 5 == 0:
#         print("iter: %d\tloss: %.5e" % (i, loss.item()))
#         torch.save(model.state_dict(), "fitted_model/model-[%s].pt" % arrow.now().date())
#         # in-sample prediction
#         model.eval()
#         C_hat, D_hat = model(C=C, D=D, M=M, cov=cov)
#         C_hat = torch.stack(C_hat, dim=0).detach().numpy()
#         D_hat = torch.stack(D_hat, dim=0).detach().numpy()
#         np.save("predictions/in-sample-cases.npy", C_hat)
#         np.save("predictions/in-sample-deaths.npy", D_hat)

# # Evaluation
# model.load_state_dict(torch.load("fitted_model/model-[2021-01-28].pt"))
# model.eval()
# C_hat, D_hat = model(C=C, D=D, M=M, cov=cov)
# C_hat = torch.stack(C_hat, dim=0).detach().numpy()
# D_hat = torch.stack(D_hat, dim=0).detach().numpy()

# C_hat = np.load("predictions/in-sample-cases.npy")
# D_hat = np.load("predictions/in-sample-deaths.npy")
# print(D.shape)
# print(D_hat.shape)

# plt.plot(D[2:,:].sum(1), c="blue")
# plt.plot(D_hat.sum(1), c="red")
# plt.show()