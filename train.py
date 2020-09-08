import arrow
import torch
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from covid19linear import COVID19linear


#--------------------------------------------------------------------------
#
# DATA MATRICES
#
#--------------------------------------------------------------------------

# confirmed cases and deaths
C = torch.FloatTensor(np.load("mat/ConfirmedCases.npy")) # [ T, counties ]
D = torch.FloatTensor(np.load("mat/death.npy"))          # [ T, counties ]
print("Case matrix shape", C.shape)
print("Death matrix shape", D.shape)

# Load covariates
M      = torch.FloatTensor(np.load("mat/mobility.npy").transpose([2,0,1])) # [ n_mobility, T, counties ]
pop    = np.load("mat/population.npy")
over60 = np.load("mat/over60.npy")
cov    = torch.FloatTensor(np.array([pop, over60]))                        # [ n_covariates, T, counties ]
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
n_counties = len(I)

#---------------------------------------------------------------
#
# Preprocess the data by applying standardization to covariates
#
#---------------------------------------------------------------

# mobility = mobility[p:]

# mobScaler = StandardScaler()
# covScaler = StandardScaler()

# mobScaler.fit(mobility.reshape((n_weeks, n_counties * 6)))
# covScaler.fit(cov)

# mobility = mobScaler.transform(mobility.reshape((n_weeks, n_counties * 6)))
# mobility = mobility.reshape((n_weeks, n_counties, 6))
# M        = mobility
# cov      = covScaler.transform(cov)

#--------------------------------------------------------------------------
#
# Train a model for each time window
#
#--------------------------------------------------------------------------

print("[%s] start fitting model..." % arrow.now())
model = COVID19linear(
    p=p, adj=adj, dist=distance,
    n_counties=n_counties, n_mobility=n_mobility, n_covariates=n_covariates)

# Use Adam optimizer for optimization with exponential learning rate
optimizer = optim.Adam(model.parameters(), lr = 9e-1)
decayRate = 0.9995
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# Complete training
for i in range(150):
    model.train()
    optimizer.zero_grad()
    C_hat, D_hat = model(C=C, D=D, M=M, cov=cov) # [ T - p, n_counties ], [ T - p, n_counties ]
    loss         = model.loss(C=C[p+1:], D=D[p+1:], C_hat=C_hat[:-1], D_hat=D_hat[:-1])
    loss.backward(retain_graph=True)
    optimizer.step()
    my_lr_scheduler.step()
    if i % 10 == 0:
        print("iter: %d\tloss: %.5e" % (i, loss.item()))
        torch.save(model.state_dict(), "fitted_model/model.pt")
