"""
Adapted from milestone_oscillation.ipynb
Will generate two plots:
- results/milestone_oscillation_svdist.png
- results/milestone_oscillation_trajectory_2sv.png
"""

from util.neuron_metadata import *
from util.plot_util import *
import numpy as np
import pandas as pd
from neural_model import NeuralModel
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')

model = NeuralModel(neuron_metadata_collection)
model.seed = 0
model.set_current_injection("AVBL", 2.3)
model.set_current_injection("AVBR", 2.3)
model.set_current_injection("PLML", 1.4)
model.set_current_injection("PLMR", 1.4)
model.init()
(v_mat, s_mat, v_normalized_mat) = model.run(2700)
# The oscillatory dynamic doesn't stabilize until about dt*300 onwards.
# Also, interactome analysis is done after the first 50 timesteps.
fwd_dynamics = v_normalized_mat[300:,:]

# Get all the motor neurons
motor_neurons = []
for id in range(neuron_metadata_collection.get_size()):
    if neuron_metadata_collection.get_metadata(id).neuron_type == NeuronType.MOTOR:
        motor_neurons.append(id)
        
# Worm atlas says 113: "A total of 113 of the 302 C. elegans neurons belong to the motor neuron category"
# But, we get 109 motor neurons. Close enough.
# Let's extract out just the motor neurons' time series.
fwd_motor_dynamics = fwd_dynamics[:,motor_neurons]
num_timesteps = fwd_motor_dynamics.shape[0]
times = np.arange(0, num_timesteps * 0.01 , 0.01)

# Perform mean-centering before PCA
X = fwd_motor_dynamics - fwd_motor_dynamics.mean(axis= 0)

pca = PCA()
projected_X = pca.fit_transform(X)
num_timesteps = fwd_motor_dynamics.shape[0]

# Plot trajectory on two SVD modes over time.
fig = plt.figure(figsize=(21,7))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot(projected_X[:,0], projected_X[:,1], times)

ax.set_title("First two SVD modes", fontsize=15)
ax.set_xlabel('Mode 1', fontsize=10)
ax.set_ylabel('Mode 2', fontsize=10)
ax.set_zlabel('Time (s)', fontsize=10)
fig.tight_layout()
fig.savefig("results/milestone_oscillation_trajectory_2sv.png")

# Plot SV distribution
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
sigma_squared = np.square(pca.singular_values_)
sigma_squared_normed = sigma_squared / sum(sigma_squared)

ax.set_title("SV distributions", fontsize=15)
x_labels = list(range(1, 6))
ax.scatter(x_labels, sigma_squared_normed[:5])
ax.set_xlabel('Mode index', fontsize=10)
ax.set_ylabel('$\sigma^2 / \sum\sigma^2$', fontsize=10)
ax.set_xticks(x_labels)
fig.tight_layout()
fig.savefig("results/milestone_oscillation_svdist.png")
