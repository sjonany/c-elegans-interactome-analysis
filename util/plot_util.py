"""
Plotting utilities.
"""
import pylab as plt
import numpy as np
from scipy.fftpack import fft

def plot_saved_dynamics(neuron_names_to_show, dynamics, neuron_metadata_collection):
  """ Plot timeseries charts for the selected neuron names using data from 'dynamics'
  Usage:
    from util.neuron_metadata import *
    neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')
    dynamics = np.load('data/dynamics_fwd_5s.npy')
    plot_saved_dynamics(['PLML', 'PLMR', 'VB01'], dynamics, neuron_metadata_collection)
  """
  dynamics_snapshot_count = dynamics.shape[0]
  num_neurons_to_show = len(neuron_names_to_show)
  fig, axes = plt.subplots(nrows=num_neurons_to_show, ncols=1,
      figsize=(10, 3 * num_neurons_to_show))
  times = np.arange(0, dynamics_snapshot_count * 0.01 , 0.01)
  for i in range(num_neurons_to_show):
    name = neuron_names_to_show[i]
    id = neuron_metadata_collection.get_id_from_name(name)
    # The neuron ids are already 0-indexed, and is a direct index to dynamics column.
    dynamic = dynamics[:, id]
    
    if num_neurons_to_show == 1:
      ax = axes
    else:
      ax = axes[i]
    ax.plot(times, dynamic)
    ax.set_title(name)
  return fig

def plot_saved_dynamics_collapsed(neuron_names_to_show, dynamics, neuron_metadata_collection):
  """
  See plot_saved_dynamics. The difference is that we just collapse all plots into 1 figure.
  """
  dynamics_snapshot_count = dynamics.shape[0]
  num_neurons_to_show = len(neuron_names_to_show)
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
  times = np.arange(0, dynamics_snapshot_count * 0.01 , 0.01)
  for i in range(num_neurons_to_show):
    name = neuron_names_to_show[i]
    id = neuron_metadata_collection.get_id_from_name(name)
    # The neuron ids are already 0-indexed, and is a direct index to dynamics column.
    dynamic = dynamics[:, id]
    ax.plot(times, dynamic, label = name)
  ax.legend()
  return fig

def plot_principal_component_fft(n_components, projected, t):
  """ Plots FFT for a projected time series of principal components
  Args:
    n_components (int): number of components to include in the plot
    projected (M, N): matrix of projected pc time series, N must be >= n_components
    t (float): 
  Usage:
    pca = PCA(n_components=4)
    projected_X = pca.fit_transform(X)
    plot_principal_component_fft(2, projected_X, 0.01)
  """
  N = len(projected[:,0])
  fig, ax = plt.subplots()

  for i in range(n_components):
    pc_fft = fft(projected[:,i]) 
    pc_fft_freq = np.linspace(0., 1./(2. * t), N//2)

    ax.plot(pc_fft_freq, 2.0/N * np.abs(pc_fft[0:N//2])) 

  ax.set_xlim(0, 5)
  ax.set_xlabel("Frequency (Hz)")
  ax.set_ylabel("Power")
  plt.grid()
  plt.show()