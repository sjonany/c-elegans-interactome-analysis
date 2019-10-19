"""
Plotting utilities.
"""
import pylab as plt
import numpy as np

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