import os
import pickle
from neural_model import NeuralModel
from sklearn.decomposition import PCA

from util.neuron_metadata import NeuronMetadataCollection
from util.analysis_util import *

neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')

# Make all the param values be exactly formatted for caching.
def adjust_param(val):
    return float('%.3f' % val)

def get_cache_file_path(C, Gc, ggap, gsyn):
    return "cached_notebook_results/cached_simulation_C={0}_Gc={1}_ggap={2}_gsyn={3}".format(C, Gc, ggap, gsyn)

def simulate_until_stable(C, Gc, ggap, gsyn,
                         min_n_timesteps = 1000,
                         max_n_timesteps = 50000,
                         n_timesteps_convergence_check = 1000,
                         max_amplitude_raw_diff = 1,
                         max_amplitude_scaled_diff = 0.05,
                         debug = True):
    """
    See simulate().
    The difference is we can make multiple simulate() calls if the amplitude
    hasn't converged yet.
    See util/analysis_util.py > get_amplitude_convergence.
    """
    
    # Check if cached result exists
    C = adjust_param(C)
    Gc = adjust_param(Gc)
    ggap = adjust_param(ggap)
    gsyn = adjust_param(gsyn)
    
    cache_file = get_cache_file_path(C, Gc, ggap, gsyn)
    
    if os.path.isfile(cache_file):
      print("Loading saved results from pickle file {}".format(cache_file))
      with open(cache_file, "rb") as f:
        return pickle.load(f)

    # If cached result doesn't exist, compute
    n_timesteps = min_n_timesteps
    increment = n_timesteps_convergence_check
    
    all_dynamics = None
    
    while(True):
        all_dynamics = simulate(C, Gc, ggap, gsyn, n_timesteps)
        last_dynamics = all_dynamics[n_timesteps - n_timesteps_convergence_check:,:]
        pca = PCA(n_components = 1)
        projected_X = pca.fit_transform(last_dynamics)
        # Check amplitude convergence of top PC.
        amplitude_diff_raw, amplitude_diff_scaled = get_amplitude_differences(projected_X[:,0])
        amplitude = get_amplitude(projected_X[:,0])
        
        if debug:
            print(("Simulation length {0:.2f}, raw amplitude diff {1:.2f}," +
                  " scaled amplitude diff {2:.2f}, amplitude {2:.2f}")
                  .format(n_timesteps, amplitude_diff_raw, amplitude_diff_scaled, amplitude))
        # Define convergence as when amplitude difference of two continguous time chunks is small enough.
        # Small-amplitude is needed to detect a focus, where the amplitude just keeps getting smaller.
        # Raw diff needed to catch stable focus, where amplitude keeps getting smaller.
        # Normalized diff needed to catch limit cycles with large amplitudes, but model has roundoff errors.
        if (amplitude_diff_raw < max_amplitude_raw_diff
            or amplitude_diff_scaled < max_amplitude_scaled_diff):
            break
        else:
            n_timesteps += increment
            # Binary search for smallest simulation length to reach convergence
            # TODO: A better way is to place this convergence logic in neural_model.py,
            # so we don't restart the simulation.
            increment *= 2

        if n_timesteps > max_n_timesteps:
            print("n_timesteps {} is too high! We give up on convergence :(".format(n_timesteps))
            break
            
    # Update cache
    with open(cache_file, "wb") as f:
        pickle.dump(all_dynamics, f)
    
    return all_dynamics

def simulate(C, Gc, ggap, gsyn,
                          n_timesteps):
    """
    Runs a standard simulation of NeuralModel with the given parameter values
    Note this function does not specify seed, it lets the model use a random seed
    Args:
        C - cell membrane capacitance pF / 100 = arb
        Gc - cell membrane conductance pS / 100 = arb
        ggap - global gap junction conductance pS / 100 = arb
        gsyn - global synaptic conductance pS / 100 = arb
        n_timesteps - how long to run the model for
    Returns:
        fwd_dynamics (n_timesteps - 300 x n_neurons) - matrix of normalized membrane potential time series for
            all neurons.
    """
    # initialize model
    model = NeuralModel(neuron_metadata_collection, C, Gc, ggap, gsyn)
    model.set_current_injection("AVBL", 2.3)
    model.set_current_injection("AVBR", 2.3)
    model.set_current_injection("PLML", 1.4)
    model.set_current_injection("PLMR", 1.4)
    model.init()

    # simulate
    (v_mat, s_mat, v_normalized_mat) = model.run(n_timesteps)
    return v_normalized_mat
