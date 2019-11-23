from neural_model import NeuralModel
from sklearn.decomposition import PCA

from util.neuron_metadata import NeuronMetadataCollection
from util.analysis_util import *

neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')

def simulate_until_stable(C, Gc, ggap, gsyn,
                         min_n_timesteps = 1000,
                         max_n_timesteps = 50000,
                         n_timesteps_to_keep = 1000,
                         max_amplitude_convergence = 0.05,
                         max_amplitude = 0.1,
                         debug = True):
    """
    See simulate().
    The difference is we can make multiple simulate() calls if the amplitude
    hasn't converged yet.
    See util/analysis_util.py > get_amplitude_convergence.
    """
    n_timesteps = min_n_timesteps
    increment = n_timesteps_to_keep
    while(True):
        dynamics = simulate(C, Gc, ggap, gsyn, n_timesteps, n_timesteps_to_keep)

        pca = PCA(n_components = 1)
        projected_X = pca.fit_transform(dynamics)
        # Check amplitude convergence of top PC.
        convergence = get_amplitude_convergence(projected_X[:,0])
        amplitude = get_amplitude(projected_X[:,0])
        
        if debug:
            print("Simulation length {0:.2f}, convergence {1:.2f}, amplitude {2:.2f}"
                  .format(n_timesteps, convergence, amplitude))
        # Define convergence as when amplitude has converged, or amplitude is small enough
        # Small-amplitude is needed to detect a focus, where the amplitude just keeps getting smaller.
        if (convergence < max_amplitude_convergence or amplitude < max_amplitude):
            return dynamics
        else:
            n_timesteps += increment
            # Binary search for smallest simulation length to reach convergence
            # TODO: A better way is to place this convergence logic in neural_model.py,
            # so we don't restart the simulation.
            increment *= 2

        if n_timesteps > max_n_timesteps:
            print("n_timesteps {} is too high! We give up on convergence :(".format(n_timesteps))
            return dynamics

def simulate(C, Gc, ggap, gsyn,
                          n_timesteps,
                          n_timesteps_to_keep):
    """
    Runs a standard simulation of NeuralModel with the given parameter values
    Note this function does not specify seed, it lets the model use a random seed
    Args:
        C - cell membrane capacitance pF / 100 = arb
        Gc - cell membrane conductance pS / 100 = arb
        ggap - global gap junction conductance pS / 100 = arb
        gsyn - global synaptic conductance pS / 100 = arb
        n_timesteps - how long to run the model for
        n_timesteps_to_keep - how many timesteps at the end to return
    Returns:
        fwd_dynamics (n_timesteps - 300 x n_neurons) - matrix of normalized membrane potential time series for
            all neurons with the last n_timesteps_to_keep data points.
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

    fwd_dynamics = v_normalized_mat[n_timesteps - n_timesteps_to_keep:,:]
    return fwd_dynamics
