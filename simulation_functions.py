from util.neuron_metadata import NeuronMetadataCollection
from neural_model import NeuralModel
neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')

def simulate_until_stable(C, Gc, ggap, gsyn):
    """
    Runs a standard simulation of NeuralModel with the given parameter values
    Note this function does not specify seed, it lets the model use a random seed
    Args:
        C - cell membrane capacitance pF / 100 = arb
        Gc - cell membrane conductance pS / 100 = arb
        ggap - global gap junction conductance pS / 100 = arb
        gsyn - global synaptic conductance pS / 100 = arb
    Returns:
        fwd_dynamics (n_timesteps - 300 x n_neurons) - matrix of normalized membrane potential time series for
            all neurons with first 300 timesteps truncated
    """
    # initialize model
    model = NeuralModel(neuron_metadata_collection, C, Gc, ggap, gsyn)
    model.set_current_injection("AVBL", 2.3)
    model.set_current_injection("AVBR", 2.3)
    model.set_current_injection("PLML", 1.4)
    model.set_current_injection("PLMR", 1.4)
    model.init()

    # simulate
    n_timesteps = 2000
    (v_mat, s_mat, v_normalized_mat) = model.run(n_timesteps)

    # assume stability after 300 timesteps
    fwd_dynamics = v_normalized_mat[300:,:]
    return fwd_dynamics
