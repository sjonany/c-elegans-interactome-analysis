from neural_model import NeuralModel
from util.neuron_metadata import NeuronMetadataCollection

neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')
model = NeuralModel(neuron_metadata_collection)

# You can tweak parameters before running

# If you want a fixed-seed run.
# If not specified, will not set seed, and use default python randomizer
model.seed = 0
model.set_current_injection("AVBL", 2.3)
model.set_current_injection("AVBR", 2.3)
model.set_current_injection("PLML", 1.4)
model.set_current_injection("PLMR", 1.4)

# This will use your tweaked parameters to read files and precompute some values.
model.init()
(v_mat, s_mat, v_normalized_mat) = model.run(100)