import numpy as np
from numpy import linalg

class NeuralModel:
  """ The C elegans model as described in the paper: "Neural Interactome: Interactive Simulation of a Neuronal System"
  Main reference: https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/initialize.py
  Usage:
    model = NeuralModel.create_model('data/chem.json', 'data/gap.json')
    # You can tweak parameters before running
    model.G_c = 0.2
    # If you want a fixed-seed run.
    # If not specified, will not set seed, and use default python randomizer
    model.seed = 0

    # This will use your tweaked parameters to read files and precompute some values.
    model.init()

    (vs, ss, scaled_vs) = model.run()
    # vs and ss are timeseries for the voltage and synaptic gating state variables
    # scaled_vs are the scaled V dynamics that interactome exports as npy files.
  """

  def __init__(self):
    # Number of neurons
    self.N = 279

    # If seed is not specified, then the initial conditions will use python's default random seed.
    self.seed = None

    # TODO: Set params = I_ext, an array of constants. Most are zeroes.
    self.I_ext = np.reshape(np.zeros(self.N), (self.N,1))

    # Cell membrane capacitance. 1.5 pF / 100 = 0.015 arb (arbitrary unit)
    self.C = 0.015

    # Cell membrane conductance, for calculating I_leak. 10pS / 100 = 0.1 arb
    self.Gc = 0.1

    # Leakage potential (mV)
    self.Ec = -35.0

    # ggap = 100pS / 100 = 1 arb
    self.ggap = 1.0

    # gsyn = 100pS / 100 = 1 arb
    self.gsyn = 1.0

    # Synaptic activity
    # Synaptic activity's rise time
    self.ar = 1.0/1.5
    # Synaptic activity's decay time 
    self.ad = 5.0/1.5
    # Width of the sigmoid (mv^-1)
    self.B = 0.125
  
  def init(self):
    # Gap junctions. Total conductivity of gap junctions, where total conductivity = #junctions * ggap.
    # An N x N matrix were Gg[i][j] = from neuron j to i.
    self.Gg = np.load('data/Gg.npy') * self.ggap

    # Synaptic junctions. Total conductivity of synapses, where total conductivity = #junctions * gsyn.
    # An N x N matrix were Gs[i][j] = from neuron j to i.
    self.Gs = np.load('data/Gs.npy') * self.gsyn

    # E. Reversal potential for each neuron, for calculating I_syn
    # 0 mV for synapses going from an excitatory neuron, -48 mV for inhibitory.
    # N-element array that is 1.0 if inhibitory.
    is_inhibitory = np.load('data/emask.npy')
    E = -48.0 * is_inhibitory
    # N x 1 matrix.
    self.E = np.reshape(E, (self.N, 1))

    if self.seed is not None:
      np.random.seed(self.seed)

    self.compute_Vth()

  def compute_Vth(self):
    b1 = -np.tile(self.Gc * self.Ec, (self.N, 1))
    s_eq = self.ar / (self.ar + 2 * self.ad)
    b3 = -s_eq * (self.Gg @ self.E)

    m1 = -self.Gc * np.identity(self.N)
    # N x 1, where each item is a row sum
    Gg_row_sums =  self.Gg.sum(axis = 1)
    # m2 is a diagonal matrix with the negative row sums as the values
    m2 = - np.diag(Gg_row_sums)
    Gs_row_sums =  self.Gs.sum(axis = 1)
    m3 = - s_eq * np.diag(Gs_row_sums)
    # I think paper is missing m4. It shouldn't be the case that A is a completely diagonal matrix.
    # However, interactome github code seems to have done this correctly.
    # Our implementation is mathematically equivalent to the github code.
    m4 = self.Gg

    A = m1 + m2 + m3 + m4
    b = b1 + b3 - self.I_ext
    self.V_th = linalg.solve(A, b)
    # TODO: Reread code, then verify value against interactome w/ 0 injection.

# TODO: Implement run()

# TODO: Delete the testing code below
model = NeuralModel()
model.init()
print(model.V_th)