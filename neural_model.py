import numpy as np
from scipy import integrate, linalg, sparse
import pdb
from util.neuron_metadata import *

class NeuralModel:
  """ The C elegans model as described in the paper: "Neural Interactome: Interactive Simulation of a Neuronal System"
  Main reference: https://github.com/shlizee/C-elegans-Neural-Interactome/blob/master/initialize.py
  Usage:
    neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')
    model = NeuralModel(neuron_metadata_collection)
    # You can tweak parameters before running
    model.G_c = 0.2
    # If you want a fixed-seed run.
    # If not specified, will not set seed, and use default python randomizer
    model.seed = 0

    # This will use your tweaked parameters to read files and precompute some values.
    model.init()

    (vs, ss, normalized_vs) = model.run()
    # vs and ss are timeseries for the voltage and synaptic gating state variables
    # normalized_vs are the scaled V dynamics that interactome exports as npy files.
  """

  def __init__(self, neuron_metadata_collection):
    self.neuron_metadata_collection = neuron_metadata_collection
    # Number of neurons
    self.N = 279

    # If seed is not specified, then the initial conditions will use python's default random seed.
    self.seed = None

    # TODO: Set params = I_ext, an array of constants. Most are zeroes.
    self.I_ext = np.zeros(self.N)

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

  def set_current_injection(self, neuron_name, current_nA):
    neuron_id = self.neuron_metadata_collection.get_id_from_name(neuron_name)
    # For reference, 2.3nA results in 23,000 current in interactome's code
    self.I_ext[neuron_id] = current_nA * 10000
  
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
    self.E = np.reshape(-48.0 * is_inhibitory, self.N)

    if self.seed is not None:
      np.random.seed(self.seed)

    self.compute_Vth()

  def compute_Vth(self):
    """
    Vth computation that I wrote from scratch, and that matches my math derivations more.
    Validations:
    - Ran interactome code and printed out their Vth.
      Vth 1 30 100, sum = -18.3342063908 -6.2498993778 -3.61729185518 -1194.25458719
    - compute_Vth_interactome(), which is a near copy paste, produces very similar results.
      Vth 1 30 100, sum = -18.334206390780512 -6.249899377800607 -3.617291855178621 -1194.2545871854843
    - compute_Vth(), this method, produces very similar results as well.
      Vth 1 30 100, sum = -18.334206390780512 -6.249899377800608 -3.6172918551786215 -1194.2545871854845
    - L2 norm of compute_Vth() and compute_Vth_interactome():  3.907985046680551e-14
    """

    b1 = -np.tile(self.Gc * self.Ec, self.N)
    # Interactome rounded to 4, so we followed suit.
    s_eq = round(self.ar / (self.ar + 2 * self.ad), 4)
    b3 = -s_eq * (self.Gs @ self.E)

    m1 = -self.Gc * np.identity(self.N)
    # N x 1, where each item is a row sum
    Gg_row_sums = self.Gg.sum(axis = 1)
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
    self.Vth = np.reshape(linalg.solve(A, b), self.N)

  def compute_Vth_interactome(self):
    """
    Vth computation using interactome's way. I copied pasted the code with very minimal modifications.
    This method is not used, unless you want to exactly replicate interactome's results without rounding differences.
    """
    N = self.N
    # Modified from
    # Gcmat = np.multiply(Gc, np.eye(N)); M1 = -Gcmat
    M1 = -self.Gc * np.identity(N)
    # Modified from
    # b1 = np.multiply(Gc, EcVec)
    b1 = np.tile(self.Gc * self.Ec, N)

    # Modified from
    # Ggap = np.multiply(ggap, Gg)
    # Our code already computes total conductance
    Ggap = self.Gg
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, N, N).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    # Modified from
    # Gs_ij = np.multiply(gsyn, Gs)
    Gs_ij = self.Gs
    s_eq = round((self.ar/(self.ar + 2 * self.ad)), 4)
    sjmat = np.multiply(s_eq, np.ones((N, N)))
    S_eq = np.multiply(s_eq, np.ones((N, 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, N, N).toarray()

    b3 = np.dot(Gs_ij, np.multiply(s_eq, self.E))

    M = M1 + M2 + M3

    # Removed the LU's, because pretty much the same result anyways.

    # (P, LL, UU) = linalg.lu(M)
    # self.Vth = linalg.solve_triangular(UU, linalg.solve_triangular(LL, b, lower = True, check_finite=False), check_finite=False)
    bbb = -b1 - b3
    bb = np.reshape(bbb, N)

    # Modified from
    # InputMask = np.multiply(Iext, InMask)
    InputMask = self.I_ext
    b = np.subtract(bb, InputMask)
    self.Vth = np.reshape(linalg.solve(M, b), self.N)
  
  def dynamic(self, t, state_vars):
    """Dictates the dynamics of the system.
    """
    v_arr, s_arr = np.split(state_vars, 2)

    # I_leak
    I_leak = self.Gc * (v_arr - self.Ec)

    # I_gap = sum_j G_ij (V_i - V_j) = V_i sum_j G_ij - sum_j G_ij V_j
    # The first term is a point-wise multiplication of V and G's squashed column.
    # The second term is matrix multiplication of G and V
    I_gap = self.Gg.sum(axis = 1) * v_arr - self.Gg @ v_arr
    
    # I_syn = sum_j G_ij s_j (V_i - E_j) = V_i sum_j G_ij s_j - sum_j G_j s_j E_j
    # First term is a point-wise multiplication of V and (Matrix mult of G and s)
    # Second term is matrix mult of G and (point mult of s_j and E_j)
    I_syn = v_arr * (self.Gs @ s_arr) - self.Gs @ (s_arr * self.E)

    dV = (-I_leak - I_gap - I_syn + self.I_ext) / self.C
    phi = np.reciprocal(1.0 + np.exp(-self.B*(v_arr - self.Vth)))
    syn_rise = self.ar * phi * (1 - s_arr)
                          
    syn_drop = self.ad * s_arr
    dS = syn_rise - syn_drop
    return np.concatenate((dV, dS))

  def get_normalized_v_arr(self, v_arr):
    """The paper performs analysis on this normalized v_arr.
    """
    vth_adjusted = v_arr - self.Vth
    vmax = 500

    # tanh: Similar to sigmoid, but squashes to between -1 and 1.
    # So, below readjusts value to range from -500 to 500.
    return vmax * np.tanh(np.divide(vth_adjusted, vmax)) 

  def run(self, num_timesteps):
    """Create initial conditions, then simulate dynamics num_timesteps times.
    Args:
      num_timesteps (int): The number of simulation timesteps to run for. Each timestep is dt second long.
    Returns:
      v_mat (N x num_timesteps): Each row is a voltage timeseries of a neuron. 
      s_mat (N x num_timesteps): Each row is an activation timeseries of a neuron's synaptic current.
      v_normalized_mat (N x num_timesteps): vs, but normalized just like the exported dynamics file from Interactome.
    """

    N = self.N
    # Each timestep is 0.01s long.
    dt = 0.01
    # Note that seed is set in init()
    init_vals = 10**(-4)*np.random.normal(0, 0.94, 2*N)

    # The variables to store our complete timeseries data.
    v_mat = []
    s_mat = []
    v_normalized_mat = []

    # TODO: with_jacobian is not needed, remove this.
    dyn = integrate.ode(self.dynamic).set_integrator('vode', atol = 1e-3, min_step = dt*1e-6, method = 'bdf', with_jacobian = True)
    dyn.set_initial_value(init_vals, 0)

    for t in range(num_timesteps):
      dyn.integrate(dyn.t + dt)
      v_arr = dyn.y[:N]
      s_arr = dyn.y[N:]
      v_normalized_arr = self.get_normalized_v_arr(v_arr)
      v_mat.append(v_arr)
      s_mat.append(s_arr)
      v_normalized_mat.append(v_normalized_arr)

    # TODO: Rotate so each entry's row = 1 neuron.
    return v_mat, s_mat, v_normalized_mat

neuron_metadata_collection = NeuronMetadataCollection.load_from_chem_json('data/chem.json')
model = NeuralModel(neuron_metadata_collection)
model.seed = 0
model.set_current_injection("AVBL", 2.3)
model.set_current_injection("AVBR", 2.3)
model.set_current_injection("PLML", 1.4)
model.set_current_injection("PLMR", 1.4)
model.init()
(v_mat, s_mat, v_normalized_mat) = model.run(10)