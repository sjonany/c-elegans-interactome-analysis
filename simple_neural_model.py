import numpy as np
from scipy import integrate, linalg, sparse, signal
import pdb
from util.plot_util import *

class SimpleNeuralModel:

    def __init__(self):
        self.seed = None
        if self.seed is not None:
            np.random.seed(self.seed)

        # number of neurons to create
        self.N = 3

        self.I_ext = np.array([(lambda t: 0) for i in range(self.N)])

        # cell membrane capacitance (LHS)
        self.C = 0.015

        # cell membrane conductance
        self.Gc = 0.1

        # leak potential
        self.Ec = -35.0

        # ggap
        self.ggap = 1.0

        # gsyn
        self.gsyn = 1.0

        # synaptic rise time
        self.ar = 1.0 / 1.5

        # synaptic decay time
        self.ad = 5.0 / 1.5

        # sigmoid width
        self.B = 0.125

    def build_network(self):
        # Gg[i, j] = gap junction conductivity from j to i (range 0 to 23)
        self.Gg = np.array([
            [0., 0., 0.],
            [0., 0., 1.],
            [1., 0., 0.]])
        r = np.random.randint(1, 23, (self.N, self.N))
        self.Gg = np.multiply(self.Gg, r)

        # Gs[i, j] = synaptic conductivity to i from j (range 0 to 35)
        self.Gs = np.array([
            [0., 1., 1.],
            [0., 0., 0.],
            [0., 1., 0.]])
        r = np.random.randint(1, 35, (self.N, self.N))
        self.Gs = np.multiply(self.Gs, r)

        # some neurons may be inhibitory
        inhibitory_arr = np.array([0] * self.N)
        self.E = np.reshape(-48.0 * inhibitory_arr, self.N)

        self.compute_Vth()

    def compute_Vth(self):
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
        b = b1 + b3 - [fn(0) for fn in self.I_ext]
        b = np.reshape(np.array(b), (self.N))
        self.Vth = np.reshape(linalg.solve(A, b), self.N)

    def set_current_injection_function(self, neuron_index, current_fn):
        self.I_ext[neuron_index] = current_fn

    def set_current_injection(self, neuron_index, current_nA):
        self.I_ext[neuron_index] = current_nA * 10000
        x = 1

    def dynamics(self, t, state):
        v_arr, s_arr = np.split(state, 2)

        # compute each component of membrane current separately
        I_leak = self.Gc * (v_arr - self.Ec)
        
        # gap current
        gap_conductance_in = self.Gg.sum(axis=1)
        I_gap_in = gap_conductance_in * v_arr # conductance_in * v_in
        I_gap_out = self.Gg @ v_arr
        I_gap = I_gap_in - I_gap_out

        # synaptic current
        I_syn = v_arr * (self.Gs @ s_arr) - self.Gs @ (s_arr * self.E)
        
        # external current (where applicable)
        I_ext = [fn(t) for fn in self.I_ext]

        # dV
        dV = (-I_leak - I_gap - I_syn + I_ext) / self.C

        # dS
        phi = np.reciprocal(1.0 + np.exp(-self.B*(v_arr - self.Vth)))
        syn_rise = self.ar * phi * (1 - s_arr)                    
        syn_drop = self.ad * s_arr
        dS = syn_rise - syn_drop

        return np.concatenate((dV, dS))

    def run(self, num_timesteps):
        """Create initial conditions, then simulate dynamics num_timesteps times.
        Args:
        num_timesteps (int): The number of simulation timesteps to run for. Each timestep is dt = 0.01 second long.
        Returns:
        v_mat (num_timesteps x N): Each column is a voltage timeseries of a neuron. 
        s_mat (num_timesteps x N): Each column is an activation timeseries of a neuron's synaptic current.
        v_normalized_mat (num_timesteps x N): v_mat, but normalized just like the exported dynamics file from Interactome.
            Note that interactome's exported data starts from timestep 50 onwards, so make sure to truncate first 50 if you
            want to compare. See milestone_responsive.ipynb for how to do this.
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

        dyn = integrate.ode(self.dynamics).set_integrator('vode', atol = 1e-3, min_step = dt*1e-6, method = 'bdf', with_jacobian = True)
        dyn.set_initial_value(init_vals, 0)

        for t in range(num_timesteps):
            dyn.integrate(dyn.t + dt)
            v_arr = dyn.y[:N]
            s_arr = dyn.y[N:]
            v_normalized_arr = self.get_normalized_v_arr(v_arr)
            v_mat.append(v_arr)
            s_mat.append(s_arr)
            v_normalized_mat.append(v_normalized_arr)

        return np.array(v_mat), np.array(s_mat), np.array(v_normalized_mat)

    def get_normalized_v_arr(self, v_arr):
        """The paper performs analysis on this normalized v_arr.
        """
        vth_adjusted = v_arr - self.Vth
        vmax = 500

        # tanh: Similar to sigmoid, but squashes to between -1 and 1.
        # So, below readjusts value to range from -500 to 500.
        return vmax * np.tanh(np.divide(vth_adjusted, vmax)) 


# initialize
model = SimpleNeuralModel()
model.build_network()
l = lambda t: 2.3 * 10000 * (1 + signal.square(2 * np.pi * 50 * t))
model.set_current_injection_function(1, l)

# simulate
(v_mat, s_mat, v_normalized_mat) = model.run(1000)

# plot
plot_saved_dynamics_by_id([0,1,2], v_normalized_mat)

# analyze (goal: get period of any oscillations)