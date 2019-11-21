from scipy.fftpack import fft
import scipy.signal as signal
import numpy as np

def get_fft(y, t):
    N = len(y)
    fft_y = fft(y)
    fft_freq = np.linspace(0., 1./(2. * t), N//2)

    fft_rs = np.reshape(fft_y, (N))
    fft_rs = 2.0/N * np.abs(fft_rs[0:N//2])

    return (fft_rs, fft_freq)

def get_dominant_periods(projected, t):
    (fft_rs, fft_freq) = get_fft(projected[:,0], t)
    widths = np.arange(1,10)
    peaks = signal.find_peaks_cwt(fft_rs, widths, min_length=1)
    freqs = fft_freq[peaks]

    return 1./freqs

def get_dominant_period(projected, dt = 0.01):
	return get_period(projected[:,0], dt)

"""
Compute the oscillation period of a timeseries.

Sample usage:
pca = PCA()
projected_X = pca.fit_transform(X)
dominant_period = get_period(projected_X[:,0])
"""
def get_period(timeseries, dt = 0.01):
	# Only compute the
    (pc_fft, pc_fft_freq) = get_fft(timeseries, dt)
    peak = np.where(pc_fft == np.amax(pc_fft))
    freq_1 = pc_fft_freq[peak]
    per_1 = 1./freq_1[0]

    return per_1

"""
Get the eigenvalues from a fitted PCA.

Sample usage:

n = X.shape[0]
pca = PCA()
eigen_vals = get_eigenvalues_from_pca(pca, n)

# You can confirm that this transformation from singular values is equivalent
# to the eigenvalues if we had done PCA manually through covariance matrix as below.
# See https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8

C = np.dot(X.T, X) / (n-1)
eigen_vals, _ = np.linalg.eig(C)

# You can then confirm these two definitions are equivalent.
"""
def get_eigenvalues_from_pca(pca, n):
	return pca.singular_values_ ** 2.0

"""
Measure of dimensionality by litwin-kumar, et al. 2017.

Sample usage:

n = X.shape[0]
pca = PCA()
eigen_vals = get_eigenvalues_from_pca(pca, n)
get_dimensionality(eigen_vals)
"""
def get_dimensionality(w):
	w_sum = sum(w)
	w_sqr_sum = w_sum * w_sum #calculate the squared sum of the eigen values
	w_sqr = w * w # Pointwise multiplication of w
	w_sum_sqr = sum(w_sqr)
	return 1.0 * w_sqr_sum / w_sum_sqr

"""
We assume the timeseries has already stabilized.
If not, discard the the first few timesteps of your timeseries.
Some papers like [Fletcher 2016 - From global to local...] filters for oscillation based on amplitude.
"""
def get_amplitude(timeseries):
    return max(timeseries) - min(timeseries)

"""
Test the convergence of amplitude calculation by comparing full vs half-time series calculation.
"""
def get_amplitude_convergence(timeseries):
    return get_amplitude(timeseries[int(len(timeseries) / 2.0):]) / get_amplitude(timeseries)
