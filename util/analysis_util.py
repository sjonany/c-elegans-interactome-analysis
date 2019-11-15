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

def get_dominant_period(projected, t):
    (pc_fft, pc_fft_freq) = get_fft(projected[:,0], t)
    peak = np.where(pc_fft == np.amax(pc_fft))
    freq_1 = pc_fft_freq[peak]
    per_1 = 1./freq_1[0]

    return per_1





