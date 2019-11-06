from scipy.fftpack import fft
import scipy.signal as signal

def get_dominant_periods(projected, t):
    N = len(projected)
    pc_fft = fft(projected[:,0])
    pc_fft_freq = np.linspace(0., 1./(2. * t), N//2)

    widths = np.arange(1,10)
    pc_fft_rs = np.reshape(pc_fft, (N))
    pc_fft_rs = 2.0/N * np.abs(pc_fft_rs[0:N//2])
    peaks = signal.find_peaks_cwt(pc_fft_rs, widths, min_length=1)

    # TODO: invert the frequency to return periods instead of peak indices?
    return peaks

def get_dominant_period(projected, t):
    # TODO: get single dominant period just using max(fft)
    return





