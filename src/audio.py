"""Loading / saving WAV files (I/O only, no signal-separation code)."""
import numpy as np
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly

SR = 16000


def load(path, sr=SR):
    rate, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != sr:
        g = gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g)
    x -= x.mean()
    return x / (np.abs(x).max() + 1e-12)


def save(path, x, sr=SR):
    x = x - x.mean()
    x = 0.9 * x / (np.abs(x).max() + 1e-12)      # normalise: ICA output scale is arbitrary
    wavfile.write(path, sr, (x * 32767).astype(np.int16))
