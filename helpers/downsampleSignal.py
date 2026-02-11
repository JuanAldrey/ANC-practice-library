from scipy.signal import resample_poly
import numpy as np

def downsampleSignal(signal, oldFs, newFs):
    if newFs == 0:
        return signal, oldFs
    gcd = np.gcd(oldFs, newFs)
    up = newFs // gcd
    down = oldFs // gcd

    return resample_poly(signal, up=up, down=down), newFs