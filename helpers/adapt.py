import numpy as np

#   This block based adaptation of the filter coefficients is explained in Elliot page 150.

def adapt(fx, e, M_w, blocklength, w, mu):
    N_w = blocklength + M_w -1
    fxPadded = np.concatenate([fx, np.zeros(M_w - 1)])
    ePadded = np.concatenate([e, np.zeros(M_w - 1)])

    crossCorrelationResult = np.fft.irfft(np.fft.rfft(fxPadded, N_w).conj() * np.fft.rfft(ePadded, N_w), N_w)
    crossCorrelationResult[M_w:] = 0
    crossCorrelationResult = crossCorrelationResult[:M_w]

    return w + mu * crossCorrelationResult