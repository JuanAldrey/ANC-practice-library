import numpy as np
import matplotlib.pyplot as plt

def perturbSecondaryPath(s, amp=0.05, phase=0.05):

    S = np.fft.rfft(s)
    
    freqs = np.linspace(0,1,len(S))

    ampPerturb = 1 + amp * np.random.randn(len(S))
    phasePerturb = np.exp(1j * phase * np.random.randn(len(S)))

    S_new = S * ampPerturb * phasePerturb

    s_new = np.fft.irfft(S_new, len(s))

    return s_new