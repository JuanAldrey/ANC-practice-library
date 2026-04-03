import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Sandbox for spectral factorization testing

######### Singlechannel #########
def singlechannelSpectralFactorization(Sxx):
    cepstrum = np.fft.irfft(np.log(Sxx + 1e-12))
    cepstrum_causal = np.zeros_like(cepstrum)
    cepstrum_causal[0] = 0.5 * cepstrum[0]
    cepstrum_causal[1:len(cepstrum)//2] = cepstrum[1:len(cepstrum)//2]
    cepstrum_causal[len(cepstrum)//2] = 0.5 * cepstrum[len(cepstrum)//2]

    F = np.exp(np.fft.rfft(cepstrum_causal))

    return F

blocklength = 128
nBlocks = 500
N = nBlocks * blocklength
v = np.random.randn(N) 
x = lfilter([1], [1, -0.9], v)
alpha = 0.95   
SxxAvg = None
SvvAvg = None
F = None

for k in range(nBlocks):
    xSignal = x[k*blocklength : (k+1)*blocklength]
    xSpectrum = np.fft.rfft(xSignal)
    Sxx = np.abs(xSpectrum)**2
    
    if SxxAvg is None:
        SxxAvg = Sxx.copy()
    else:
        SxxAvg = alpha*SxxAvg + (1-alpha)*Sxx
    
    if(k%20 == 0 and k>19):
        F = singlechannelSpectralFactorization(SxxAvg)
        
    if F is not None:
        V = xSpectrum / (F + 1e-12)
        Svv = np.abs(V)**2

        if SvvAvg is None:
            SvvAvg = Svv.copy()
        else:
            SvvAvg = alpha*SvvAvg + (1-alpha)*Svv

    if k % 100 == 0 and k > 1 and SvvAvg is not None:
        plt.figure()
        plt.plot(10*np.log10(np.abs(F)**2 / np.mean(np.abs(F)**2) + 1e-12), label="|F|^2")
        plt.plot(10*np.log10(SxxAvg / np.mean(SxxAvg) + 1e-12), label="SxxAvg", linestyle='--')
        plt.plot(10*np.log10(SvvAvg / np.mean(SvvAvg) + 1e-12), label="SvvAvg whitened")
        plt.legend()
        plt.show()

        

