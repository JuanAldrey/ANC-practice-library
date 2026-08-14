import numpy as np

def adaptMultichannel(R, e, M_w, blocklength, w, mu):

    NwAdapt = blocklength + M_w - 1

    rSpectrum = np.fft.rfft(R,n=NwAdapt,axis=-1)
    eSpectrum = np.fft.rfft(e,n=NwAdapt,axis=-1)
    
    crossCorrelationSpectrum = np.einsum('emkf,ef->mkf',np.conj(rSpectrum),eSpectrum)
    crossCorrelation = np.fft.irfft(crossCorrelationSpectrum,n=NwAdapt,axis=-1)
    crossCorrelation = (crossCorrelation[..., :M_w]/ blocklength)

    return w + mu * crossCorrelation