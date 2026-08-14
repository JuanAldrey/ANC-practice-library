import numpy as np

def adaptSHat(
    actuatorsSignals,
    spmError,
    M_g,
    blocklength,
    gHat,
    muS
):
    N_g = blocklength + M_g - 1

    actuatorSpectrum = np.fft.rfft(actuatorsSignals, n=N_g, axis=-1)
    errorSpectrum = np.fft.rfft(spmError, n=N_g, axis=-1)

    crossCorrelationSpectrum = np.einsum('mf,ef->emf', np.conj(actuatorSpectrum), errorSpectrum)
    crossCorrelation = np.fft.irfft(crossCorrelationSpectrum, n=N_g, axis=-1)
    crossCorrelation = crossCorrelation[..., :M_g] / blocklength
    
    inputPower = np.mean(actuatorsSignals ** 2, axis=-1)+ 1e-8
    crossCorrelation /= inputPower[None, :, None]

    return gHat - muS * crossCorrelation

def adaptPHat(
    refSignals,
    modelingErrorSignals,
    nTaps,
    blocklength,
    pHat,
    muP
):
    nFft = blocklength + nTaps - 1

    refSpectrum = np.fft.rfft(refSignals, n=nFft, axis=-1)
    errorSpectrum = np.fft.rfft(modelingErrorSignals, n=nFft, axis=-1)

    crossCorrelationSpectrum = np.einsum('kf,ef->ekf', np.conj(refSpectrum), errorSpectrum)
    crossCorrelation = np.fft.irfft(crossCorrelationSpectrum, n=nFft, axis=-1)
    crossCorrelation = crossCorrelation[..., :nTaps]/ blocklength

    inputPower = np.mean(refSignals ** 2, axis=-1)+ 1e-8
    crossCorrelation /= inputPower[None, :, None]

    return pHat + muP * crossCorrelation