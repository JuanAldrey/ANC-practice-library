import numpy as np

def adaptSPM(identificationNoiseBlock, spmErrorSignals, M_g, blocklength, gHat, activatedActuator, mu):
    gHatNew = gHat.copy()

    N_g = blocklength + M_g - 1

    inputPadded = np.concatenate([identificationNoiseBlock, np.zeros(M_g - 1)])
    inputFFT = np.fft.rfft(inputPadded, N_g)

    for nError in range(spmErrorSignals.shape[0]):
        errorPadded = np.concatenate([spmErrorSignals[nError], np.zeros(M_g - 1)])

        crossCorrelation = np.fft.irfft(
            inputFFT.conj() * np.fft.rfft(errorPadded, N_g),
            N_g
        )

        crossCorrelation[M_g:] = 0
        crossCorrelation = crossCorrelation[:M_g] / blocklength

        gHatNew[nError, activatedActuator] += mu * crossCorrelation

    return gHatNew