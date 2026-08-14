import numpy as np

def adaptSPMWithoutNoise(actuatorsSignals, spmError, M_g, blocklength, gHat, muS):
    gHatNew = gHat.copy()

    nErrorMics = gHat.shape[0]
    nActuators = gHat.shape[1]

    N_g = blocklength + M_g - 1

    for nActuator in range(nActuators):
        actuatorPadded = np.concatenate([
            actuatorsSignals[nActuator],
            np.zeros(M_g - 1)
        ])

        actuatorFFT = np.fft.rfft(actuatorPadded, N_g)

        for nError in range(nErrorMics):
            errorPadded = np.concatenate([
                spmError[nError],
                np.zeros(M_g - 1)
            ])

            crossCorrelation = np.fft.irfft(
                actuatorFFT.conj() * np.fft.rfft(errorPadded, N_g),
                N_g
            )

            crossCorrelation[M_g:] = 0
            crossCorrelation = crossCorrelation[:M_g] / blocklength

            inputPower = np.mean(actuatorsSignals[nActuator] ** 2) + 1e-8
            crossCorrelation /= inputPower

            gHatNew[nError, nActuator] -= muS * crossCorrelation

    return gHatNew