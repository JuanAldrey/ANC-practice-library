import numpy as np

def addNoise(signal, snrDb=20):
    signal = np.asarray(signal)
    signalPower = np.mean(signal**2)
    snrLinear = 10 ** (snrDb / 10)
    noisePower = signalPower / snrLinear
    noise = np.random.randn(*signal.shape)
    noise *= np.sqrt(noisePower / np.mean(noise**2))

    return signal + noise