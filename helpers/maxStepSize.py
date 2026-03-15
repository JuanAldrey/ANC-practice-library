import numpy as np
import scipy

def maxStepSize(referenceSignal, plantResponse, nCoefficients):
    filteredReference = scipy.signal.lfilter(plantResponse, [1], referenceSignal)

    threshold = 0.05 * np.max(np.abs(plantResponse))
    plantDelay = np.argmax(np.abs(plantResponse) > threshold)

    return 2/((nCoefficients + plantDelay) * np.mean(filteredReference**2))


if __name__ == "__main__":
    np.random.seed(0)

    N = 20000
    referenceSignal = np.random.randn(N)

    # planta secundaria sintética
    delay = 30
    fir = scipy.signal.firwin(16, 0.3)
    plantResponse = np.concatenate([np.zeros(delay), fir])

    # controlador
    nCoefficients = 32


    # ---------- test ----------

    alpha_max = maxStepSize(referenceSignal, plantResponse, nCoefficients)

    print("Plant delay:", delay)
    print("Controller length:", nCoefficients)
    print("Estimated alpha_max:", alpha_max)
    print("Suggested working alpha:", 0.1 * alpha_max)