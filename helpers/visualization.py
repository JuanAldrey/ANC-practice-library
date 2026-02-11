import matplotlib.pyplot as plt

def plotCorrelationResult(signal1, signal2, gccResult, lags):
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(signal1, label="Señal 1")
    plt.plot(signal2, label="Señal 2", alpha=0.7)
    plt.title("Señales temporales")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(lags, gccResult)
    plt.title("Correlación cruzada")
    plt.xlabel("Retardo (muestras)")
    plt.ylabel("Amplitud")
    plt.grid(True)

    plt.tight_layout()
    plt.show()