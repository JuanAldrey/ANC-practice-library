import numpy as np

# For an explanaiton on the Goertzel algorithm: https://www.embedded.com/single-tone-detection-with-the-goertzel-algorithm/

def goertzel(block, toneFrequency, fs):
    N = len(block)
    omega = 2*np.pi*toneFrequency/fs
    coeff = 2*np.cos(omega)

    s_prev = 0.0
    s_prev2 = 0.0

    for n in range(N):
        s = block[n] + coeff*s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    return s_prev - np.exp(-1j*omega)*s_prev2

if __name__ == "__main__":

    # Parámetros
    fs = 1000
    N = 256
    k = 13
    f0 = k*fs/N

    n = np.arange(N)

    # Señal de prueba (seno puro)
    A = 2.0
    phi = 0
    x = A*np.cos(2*np.pi*f0*n/fs + phi)

    # Goertzel
    Xg = goertzel(x, f0, fs)

    # FFT
    Xfft = np.fft.fft(x)
    Xf = Xfft[k]

    print("Goertzel:", Xg)
    print("FFT bin  :", Xf)
    print("Magnitudes:")
    print("Goertzel:", np.abs(Xg))
    print("FFT      :", np.abs(Xf))
    print("Fases:")
    print("Goertzel:", np.angle(Xg))
    print("FFT      :", np.angle(Xf))
