import matplotlib.pyplot as plt
import numpy as np

def generateRirs(fs, room):
    room.compute_rir()
    rir_r = room.rir[0][0]   # noise → reference mic
    rir_p = room.rir[1][0]   # noise → error mic
    rir_s = room.rir[1][1]   # speaker → error mic

    plt.figure()
    plt.plot(rir_r, label='r(z): noise → reference mic')
    plt.plot(rir_p, label='p(z): noise → error mic')
    plt.plot(rir_s, label='s(z): speaker → error mic')
    plt.legend()
    plt.title("Impulse responses")
    plt.xlabel("Samples")
    plt.show()

    delay_r = np.argmax(np.abs(rir_r))
    delay_p = np.argmax(np.abs(rir_p))
    delay_s = np.argmax(np.abs(rir_s))

    print("Delay R:", delay_r * 1000 / fs, "ms")
    print("Delay P:", delay_p * 1000 / fs, "ms")
    print("Delay S:", delay_s * 1000 / fs, "ms")

    print("In order to comply with causality, computational time must be smaller than:", (delay_p * 1000 / fs) - (delay_r * 1000 / fs) - (delay_s * 1000 / fs), "ms")

    print("Reference path length: ", len(rir_r))
    print("Primary path length: ", len(rir_p))
    print("Secondary path length: ", len(rir_s))

    return rir_r, rir_p, rir_s
