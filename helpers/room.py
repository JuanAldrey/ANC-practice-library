import pyroomacoustics as pra
import matplotlib.pyplot as plt

def room(
        fs=48000, 
        roomDimensions=[10, 10, 5], 
        sourcePosition = [1, 5, 2.5], 
        controllerSpeakerPosition = [8, 2.5, 2.5],
        referenceMicPosition = [3, 5, 2.5],
        errorMicPosition = [8, 5, 2.5]
        ):
    # Create a room
    room = pra.ShoeBox(roomDimensions, fs=fs, max_order=0)

    # Sound source
    room.add_source(sourcePosition)

    # Controller speaker
    room.add_source(controllerSpeakerPosition)

    # Reference microphone
    referenceMicPosition = [3, 5, 2.5]

    # Error microphone
    errorMicPosition = [8, 5, 2.5]

    # Add microphones to the room
    room.add_microphone(referenceMicPosition, fs=fs)
    room.add_microphone(errorMicPosition, fs=fs)

    fig, ax = room.plot()
    ax.set_title("Vista superior del recinto")
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.xlabel('X (metros)')
    plt.ylabel('Y (metros)')
    plt.show()

    return room

