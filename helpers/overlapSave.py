import numpy as np

#   Given a signal x(n) with infinite samples, and h(n) with M samples, the objective is to
#   divide x(n) in blocks and filter each block separately, and then add all blocks together.
#
#   Remember that:
#   Given len(x(n)) = L (a block of x), len(h(n)) = M and len(y(n)) = N, then N = L + M - 1.
#
#   Overlap save method takes the last M - 1 samples from the previous block, and the next L samples for x(n) in
#   order to form a block of N samples. The first M - 1 samples from the first block will be null
#   values since there isnt a previous block.
#
#   Perform afterwards the DFT of each block. h(n) has to be length N, so it will be zero padded at the end.
#   Perform DFT of h(n). Multiply each block DFT by H(k), and then perform IFFT of each result.
#
#   To unite all blocks, in order to not repeat data, discard the first M - 1 samples from each block, and join.

def overlapSaveOld(blocklength, x, h):
    # This version of overlapSave is for offline use since it requires the entire x(n) signal to work
    L = blocklength
    M = len(h)
    N = L + M - 1

    # Zero-pad x to an integer number of blocks
    nBlocks = (len(x) + L - 1) // L
    x = np.concatenate((x, np.zeros(nBlocks * L - len(x))))

    # Initialize buffer
    buffer = np.zeros(M - 1)

    # DFT of h(n)
    hSpectrum = np.fft.rfft(h, N)

    # Initialize output
    outputBlocks = []

    for k in range(nBlocks):
        # Create block
        block = np.concatenate((buffer, x[k*L : (k+1)*L]))

        # Filter block
        blockSpectrum = np.fft.rfft(block, N)
        blockOutput = np.fft.irfft(blockSpectrum * hSpectrum, N)

        # Discard repeated samples
        blockOutput = blockOutput[M - 1:]

        # Concatenate with output
        outputBlocks.append(blockOutput)

        # generate buffer for next loop
        buffer = block[-(M - 1):]
    
    return np.concatenate(outputBlocks)


def overlapSaveBlock(x_block, hSpectrum, buffer, M, N):
    # Block by block implementation for online use

    # Concatenate buffer (last M - 1 samples) and block
    block = np.concatenate([buffer, x_block])

    # Filter block
    blockSpectrum = np.fft.rfft(block, N)
    blockOutput = np.fft.irfft(blockSpectrum * hSpectrum, N)

    # Discard repeated samples
    blockOutput = blockOutput[M - 1:]

    # Update buffer
    bufferNew = block[-(M-1):]

    return blockOutput, bufferNew

def overlapSave(blocklength, x, h):
    L = blocklength
    M = len(h)
    N = L + M - 1

    # Zero-pad x to an integer number of blocks
    nBlocks = (len(x) + L - 1) // L
    x = np.concatenate((x, np.zeros(nBlocks * L - len(x))))

    # Initialize buffer
    buffer = np.zeros(M - 1)

    # DFT of h(n)
    hSpectrum = np.fft.rfft(h, N)

    # Initialize output
    outputBlocks = []

    for k in range(nBlocks):
        block = x[k*L : (k+1)*L]
        blockOutput, buffer = overlapSaveBlock(block, hSpectrum, buffer, M, N)

        # Concatenate with output
        outputBlocks.append(blockOutput)
    
    return np.concatenate(outputBlocks)
