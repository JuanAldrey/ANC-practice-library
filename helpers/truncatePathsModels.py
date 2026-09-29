import numpy as np

def truncatePathsModels(paths, relativeThreshold=0.1):
    absPaths = np.abs(paths)
    pathPeaks = np.max(absPaths, axis=-1, keepdims=True)
    significantTaps = absPaths >= relativeThreshold * pathPeaks
    tapIndices = np.arange(paths.shape[-1])
    lastSignificantTaps = np.max(np.where(significantTaps, tapIndices, -1), axis=-1)
    pathLengths = np.maximum(lastSignificantTaps + 1, 1)
    maxPathLength = int(np.max(pathLengths))
    tapMask = tapIndices < pathLengths[..., None]
    truncatedPaths = np.where(tapMask, paths, 0.0)
    truncatedPaths = truncatedPaths[..., :maxPathLength]

    return truncatedPaths