import sys
from evaluation.FLIR_data_reader.FLIR_reader import getAllFlirSets
from evaluation.exposure_measure.exposure_measure import exposure

dataReader = getAllFlirSets

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IndexError("to few input arguments! Provide absolute Path to FLIR Dataset")

    flirPath = sys.argv[1]
    train, val, video = dataReader(flirPath)
    under_exposure, over_exposure = exposure(video['8bit'])
    print("under exposure: ", under_exposure)
    print("over exposure: ", over_exposure)
