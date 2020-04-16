from pathlib import Path

imageType8Bit = "jpeg"
imageType16Bit = "tiff"
folder8Bit = "thermal_8_bit"
folder16Bit = "thermal_16_bit"
sizeTrainSet = 8862
sizeValSet = 1366
sizeVideoSet = 4224


def getFlirSet(folder):
    lowDynamicRange = [x for x in (folder / folder8Bit).iterdir() if
                            x.is_file() & (x.name[-4:] == imageType8Bit)]
    highDynamicRange = [x for x in (folder / folder16Bit).iterdir() if
                            x.is_file() & (x.name[-4:] == imageType16Bit)]

    if len(lowDynamicRange) == 0 & len(highDynamicRange) == 0:
        raise IndexError("'video' is an empty directory")

    lowDynamicRange.sort()
    highDynamicRange.sort()
    return {'8bit': lowDynamicRange, '16bit': highDynamicRange}

def getAllFlirSets(flirRootPath):
    # check for correct FLIR data structure
    flirPath = Path(flirRootPath)
    for folder in flirPath.iterdir():
        if folder.name == "video":
            if folder.is_dir():
                video = getFlirSet(folder)

        if folder.name == "val":
            if folder.is_dir():
                val = getFlirSet(folder)

        if folder.name == "train":
            if folder.is_dir():
                train = getFlirSet(folder)

    return train, val, video


if __name__ == "__main__":
    print("test is running")
    train, val, video = getAllFlirSets("/Data/FLIR_ADAS_IR_ObjDet_Dataset/FLIR_ADAS_1_3")
    assert (len(train['8bit']) == sizeTrainSet & len(train['16bit']) == sizeTrainSet)
    assert (len(val['8bit']) == sizeValSet & len(val['16bit']) == sizeValSet)
    assert (len(video['8bit']) == sizeVideoSet & len(video['16bit']) == sizeVideoSet)
    print("test succeeded")
