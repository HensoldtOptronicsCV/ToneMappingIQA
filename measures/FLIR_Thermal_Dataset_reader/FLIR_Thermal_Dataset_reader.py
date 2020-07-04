# MIT License
#
# Copyright (c) 2020 HENSOLDT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path

image_type_ldr = "jpeg"
image_type_hdr = "tiff"
folder_ldr = "thermal_8_bit"
folder_hdr = "thermal_16_bit"
size_train_set = 8862
size_val_set = 1366
size_video_set = 4224


def get_subset(folder):
    images_ldr = [x for x in (folder / folder_ldr).iterdir() if x.is_file() and (x.name[-4:] == image_type_ldr)]
    images_hdr = [x for x in (folder / folder_hdr).iterdir() if x.is_file() and (x.name[-4:] == image_type_hdr)]

    if len(images_ldr) == 0 or len(images_hdr) == 0:
        raise IndexError("'" + folder.name + "' is an empty directory")

    images_ldr.sort()
    images_hdr.sort()
    return {'ldr': images_ldr, 'hdr': images_hdr}


def flir_thermal_dataset(flir_thermal_dataset_root_path):
    # check for correct FLIR data structure
    flir_thermal_dataset_path = Path(flir_thermal_dataset_root_path)
    for folder in flir_thermal_dataset_path.iterdir():
        if folder.name == "video":
            if folder.is_dir():
                video = get_subset(folder)

        if folder.name == "val":
            if folder.is_dir():
                val = get_subset(folder)

        if folder.name == "train":
            if folder.is_dir():
                train = get_subset(folder)

    return {'video': video, 'val': val, 'train': train}


def main():
    print("test is running")
    data = flir_thermal_dataset("/Data/FLIR_ADAS_IR_ObjDet_Dataset/FLIR_ADAS_1_3")
    assert(len(data['train']['ldr']) == size_train_set and len(data['train']['hdr']) == size_train_set)
    assert(len(data['val']['ldr']) == size_val_set and len(data['val']['hdr']) == size_val_set)
    assert(len(data['video']['ldr']) == size_video_set and len(data['video']['hdr']) == size_video_set)
    print("test succeeded")

if __name__ == "__main__":
    main()
