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
import json
from pathlib import Path


class BaseLoader:
    image_type_ldr = ".jpeg"
    image_type_hdr = ".tiff"
    ldr_image_dir = "thermal_8_bit"
    hdr_image_dir = "thermal_16_bit"
    ldr_noisy_image_dir = "thermal_8_bit_noisy"
    hdr_noisy_image_dir = "thermal_16_bit_noisy"

    def __init__(self, dataset_root_path: str, load_noisy_data: bool = False):
        self.dataset_root_path = Path(dataset_root_path)
        self.load_noisy_data = load_noisy_data

        self.valid_dir_names = ["video", "val", "train"]
        self.train_set_size = 8862
        self.val_set_size = 1366
        self.video_set_size = 4224

    def load_subset(self, dir_):
        raise NotImplementedError()

    def get_dataset(self):
        result = {}
        for dir_ in self.dataset_root_path.iterdir():
            if dir_.name in self.valid_dir_names and dir_.is_dir():
                result[dir_.name] = self.load_subset(dir_)
        return result


class FlirThermalLoader(BaseLoader):
    def load_subset(self, dir_):
        images_ldr = [file for file in sorted((dir_ / FlirThermalLoader.ldr_image_dir).iterdir()) if
                      file.is_file() and file.suffix == FlirThermalLoader.image_type_ldr]
        assert len(images_ldr) == self.__dict__.get(f"{dir_.name}_set_size")

        images_hdr = [file for file in sorted((dir_ / FlirThermalLoader.hdr_image_dir).iterdir())
                      if file.is_file() and file.suffix == FlirThermalLoader.image_type_hdr]
        assert len(images_hdr) == self.__dict__.get(f"{dir_.name}_set_size")

        subset = {"ldr": images_ldr, "hdr": images_hdr}

        if self.load_noisy_data and Path(self.hdr_noisy_image_dir).exists():
            images_hdr_noisy = [file for file in sorted((dir_ / FlirThermalLoader.hdr_noisy_image_dir).iterdir()) if
                                file.is_file() and file.suffix == FlirThermalLoader.image_type_hdr]
            assert len(images_hdr_noisy) == self.__dict__.get(f"{dir_.name}_set_size")
            subset["hdr_noisy"] = images_hdr_noisy

        return subset


class ExperimentLoader(BaseLoader):
    def load_subset(self, dir_):
        images_ldr = [file for file in sorted((dir_ / ExperimentLoader.ldr_image_dir).iterdir()) if
                      file.is_file() and file.suffix == ExperimentLoader.image_type_ldr]
        assert len(images_ldr) == self.__dict__.get(f"{dir_.name}_set_size")

        subset = {"ldr": images_ldr}

        if self.load_noisy_data and Path(self.ldr_noisy_image_dir).exists():
            images_ldr_noisy = [file for file in sorted((dir_ / ExperimentLoader.ldr_noisy_image_dir).iterdir()) if
                                file.is_file() and file.suffix == ExperimentLoader.image_type_ldr]
            assert len(images_ldr_noisy) == self.__dict__.get(f"{dir_.name}_set_size")
            subset["ldr_noisy"] = images_ldr_noisy

        return subset


def main():
    dataset_root = (json.load(open(Path(__file__).parent.parent / "config.json", "r"))).get("dataset").get("path")
    ds_reader = FlirThermalLoader(dataset_root)
    dataset = ds_reader.get_dataset()


if __name__ == '__main__':
    main()
