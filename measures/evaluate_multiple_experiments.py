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
import os
from datetime import datetime

import evaluate_single_experiment

def main():
    config_file_dir = os.path.join(os.path.dirname(__file__), "./configs/")
    assert os.path.isdir(config_file_dir)

    # list available config files
    config_files = [pos_json for pos_json in os.listdir(config_file_dir) if pos_json.endswith('.json')]
    print(f"# config files to be processed: {len(config_files)}")

    # process all config files in directory
    for config_file in config_files:
        config_file_path = os.path.join(config_file_dir, config_file)
        assert os.path.isfile(config_file_path)
        config = json.load(open(config_file_path, encoding = 'utf-8', mode='r'))
        evaluate_single_experiment.process_experiment(config)

    print(f"{len(config_files)} experiments processed successfully.")


if __name__ == "__main__":
    main()
