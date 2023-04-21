import json

from measures.flir_dataset_loader.loader import FlirThermalLoader, ExperimentLoader


class BaseMeasure:
    def __init__(self, config, load_noisy_data=False):
        self.config = config

        self.dataset_data = None
        self.experiment_data = None
        self.norm_value_hdr = config["norm_values"]["hdr"]
        self.norm_value_ldr = config["norm_values"]["ldr"]
        self.eval_sequences = config["experiment"]["eval_sequences"]

        self.prepare_data(load_noisy_data)

    def prepare_data(self, load_noisy_data=False):
        self.dataset_data = FlirThermalLoader(self.config["dataset"]["path"], load_noisy_data).get_dataset()
        self.experiment_data = ExperimentLoader(self.config["experiment"]["path"], load_noisy_data).get_dataset()

    def calculate(self):
        raise NotImplementedError


def main():
    with open("config.json", "r") as cfg_file:
        cfg = json.load(cfg_file)
    BaseMeasure(cfg)


if __name__ == '__main__':
    main()
