

import os
from pathlib import Path
from .model_persister import ModelPersister
from safetensors.torch import load_file, save_file




class SafetensorModelPersister(ModelPersister):


    def __init__(self, *args, **kwargs):


        super(SafetensorModelPersister, self).__init__(*args, **kwargs)


    def model_persist_exist(self, layer_name, saving_path):

        safetensor_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors')))
        done_marker_exists = os.path.exists(str(saving_path / (layer_name + 'safetensors.done')))

        return safetensor_exists and done_marker_exists

    def persist_model(self, state_dict, layer_name, saving_path):
        save_file(state_dict, saving_path / (layer_name + 'safetensors'))

        print(f"saved as: {saving_path / (layer_name + 'safetensors')}")

        # set done marker
        (saving_path / (layer_name + 'safetensors.done')).touch()


    def load_model(self, layer_name, path):
        layer_state_dict = load_file(Path(path) / (layer_name + ".safetensors"), device="cpu")
        return layer_state_dict