import os.path as osp
import torch

from .hardnet_model import HardNet128
from .hynet_model import HyNet
from .sosnet_model import SOSNet

ACCEPTED_MODEL_NAMES = ["HyNet", "SOSNet", "HardNet128"]


def model_factory(model_name, model_weights_path):
    assert model_name in ACCEPTED_MODEL_NAMES
    assert osp.exists(model_weights_path)

    state_dict = torch.load(model_weights_path)
    if model_name == "HyNet":
        model = HyNet()
        model.load_state_dict(state_dict)
    elif model_name == "SOSNet":
        model = SOSNet()
        model.load_state_dict(state_dict)
    elif model_name == "HardNet128":
        model = HardNet128()
        model.load_state_dict(state_dict["state_dict"])
    else:
        raise
    return model
