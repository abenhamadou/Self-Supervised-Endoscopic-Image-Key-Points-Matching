import hydra
import os

def get_cwd():
    original_cwd = hydra.utils.get_original_cwd()
    cwd_dir = os.getcwd()
    os.chdir(original_cwd)
    return cwd_dir