import hydra
import os


def get_cwd():
    """
    custom function to get the current hydra output directory while keeping the original working directory
    """
    original_cwd = hydra.utils.get_original_cwd()
    cwd_dir = os.getcwd()
    os.chdir(original_cwd)
    return cwd_dir
