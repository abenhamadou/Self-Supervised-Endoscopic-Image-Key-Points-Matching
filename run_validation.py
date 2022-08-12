import os
import logging
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf

import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import statistics

from src.utils.matcher import feature_match, feature_extraction, evaluate_matches
from src.datasets.test_patche_dataset import PatchDataset
from src.models.arch_factory import model_factory

@hydra.main(version_base=None, config_path="config", config_name="config_evaluation")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    root_folder = os.path.abspath(os.path.split(__file__)[0] + "/")
    logger.info("Working dir: " + root_folder)
    logger.info("Loading parameters from config file")
    validation_data_root = cfg.paths.validation_data
    model_name = cfg.params.model_name
    model_weights_path = cfg.params.weights_path
    image_size = cfg.params.image_size
    patch_size = cfg.params.patch_size
    distance_matching_threshold = cfg.params.distance_matching_threshold
    matching_threshold = cfg.params.matching_threshold

    # load the model to be evaluated
    model = model_factory(model_name, model_weights_path)

    # generate patches from video frames
    # generate testing data
    test_dataset = PatchDataset(validation_data_root, patch_size)

    # back metrics
    precision = []
    matching_score = []

    # go through the patches, frame by frame
    for i, data in enumerate(test_dataset):

        patch_src = data["patch_src"]
        patch_dst = data["patch_dst"]
        gt_keypoint_src = data["keypoint_src"]
        gt_keypoint_dst = data["keypoint_dst"]

        # extract feature vector for all patches
        list_desc_src = feature_extraction(patch_src, model, image_size)
        list_desc_dst = feature_extraction(patch_dst, model, image_size)

        # do matching
        matches, _ = feature_match(list_desc_src, list_desc_dst, matching_threshold)

        # compute evaluation metrics
        nb_false_matching, nb_true_matches, nb_rejected_matches = evaluate_matches(
            gt_keypoint_src, gt_keypoint_dst, matches, distance_matching_threshold
        )
        precision.append(nb_true_matches / (nb_false_matching + nb_true_matches))
        matching_score.append(nb_true_matches / (nb_false_matching + nb_true_matches + nb_rejected_matches))

    logger.info(f"Precision= {statistics.mean(precision):0.4f}")
    logger.info(f"Matching_score= {statistics.mean(matching_score):0.4f}")


if __name__ == "__main__":
    main()
