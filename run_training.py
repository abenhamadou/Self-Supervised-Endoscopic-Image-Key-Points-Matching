import os
import logging
from _version import __version__
import hydra
import json
from omegaconf import OmegaConf

import statistics
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.models.hardnet_model import HardNet128
from src.datasets.train_triplet_loader import TripletDataset
from src.losses.triplet_loss_layers import loss_factory


writer = SummaryWriter()
logger = logging.getLogger("Training")
logger.setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="config", config_name="config_train")
def main(cfg):
    logger.info("Version: " + __version__)
    dict_cfg = OmegaConf.to_container(cfg)
    cfg_pprint = json.dumps(dict_cfg, indent=4)
    logger.info(cfg_pprint)

    root_folder = os.path.abspath(os.path.split(__file__)[0] + "/")
    logger.info("Working dir: " + root_folder)
    logger.info("Loading parameters from config file")
    data_dir_list = cfg.paths.train_data
    nb_epoch = cfg.params.nb_epoch
    batch_size = cfg.params.batch_size
    image_size = cfg.params.image_size
    initial_lr = cfg.params.lr
    margin_value = cfg.params.margin_value
    export_checkpoint_path = f"{cfg.params.model}.pth"

    # gathers all epoch losses
    loss_list = []

    # creates dataset and datalaoder
    dataset = TripletDataset(data_dir_list, image_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    model = HardNet128()

    criterion = loss_factory(
        cfg.params.loss_layer,
        batch_size=batch_size,
        margin_value=cfg.params.margin_value,
        loss_weight=cfg.params.loss_weight,
    )

    optimizer = optim.SGD(
        model.parameters(), lr=initial_lr, momentum=cfg.params.momentum, weight_decay=cfg.params.weight_decay
    )

    logger.info("Start Epochs ...")
    for epoch in range(nb_epoch):
        logger.info("___________________________________________________________")

        loss_epoch = []
        dist_positive_epoch = []
        dist_negative_epoch = []

        for (idx, data) in enumerate(train_loader):
            _, inputs = data
            input_var = torch.autograd.Variable(inputs)
            if not (list(input_var.size())[0] == batch_size):
                continue

            inputs_var_batch = input_var.view(batch_size * 3, 1, image_size, image_size)

            # computed output
            output1 = model(inputs_var_batch).to(device)
            output = output1.view(output1.size(0), -1).cpu()
            dist_positive, dist_negative, loss = criterion(output)

            loss_epoch.append(loss.item())
            # writer.add_scalar('Loss', loss.item(),  epoch)
            # TODO add comment to explain the line below
            if len(dist_positive) == 0 and len(dist_negative) == 0:
                continue

            # store some metric values
            loss_epoch.append(loss.item())
            dist_positive_epoch.append(dist_positive[0].item())
            dist_negative_epoch.append(dist_negative[0].item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(statistics.mean(loss_epoch))

        logger.info(f"Epoch= {epoch}  Loss= {statistics.mean(loss_epoch):0.4f}")
        logger.info(f"----> dist_positive: {statistics.mean(dist_positive_epoch):0.4f}")
        logger.info(f"----> dist_negative: {statistics.mean(dist_negative_epoch):0.4f}")

        writer.add_scalar("loss", statistics.mean(loss_epoch), epoch)
    logger.info(f"Save checkpoint to : {export_checkpoint_path}")
    checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss}
    torch.save(checkpoint, os.path.join(os.getcwd(), export_checkpoint_path))


if __name__ == "__main__":
    main()
