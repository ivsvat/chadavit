"""Inference functions taken from HOWTO notebook"""

import os
import gc
import time
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from src.backbones.vit.chada_vit import ChAdaViT
from src.utils.demo_utils import bytes_to_giga_bytes, print_model

from pprint import pformat
import logging
logger = logging.getLogger(__name__)

CONFIG={
    'PATCH_SIZE' : 16,
    'EMBED_DIM' : 192,
    'RETURN_ALL_TOKENS' : False,
    'MAX_NUMBER_CHANNELS' : 10,
    'MIXED_CHANNELS' : False,
    'BATCH_SIZE' : 4,
    'NUM_BATCHES' : 100,
    'LOG_DIR' : '/projects/delight/ivan/chada_vit/logs',
    'DEVICE' : 'cuda:1',
    'XFORMERS' : False,
    'NO_GRAD' : False,
}


def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()


def generate_data(
        num_images: int, 
        random_num_channels=True,
        max_num_channels=10):
    imgs = []
    labels = []
    for i in range(num_images):
        if random_num_channels:
            num_channels = np.random.randint(1, max_num_channels + 1)
        else: num_channels = max_num_channels
        imgs.append(torch.randn(num_channels, 224, 224))
        labels.append(torch.randint(0, 1, (1,)))
    data = list(zip(imgs, labels))
    return data


def collate_images(batch: list):
    """
    Collate a batch of images into a list of channels, a list of labels and a mapping of the number of channels per image.

    Args:
        batch (list): A list of tuples of (img, label)

    Return:
        channels_list (torch.Tensor): A tensor of shape (X*num_channels, 1, height, width)
        labels_list (torch.Tensor): A tensor of shape (batch_size, )
        num_channels_list (list): A list of the number of channels per image
    """
    num_channels_list = []
    channels_list = []
    labels_list = []

    # Iterate over the list of images and extract the channels
    for image, label in batch:
        labels_list.append(label)
        num_channels = image.shape[0]
        num_channels_list.append(num_channels)

        for channel in range(num_channels):
            channel_image = image[channel, :, :].unsqueeze(0)
            channels_list.append(channel_image)

    channels_list = torch.cat(channels_list, dim=0).unsqueeze(
        1
    )  # Shape: (X*num_channels, 1, height, width)

    batched_labels = torch.tensor(labels_list)

    return channels_list, batched_labels, num_channels_list

def extract_features(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mixed_channels: bool,
    return_all_tokens: bool,
    device: str,
    disable_gradients: bool,
):
    """
    Forwards a batch of images X and extracts the features from the backbone.

    Args:
        model (nn.Module): The model to forward the images through.
        X (torch.Tensor): The input tensor of shape (batch_size, 1, height, width).
        list_num_channels (list): A list of the number of channels per image.
        index (int): The index of the image to extract the features from.
        mixed_channels (bool): Whether the images have mixed number of channels or not.
        return_all_tokens (bool): Whether to return all tokens or not.

    Returns:
        feats (Dict): A dictionary containing the extracted features.
    """
    model.eval()

    # Overwrite model "mixed_channels" parameter for evaluation on "normal" datasets with uniform channels size
    model.mixed_channels = mixed_channels

    X, targets, list_num_channels = batch
    X = X.to(device, non_blocking=True)

    if disable_gradients:
        with torch.no_grad():
            feats = model(x=X, index=0, list_num_channels=[list_num_channels])
    else:
        feats = model(x=X, index=0, list_num_channels=[list_num_channels])

    if not mixed_channels:
        if return_all_tokens:
            # Concatenate feature embeddings per image
            chunks = feats.view(sum(list_num_channels), -1, feats.shape[-1])
            chunks = torch.split(chunks, list_num_channels, dim=0)
            # Concatenate the chunks along the batch dimension
            feats = torch.stack(chunks, dim=0)
        # Assuming tensor is of shape (batch_size, num_tokens, backbone_output_dim)
        feats = feats.flatten(start_dim=1)

    return feats


def main(cfg):
    device = cfg['DEVICE']

    now = datetime.now()
    log_name = now.strftime("%y%m%d_%H:%M:%S")

    logging.basicConfig(filename=os.path.join(cfg['LOG_DIR'], log_name+'.log'), level=logging.INFO)

    logger.info(f"nodename: {os.uname()[1]}")
    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"cuda device: {torch.cuda.get_device_properties(device)}")
    
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    logger.info(
        f"gpu info \t available: {bytes_to_giga_bytes(free_mem)} (GB), total: {bytes_to_giga_bytes(total_mem)} (GB)"
    )

    logger.info(pformat(cfg))

    model = ChAdaViT(
        patch_size=cfg['PATCH_SIZE'],
        embed_dim=cfg['EMBED_DIM'],
        return_all_tokens=cfg['RETURN_ALL_TOKENS'],
        max_number_channels=cfg['MAX_NUMBER_CHANNELS'],
    )
    model.to(device)
    print_model(model, logger=logger, level=20)
    
    times = []
    for i in tqdm(range(cfg['NUM_BATCHES'])):
        datalist =  generate_data(
            num_images=cfg['BATCH_SIZE'], 
            max_num_channels=cfg['MAX_NUMBER_CHANNELS'], 
            random_num_channels=cfg['MIXED_CHANNELS'])
        batch = collate_images(datalist)
        t0 = time.time()
        feats = extract_features(
            model, 
            batch, 
            mixed_channels=cfg['MIXED_CHANNELS'], 
            return_all_tokens=False, 
            device=device, disable_gradients=cfg['NO_GRAD'])
        t1 = time.time()
        times.append(t1-t0)
        logger.info(f"dt: {t1-t0}")

    
    logger.info(f"feats.shape {feats.shape}")
    logger.info(f"avg forward time/ batch {np.mean(times)} (s)")
    logger.info(f"max_memory_allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated(device))} (GB)")


if __name__=="__main__":
    main(CONFIG)