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
from src.utils.bench_utils import generate_data, collate_images, tokenise, extract_features

from pprint import pformat
# import logging
# logger = logging.getLogger(__name__)

CONFIG={
    'PATCH_SIZE' : 16,
    'EMBED_DIM' : 192,
    'RETURN_ALL_TOKENS' : False,
    'MAX_NUMBER_CHANNELS' : 10,
    'MIXED_CHANNELS' : False,
    'BATCH_SIZE' : 32,
    'NUM_BATCHES' : 100,
    'LOG_DIR' : '/projects/delight/ivan/chada_vit/logs/',
    'DTYPE' : torch.float32,
    'DEVICE' : 'cuda:1',
    'XFORMERS' : False
}


def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()


def main(cfg):
    device = cfg['DEVICE']

    now = datetime.now()
    log_name = now.strftime("%y%m%d_%H:%M:%S")

    # logging.basicConfig(filename=os.path.join(cfg['LOG_DIR'], log_name+'.log'), level=logging.INFO)

    # logger.info(f"nodename: {os.uname()[1]}")
    # logger.info(f"torch version: {torch.__version__}")
    # logger.info(f"cuda device: {torch.cuda.get_device_properties(device)}")
    
    # free_mem, total_mem = torch.cuda.mem_get_info(device)
    # logger.info(
        # f"gpu info \t available: {bytes_to_giga_bytes(free_mem)} (GB), total: {bytes_to_giga_bytes(total_mem)} (GB)"
    # )

    # logger.info(pformat(cfg))

    model = ChAdaViT(
        patch_size=cfg['PATCH_SIZE'],
        embed_dim=cfg['EMBED_DIM'],
        return_all_tokens=cfg['RETURN_ALL_TOKENS'],
        max_number_channels=cfg['MAX_NUMBER_CHANNELS'],
    )
    model.to(device)
    print_model(model, logger=None)
    
    times = []
    
    datalist =  generate_data(
        num_images=cfg['BATCH_SIZE'], 
        max_num_channels=cfg['MAX_NUMBER_CHANNELS'], 
        random_num_channels=cfg['MIXED_CHANNELS'])
    batch = collate_images(datalist)
    tokens = tokenise(model, batch, device=device)
    print(f"input shape: {batch[0].shape}")
    print(f"tokens.shape: {tokens.shape}")
    feats = extract_features(
        model, batch, 
        mixed_channels=cfg['MIXED_CHANNELS'], 
        return_all_tokens=False, 
        disable_gradients=True,
        device=device, 
        dtype=cfg['DTYPE'], 
        )
    print(f"feats.shape {feats.shape}")


if __name__=="__main__":
    main(CONFIG)