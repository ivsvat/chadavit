"""Inference functions taken from HOWTO notebook"""

import os
import time
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from src.backbones.vit import chada_vit
from src.utils.demo_utils import bytes_to_giga_bytes, print_model
from src.utils.bench_utils import generate_data, collate_images, tokenise, extract_features, flush

from pprint import pformat
import logging
logger = logging.getLogger(__name__)

CONFIG={
    'PATCH_SIZE' : 16,
    'MODEL' : 'dev_chada_vit', 
    'EMBED_DIM' : 192,
    'RETURN_ALL_TOKENS' : False,
    'MAX_NUMBER_CHANNELS' : 10,
    'MIXED_CHANNELS' : False,
    'BATCH_SIZE' : 32,
    'NUM_BATCHES' : 100,
    'LOG_DIR' : '/projects/delight/ivan/chada_vit/logs/dispatched',
    'DEVICE' : 'cuda:1',
    'DTYPE' : torch.float16,
    'XFORMERS' : False,
    'NO_GRAD' : True,
    'ATTN_TYPE' : 'torch.flash',
}


def main(cfg):
    device = cfg['DEVICE']
    free_mem, total_mem = torch.cuda.mem_get_info(device)

    now = datetime.now()
    log_name = now.strftime("%y%m%d_%H:%M:%S")

    logging.basicConfig(filename=os.path.join(cfg['LOG_DIR'], log_name+'.log'), level=logging.INFO)
    logger.info(f"nodename: {os.uname()[1]}")
    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"cuda device: {torch.cuda.get_device_properties(device)}")    
    logger.info(
        f"gpu info \t available: {bytes_to_giga_bytes(free_mem)} (GB), total: {bytes_to_giga_bytes(total_mem)} (GB)"
    )

    logger.info(pformat(cfg))

    model = getattr(chada_vit, cfg['MODEL'])(
        patch_size=cfg['PATCH_SIZE'],
        embed_dim=cfg['EMBED_DIM'],
        return_all_tokens=cfg['RETURN_ALL_TOKENS'],
        max_number_channels=cfg['MAX_NUMBER_CHANNELS'],
        attn_type=cfg['ATTN_TYPE'],
    )
    model.to(device)
    print_model(model, logger=logger, level=20)
    
    times = []
    for i in tqdm(range(cfg['NUM_BATCHES'])):
        datalist =  generate_data(
            num_images=cfg['BATCH_SIZE'], 
            max_num_channels=cfg['MAX_NUMBER_CHANNELS'], 
            random_num_channels=cfg['MIXED_CHANNELS'],
            dtype=cfg['DTYPE'],
            )
        batch = collate_images(datalist)
        t0 = time.time()
        feats = extract_features(
            model, 
            batch, 
            mixed_channels=cfg['MIXED_CHANNELS'], 
            return_all_tokens=False, 
            disable_gradients=cfg['NO_GRAD'],
            dtype=cfg['DTYPE'],
            device=device, 
            )
        t1 = time.time()
        times.append(t1-t0)
        logger.info(f"dt: {t1-t0}")

    
    logger.info(f"feats.shape {feats.shape}")
    logger.info(f"avg forward time/ batch {np.mean(times)} (s)")
    logger.info(f"max_memory_allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated(device))} (GB)")


if __name__=="__main__":
    main(CONFIG)