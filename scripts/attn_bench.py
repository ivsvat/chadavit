import os
from datetime import datetime

import torch
from torch import nn
from torch.nn.modules.activation import MultiheadAttention
# import xformers.ops as ops

from src.utils.demo_utils import bytes_to_giga_bytes, print_model, gen_qkv

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
    'LOG_DIR' : '/projects/delight/ivan/chada_vit/logs/mha',
    'DEVICE' : 'cuda:1',
    'DTYPE' : torch.float32,
    'XFORMERS' : False,
    'NO_GRAD' : False,
}


def gen_attn_layer(
    embed_dim,
    num_heads=6,
    dropout=0.0,
    **kwargs,
):
    return MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, **kwargs)


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
    model = gen_attn_layer(
        embed_dim=cfg['EMBEDDING_DIM'], 
        num_heads=cfg['NUM_HEADS'], dropout=0.0)
    logger.info(pformat(cfg))

    return


if __name__=='__main__':
    main(CONFIG)