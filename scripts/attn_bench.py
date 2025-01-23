import os
from datetime import datetime

import numpy as np
import torch

from src.utils.demo_utils import bytes_to_giga_bytes
from src.utils.bench_utils import gen_qkv, flush, compute_attn

from tqdm import tqdm
from pprint import pformat
import logging
logger = logging.getLogger(__name__)

CONFIG={
    'ATTN_TYPE' : 'xformers', # or 'xformers'
    'CUDA_BACKEND' : 'math',
    'HEAD_FORMAT' : 'xformers',
    'PATCH_SIZE' : 16,
    'EMBED_DIM' : 192,
    'MAX_NUMBER_CHANNELS' : 10,
    'NUM_HEADS' : 8,
    'BATCH_SIZE' : 32,
    'NUM_BATCHES' : 100,
    'LOG_DIR' : '/projects/delight/ivan/chada_vit/logs/mha',
    'DEVICE' : 'cuda:1',
    'DTYPE' : torch.float32,
    # 'NO_GRAD' : True,
    'SELF_ATTN' : True,
    # 'MODEL_EVAL' : True,
    'QKV_GRAD': True,
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

    times = []

    for i in tqdm(range(cfg['NUM_BATCHES'])):
        # sample random batches of mha (q,k,v)
        q, k, v = gen_qkv(
            format=cfg['HEAD_FORMAT'],
            batch_size=cfg['BATCH_SIZE'],
            n_tokens=10*196,
            embed_dim=cfg['EMBED_DIM'],
            dtype=cfg['DTYPE'],
            device=device,
            nhead=cfg['NUM_HEADS'],
            self_attn=cfg['SELF_ATTN'],
            requires_grad=cfg['QKV_GRAD']
        )

        dt = compute_attn(
            q, k, v, 
            type=cfg['ATTN_TYPE'],
            backend=cfg['CUDA_BACKEND']
            )
 
        logger.info(dt)
        times.append(dt)

    logger.info(f"input.shape {q.shape}")
    logger.info(f"avg forward time/ batch {np.mean(times)} (s)")
    logger.info(f"max_memory_allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated(device))} (GB)")

    flush()
    return


if __name__=='__main__':
    main(CONFIG)