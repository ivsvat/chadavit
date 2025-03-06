import os
from datetime import datetime

import time

import numpy as np
import torch

from src.utils.demo_utils import bytes_to_giga_bytes
from src.utils.bench_utils import gen_qkv, flush, compute_attn, generate_data, count_trainable_params
from src.backbones.vit.dev_attn import build_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from pprint import pformat
import logging
logger = logging.getLogger(__name__)

SUPPORTED_ATTN = ['torch', 'custom_mha']

BACKENDS = {
    'math' : SDPBackend.MATH,
    'efficient' : SDPBackend.EFFICIENT_ATTENTION,
    'cudnn' : SDPBackend.CUDNN_ATTENTION,
    'flash' : SDPBackend.FLASH_ATTENTION,
}

CONFIG={
    'ATTN_TYPE' : 'torch',
    'CUDA_BACKEND' : ['efficient'], # 
    'HEAD_FORMAT' : 'pt', # 'pt' or 'xformers'
    'PATCH_SIZE' : 16,
    'NESTED_TENSORS' : False,
    'EMBED_DIM' : 256,
    'MAX_NUMBER_CHANNELS' : 10,
    'NUM_HEADS' : 8,
    'BATCH_SIZE' : 32,
    'NUM_BATCHES' : 100,
    'LOG_DIR' : '/projects/delight/ivan/chada_vit/logs/mha/layer',
    'DEVICE' : 'cuda:1',
    'DTYPE' : torch.float16,
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

    mha = build_attention(
        embed_dim=cfg['EMBED_DIM'],
        num_heads=cfg['NUM_HEADS'],
        dropout=0.0,
        batch_first=True,
        attn_type=cfg['ATTN_TYPE'],
        dtype=cfg['DTYPE'],
        device=cfg['DEVICE'],
    ).train()
    logger.info(mha)
    logger.info(f"trainable parameters: {count_trainable_params(mha)}")

    for i in tqdm(range(cfg['NUM_BATCHES'])):
        # sample random batches of mha (q,k,v)
        datalist = generate_data(
            num_images=cfg['BATCH_SIZE'],
            max_num_channels=cfg['MAX_NUMBER_CHANNELS'],
            random_num_channels=cfg['NESTED_TENSORS'],
            dtype=cfg['DTYPE'],
            shape=(196, cfg['EMBED_DIM'])
        )
        datalist = list(zip(*datalist))[0]
        chunks = [chunk.view(-1, cfg['EMBED_DIM']) for chunk in datalist]
        if cfg['NESTED_TENSORS']:
            nt = torch.nested.nested_tensor(chunks, layout=torch.jagged).to(cfg['DEVICE'])
        else:
            nt = torch.stack(chunks).to(cfg['DEVICE'])
        logging.info(nt.size())
        t0 = time.time()
        with sdpa_kernel(backends=[BACKENDS[b] for b in cfg['CUDA_BACKEND']]):
            res, _ = mha(nt, nt, nt)
        t1 = time.time()
        # logging.info(res)
        dt = t1-t0
        logger.info(dt)
        times.append(dt)

    logger.info(f"avg forward time/ batch {np.mean(times)} (s)")
    logger.info(f"max_memory_allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated(device))} (GB)")

    flush()


if __name__=='__main__':
    main(CONFIG)