import os
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.attention import SDPBackend, sdpa_kernel

# import xformers.ops as ops

from src.utils.demo_utils import bytes_to_giga_bytes, print_model
from src.utils.bench_utils import gen_qkv, flush, benchmark_torch_function_in_seconds

from tqdm import tqdm
from pprint import pformat
import logging
logger = logging.getLogger(__name__)

CONFIG={
    'ATTN_TYPE' : 'vanilla', # or 'xformers'
    'CUDA_BACKEND' : 'efficient',
    'HEAD_FORMAT' : 'pt',
    'PATCH_SIZE' : 16,
    'EMBED_DIM' : 192,
    'MAX_NUMBER_CHANNELS' : 10,
    'NUM_HEADS' : 8,
    'BATCH_SIZE' : 32,
    'NUM_BATCHES' : 100,
    'LOG_DIR' : '/projects/delight/ivan/chada_vit/logs/mha',
    'DEVICE' : 'cuda:1',
    'DTYPE' : torch.float32,
    'NO_GRAD' : True,
    'SELF_ATTN' : True,
    'MODEL_EVAL' : True,
    'QKV_GRAD': False,
}


def gen_attn_layer(
    type,
    embed_dim,
    num_heads=1,
    dropout=0.0,
    **kwargs,
):  
    if type=='vanilla':
        from torch.nn import MultiheadAttention
        return MultiheadAttention(embed_dim=embed_dim, 
                                  num_heads=num_heads, 
                                  dropout=dropout, **kwargs)
    elif type=='flash_sdp':
        from torch.nn import MultiheadAttention
        torch.backends.cuda.enable_flash_sdp(enabled=True)
        return MultiheadAttention(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  dropout=dropout, **kwargs)
    elif type=='mem_efficient_sdp':
        from torch.nn import MultiheadAttention
        torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
        return MultiheadAttention(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  dropout=dropout, **kwargs)
    elif type=='math_sdp':
        from torch.nn import MultiheadAttention
        torch.backends.cuda.enable_math_sdp(enabled=True)
        return MultiheadAttention(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  dropout=dropout, **kwargs)
    else:
        raise NotImplementedError


def compute_attn(
    q, k, v,
    type,
    backend,
):
    backends_dict = {
        'math' : SDPBackend.MATH,
        'flash' : SDPBackend.FLASH_ATTENTION,
        'efficient' : SDPBackend.EFFICIENT_ATTENTION,
    }
    if type=='vanilla':
        with sdpa_kernel(backends_dict[backend]):
            dt =  benchmark_torch_function_in_seconds(F.scaled_dot_product_attention, q, k, v)
            return dt
    else:
        raise NotImplementedError


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

    # model = gen_attn_layer(
    #     type=cfg['ATTN_TYPE'],
    #     embed_dim=cfg['EMBED_DIM'], 
    #     num_heads=cfg['NUM_HEADS'], 
    #     dropout=0.0,
    #     batch_first=True,
    #     )
    # model.to(device)
    # if cfg['MODEL_EVAL']:
    #     model.eval()
    # print_model(model, logger, level=20)

    times = []
    # for i in tqdm(range(cfg['NUM_BATCHES'])):
    #     q, k, v = gen_qkv(
    #         format='xformers',
    #         batch_size=cfg['BATCH_SIZE'],
    #         n_tokens=10*196,
    #         embed_dim=cfg['EMBED_DIM'],
    #         dtype=cfg['DTYPE'],
    #         device=device,
    #         nhead=0,
    #         self_attn=cfg['SELF_ATTN'],
    #         requires_grad=cfg['QKV_GRAD']
    #     )

    #     t0 = time.time()
    #     if cfg['NO_GRAD']:
    #         with torch.no_grad():
    #             feats = list(model(q, k, v))[0] # handle variable output len
    #     else:
    #         feats = list(model(q, k, v))[0]
    #     t1 = time.time()
    #     # logger.info(t1-t0)
    #     times.append(t1-t0)
    for i in tqdm(range(cfg['NUM_BATCHES'])):
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
 
        # logger.info(t1-t0)
        times.append(dt)

    # logger.info(f"feats.shape {feats.shape}")
    logger.info(f"avg forward time/ batch {np.mean(times)} (s)")
    logger.info(f"max_memory_allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated(device))} (GB)")

    flush()
    return


if __name__=='__main__':
    main(CONFIG)