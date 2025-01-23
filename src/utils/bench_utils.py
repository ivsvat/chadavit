import gc
import torch 
import numpy as np
import torch.utils.benchmark as benchmark


def generate_data(
        num_images: int, 
        random_num_channels=True,
        dtype=torch.float32,
        max_num_channels=10):
    imgs = []
    labels = []
    for i in range(num_images):
        if random_num_channels:
            num_channels = np.random.randint(1, max_num_channels + 1)
        else: num_channels = max_num_channels
        imgs.append(torch.randn(num_channels, 224, 224, dtype=dtype))
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


@torch.no_grad()
def tokenise(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: str,
):
    model.eval()
    X, _, _ = batch
    X = X.to(device)
    return model.token_learner(X)


def extract_features(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mixed_channels: bool,
    return_all_tokens: bool,
    disable_gradients: bool,
    device: str,
    dtype: torch.dtype=torch.float32,
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


def gen_pre_qkv(
    batch_size,
    n_tokens=196*10,
    embed_dim=192,
    dtype=torch.float32,
    nhead=0,
    device='cpu',
    self_attn=False,
    requires_grad=True,
    ):
    r"""Generate random (Q, K, V) triplets

    Args:
        batch_size (int): batch size B to use
        n_tokens (int): sequence length L to use
        embed_dim (int): embedding dimension D to use
        nhead (int): compatibility feature. If >0 outputs will be of shape (B, L, H, D). 
            Otherwise generates with (B, L, D)
    """
    if nhead>0:
        if self_attn:
            q = torch.randn(
                (batch_size, n_tokens, nhead, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
            return q, q, q
        q = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
        return q, k, v
    else:
        if self_attn:
            q = torch.randn(
                (batch_size, n_tokens, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
            return q, q, q

        q = torch.randn(
            (batch_size, n_tokens, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
        k = torch.randn(
            (batch_size, n_tokens, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
        v = torch.randn(
            (batch_size, n_tokens, embed_dim), dtype=dtype, device=device, requires_grad=requires_grad)
        return q, k, v


def gen_qkv(
    batch_size,
    n_tokens=196*10,
    embed_dim=192,
    dtype=torch.float32,
    nhead=1,
    format='xformers',
    device='cpu',
    self_attn=False,
    requires_grad=True,
    ):
    r"""Generate random (Q, K, V) triplets

    Args:
        batch_size (int): batch size B to use
        n_tokens (int): sequence length L to use
        embed_dim (int): embedding dimension D to use
        nhead (int): number of heads in mha
    """
    if nhead>0:
        if format=='xformers':
            q_proj = torch.randn(
                (batch_size, n_tokens, nhead, embed_dim//nhead), 
                dtype=dtype, device=device, requires_grad=requires_grad)
            return q_proj, q_proj, q_proj
        q_proj = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim//nhead), 
            dtype=dtype, device=device, requires_grad=requires_grad)
        k_proj = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim//nhead), 
            dtype=dtype, device=device, requires_grad=requires_grad)
        v_proj = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim//nhead), 
            dtype=dtype, device=device, requires_grad=requires_grad)
        return q_proj, k_proj, v_proj
    elif format=='pt':
        if self_attn:
            q_proj = torch.randn(
                (batch_size, n_tokens, nhead, embed_dim//nhead), 
                dtype=dtype, device=device, requires_grad=requires_grad)
            return q_proj, q_proj, q_proj

        q_proj = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim//nhead), 
            dtype=dtype, device=device, requires_grad=requires_grad)
        k_proj = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim//nhead), 
            dtype=dtype, device=device, requires_grad=requires_grad)
        v_proj = torch.randn(
            (batch_size, n_tokens, nhead, embed_dim//nhead), 
            dtype=dtype, device=device, requires_grad=requires_grad)
        return q_proj, k_proj, v_proj
    else: raise NotImplementedError


def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()


def benchmark_torch_function_in_seconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean