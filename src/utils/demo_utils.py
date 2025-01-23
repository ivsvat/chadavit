"""Print formatting, type conversions etc"""


def bytes_to_giga_bytes(bytes):
  r"""bytes / 1024**3 """
  return bytes / 1024 / 1024 / 1024


def print_model(model, logger=None, **kwargs):
  print(model)
  print(f"Trainable parameters {sum([p.numel() for p in model.parameters() if p.requires_grad]):,}")
  print(f"Total parameters {sum([p.numel() for p in model.parameters()]):,}")
  if logger is not None:
    logger.log(
      msg=str(model), **kwargs
    )
    logger.log(
      msg=f"Trainable parameters {sum([p.numel() for p in model.parameters() if p.requires_grad]):,}", **kwargs
    )
    logger.log(
      msg=f"Total parameters {sum([p.numel() for p in model.parameters()]):,}", **kwargs
    )
