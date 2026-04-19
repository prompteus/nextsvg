from typing import Iterator

import torch


def build_optimizer(
    params: Iterator[torch.nn.Parameter],
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    cfg = dict(optimizer_config)
    kind = cfg.pop("kind")
    optim_class = getattr(torch.optim, kind)
    optimizer = optim_class(params, **cfg)
    assert isinstance(optimizer, torch.optim.Optimizer)
    return optimizer
