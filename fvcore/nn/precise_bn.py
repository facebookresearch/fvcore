# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict
# pyre-ignore-all-errors[2,6,16]

import itertools
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import torch
import tqdm
from torch import nn


# pyre-fixme[9]: BN_MODULE_TYPES has type `Tuple[Type[Module]]`; used as
#  `Tuple[Type[BatchNorm1d], Type[BatchNorm2d], Type[BatchNorm3d],
#  Type[SyncBatchNorm]]`.
BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)

logger: logging.Logger = logging.getLogger(__name__)


class _MeanOfBatchVarianceEstimator:
    """
    Note that PyTorch's running_var means "running average of
    bessel-corrected batch variance". (PyTorch's BN normalizes by biased
    variance, but updates EMA by unbiased (bessel-corrected) variance).
    So we estimate population variance by "simple average of bessel-corrected
    batch variance". This is the same as in the BatchNorm paper, Sec 3.1.
    This estimator converges to population variance as long as batch size
    is not too small, and total #samples for PreciseBN is large enough.
    Its convergence is affected by small batch size.

    In this implementation, we also don't distinguish differences in batch size.
    We assume every batch contributes equally to the population statistics.
    """

    def __init__(self, mean_buffer: torch.Tensor, var_buffer: torch.Tensor) -> None:
        self.pop_mean: torch.Tensor = torch.zeros_like(mean_buffer)
        self.pop_var: torch.Tensor = torch.zeros_like(var_buffer)
        self.ind = 0

    def update(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_size: int
    ) -> None:
        self.ind += 1
        self.pop_mean += (batch_mean - self.pop_mean) / self.ind
        self.pop_var += (batch_var - self.pop_var) / self.ind


class _PopulationVarianceEstimator:
    """
    Alternatively, one can estimate population variance by the sample variance
    of all batches combined. This needs to use the batch size of each batch
    in this function to undo the bessel-correction.
    This produces better estimation when each batch is small.
    See Appendix of the paper "Rethinking Batch in BatchNorm" for details.

    In this implementation, we also take into account varying batch sizes.
    A batch of N1 samples with a mean of M1 and a batch of N2 samples with a
    mean of M2 will produce a population mean of (N1M1+N2M2)/(N1+N2) instead
    of (M1+M2)/2.
    """

    def __init__(self, mean_buffer: torch.Tensor, var_buffer: torch.Tensor) -> None:
        self.pop_mean: torch.Tensor = torch.zeros_like(mean_buffer)
        self.pop_square_mean: torch.Tensor = torch.zeros_like(var_buffer)
        self.tot = 0

    def update(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_size: int
    ) -> None:
        self.tot += batch_size
        batch_square_mean = batch_mean.square() + batch_var * (
            (batch_size - 1) / batch_size
        )
        self.pop_mean += (batch_mean - self.pop_mean) * (batch_size / self.tot)
        self.pop_square_mean += (batch_square_mean - self.pop_square_mean) * (
            batch_size / self.tot
        )

    @property
    def pop_var(self) -> torch.Tensor:
        return self.pop_square_mean - self.pop_mean.square()


@torch.no_grad()
def update_bn_stats(
    model: nn.Module,
    data_loader: Iterable[Any],
    num_iters: int = 200,
    progress: Optional[str] = None,
) -> None:
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    See Sec. 3 of the paper "Rethinking Batch in BatchNorm" for details.

    Args:
        model (nn.Module): the model whose bn stats will be recomputed.

            Note that:

            1. This function will not alter the training mode of the given model.
               Users are responsible for setting the layers that needs
               precise-BN to training mode, prior to calling this function.

            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.
        data_loader (iterator): an iterator. Produce data as inputs to the model.
        num_iters (int): number of iterations to compute the stats.
        progress: None or "tqdm". If set, use tqdm to report the progress.
    """
    bn_layers = get_bn_modules(model)

    if len(bn_layers) == 0:
        return
    logger.info(f"Computing precise BN statistics for {len(bn_layers)} BN layers ...")

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    batch_size_per_bn_layer: Dict[nn.Module, int] = {}

    def get_bn_batch_size_hook(
        module: nn.Module, input: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        assert (
            module not in batch_size_per_bn_layer
        ), "Some BN layers are reused. This is not supported and probably not desired."
        x = input[0]
        assert isinstance(
            x, torch.Tensor
        ), f"BN layer should take tensor as input. Got {input}"
        # consider spatial dimensions as batch as well
        batch_size = x.numel() // x.shape[1]
        batch_size_per_bn_layer[module] = batch_size
        return (x,)

    hooks_to_remove = [
        bn.register_forward_pre_hook(get_bn_batch_size_hook) for bn in bn_layers
    ]

    estimators = [
        _PopulationVarianceEstimator(bn.running_mean, bn.running_var)
        for bn in bn_layers
    ]

    ind = -1
    for inputs in tqdm.tqdm(
        itertools.islice(data_loader, num_iters),
        total=num_iters,
        disable=progress != "tqdm",
    ):
        ind += 1
        batch_size_per_bn_layer.clear()
        model(inputs)

        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            batch_size = batch_size_per_bn_layer.get(bn, None)
            if batch_size is None:
                continue  # the layer was unused in this forward
            estimators[i].update(bn.running_mean, bn.running_var, batch_size)
    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = estimators[i].pop_mean
        bn.running_var = estimators[i].pop_var
        bn.momentum = momentum_actual[i]
    for hook in hooks_to_remove:
        hook.remove()


def get_bn_modules(model: nn.Module) -> List[nn.Module]:
    """
    Find all BatchNorm (BN) modules that are in training mode. See
    fvcore.precise_bn.BN_MODULE_TYPES for a list of all modules that are
    included in this search.

    Args:
        model (nn.Module): a model possibly containing BN modules.

    Returns:
        list[nn.Module]: all BN modules in the model.
    """
    # Finds all the bn layers.
    bn_layers = [
        m for m in model.modules() if m.training and isinstance(m, BN_MODULE_TYPES)
    ]
    return bn_layers
