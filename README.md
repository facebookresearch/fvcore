# fvcore

fvcore is a light-weight core library that provides the most common and essential
functionality shared in various computer vision frameworks developed in FAIR,
such as [Detectron2](https://github.com/facebookresearch/detectron2/),
[PySlowFast](https://github.com/facebookresearch/SlowFast)
[ClassyVision](https://github.com/facebookresearch/ClassyVision).
All components in this library are type-annotated, tested, and benchmarked.

The computer vision team in FAIR is responsible for maintaining this library.

## Features:

Besides some basic utilities, fvcore includes the following features:
* Common pytorch layers, functions and losses in [fvcore.nn](fvcore/nn/).
* A hierarchical per-operator flop counting tool: see [this note for details](./docs/flop_count.md).
* Recursive parameter counting: see [API doc](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.parameter_count).
* Recompute BatchNorm population statistics: see its [API doc](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.update_bn_stats).
* A stateless, scale-invariant hyperparameter scheduler: see its [API doc](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.common.param_scheduler.ParamScheduler).

## Install:

fvcore requires pytorch and python >= 3.6.

Use one of the following ways to install:

### 1. Install from PyPI (updated nightly)
```
pip install -U fvcore
```

### 2. Install from Anaconda Cloud (updated nightly)

```
conda install -c fvcore -c iopath -c conda-forge fvcore
```

### 3. Install latest from GitHub
```
pip install -U 'git+https://github.com/facebookresearch/fvcore'
```

### 4. Install from a local clone
```
git clone https://github.com/facebookresearch/fvcore
pip install -e fvcore
```

## License

This library is released under the [Apache 2.0 license](https://github.com/facebookresearch/fvcore/blob/master/LICENSE).
