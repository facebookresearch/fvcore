# fvcore

fvcore is a light-weight core library that provides the most common and essential
functionality shared in various computer vision frameworks developed in FAIR,
such as Detectron2. This library is based on Python 3.6+ and PyTorch. All the components
in this library are type-annotated, tested, and benchmarked.

The computer vision team in FAIR is responsible for maintaining this library.

## Install:

fvcore requires python >= 3.6.

Run one of the following commands to install:

### 1. Install from Anaconda Cloud

```
conda install -c fvcore fvcore
```

### 2. Install from PyPI
```
pip install fvcore
```

### 3. Install from GitHub
```
pip install -U 'git+https://github.com/facebookresearch/fvcore'
```

### 4. Install from a local clone
```
git clone https://github.com/facebookresearch/fvcore
cd fvcore && pip install -e .
```

## License

This library is released under the [Apache 2.0 license](https://github.com/facebookresearch/fvcore/blob/master/LICENSE).
