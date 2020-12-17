# fvcore

fvcore is a light-weight core library that provides the most common and essential
functionality shared in various computer vision frameworks developed in FAIR,
such as Detectron2. This library is based on Python 3.6+ and PyTorch. All the components
in this library are type-annotated, tested, and benchmarked.

The computer vision team in FAIR is responsible for maintaining this library.

## Install:

fvcore requires python >= 3.6.

Use one of the following ways to install:

### 1. Install from PyPI (updated nightly)
```
pip install -U fvcore
```

### 2. Install from Anaconda Cloud (updated nightly)

```
conda install -c conda-forge -c iopath -c fvcore fvcore
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
