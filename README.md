# highway-env

## Parallel Parking task

[![build](https://github.com/eleurent/highway-env/workflows/build/badge.svg)](https://github.com/eleurent/highway-env/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/Farama-Foundation/HighwayEnv/actions/workflows/build-docs-dev.yml/badge.svg)](https://farama-foundation.github.io/HighwayEnv/)
[![Downloads](https://img.shields.io/pypi/dm/highway-env)](https://pypi.org/project/highway-env/)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/63847d9328f64fce9c137b03fcafcc27)](https://app.codacy.com/manual/eleurent/highway-env?utm_source=github.com&utm_medium=referral&utm_content=eleurent/highway-env&utm_campaign=Badge_Grade_Dashboard)
[![GitHub contributors](https://img.shields.io/github/contributors/eleurent/highway-env)](https://github.com/eleurent/highway-env/graphs/contributors)

### DreamerV3 setup

First you need to install the necessay packages. you reqiure cuda12 or higher. for this project im using cuda 13 with a RTX5070 gpu.

also need cuda tool kit installed and running

ps: this project was tested in ubuntu

also, you need conda installed.

#### install deps

First, we need to create a new venv wiith conda using the existing yml configurations i have provided in the folder

```bash
    conda env create -f dreamer_env_config.yml -n dreamer
    conda activate dreamer
```

Next we install dreamerV3 deps

```bash
    pip install -U -e ./dreamerV3/requirements.txt
```

then we install gymnasium

```bash
    pip install gymnasium
    pip install -U "jax[cuda13]"
```

now, install and register the parallel parking environment:

```bash
    pip install -e . #run this command in the root folder where we have setup.py
```

DreamerV3 should now be training 🚀🚀🚀

### SAC Setup

First, create venv:

```bash
    conda create env -f sac_env_config.yml -n sac
    conda activate sac
```

Next, we install HighwayEnv deps and stable_baseline

```bash
    python3 -m pip install -U pygame --user
    pip install -e .
    pip install gymnasium

```

`pip install -e .` registers the parallel-parking-environement

this installs most deps needed.

Next, run the training script:

```bash
    ./train_sac.sh
```

SAC should be training now 🚀🚀🚀

## Notice:

The code has a number of environments, to run the latest submitted in the dist, copy the contents of `./archive/parallel_parking_latest.py` into `./highway_env/envs/parallel_parking_env.py` and retrain.
