# Is High Variance Unavoidable in RL? A Case Study in Continuous Control -- ICLR 2022

### Johan Bjorck, Carla Gomes, Kilian Weinberger

[[paper](https://openreview.net/forum?id=9xhgmsNVHu)]

## Overview

Reinforcement learning (RL) experiments have notoriously high variance. We demonstrate that one can optimize directly for lower variance, without hurting average performance. Specifically, using a few simple methods, we decrease the variance by a factor ~3x (over 21 DMC tasks) for the competitive actor-critic agent [DRQv2](https://github.com/facebookresearch/drqv2) without decreasing the average reward. This repo contains minimal modifications on top of the DRQv2 code base to reproduce our results.



## Installation

Simply use the conda environment:

```sh
conda env create -f conda_env.yml
conda activate drqv2
```


## Usage


1. To run without any tricks enabled, simply run:

```
python train.py task=cheetah_run
```

2. To use all all tricks, run:

```
python train.py task=cheetah_run  agent.pnorm_critic=True agent.pnorm_actor=True agent.asymmetric_clip=True agent.action_penalty=0.0001 agent.cpc_until=10000
```

Note that the number of frames where CPC is used and the parameter for the action penalty are explicitly set. 


3. Our proposed methods can be independently toggled with

    | Method                      | Flags                                     |
    |-----------------------------|-------------------------------------------|
    | pnorm for critic            | `agent.pnorm_critic=True`                 |
    | pnorm for actor             | `agent.pnorm_actor`                       |
    | assymetric clip             | `agent.asymmetric_clip=True`              |
    | action penalty              | `agent.action_penalty=0.0001`             |
    | contrastive learning        | `agent.cpc_until=10000`                   |






## Citation

For citation, please use:


```
@article{bjorck2021high,
  title={Is High Variance Unavoidable in RL? A Case Study in Continuous Control},
  author={Bjorck, Johan and Gomes, Carla P and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:2110.11222},
  year={2021}
}
```

and for DRQv2:

```
@article{yarats2021drqv2,
  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},
  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
  journal={arXiv preprint arXiv:2107.09645},
  year={2021}
}
```



## Acknowledgements

Our experiments are built on top of the open-sourced code for [DRQv2](https://github.com/facebookresearch/drqv2).




