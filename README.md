# Graph Highway Networks in JAX

This is a non-official implementation of the recent GHNets in JAX. The code contains the Graph Highway Networks definition with the three types of node feature infusion. More details in the original paper [Graph Highway Networks](https://arxiv.org/abs/2004.04635).

## Usage
Run 

```python train.py```

to train a model on the Cora dataset.
Cora is the only dataset implemented for now.

## Differences with the paper
I'm using a dropout ratio of 0.1 by default (that's the probability of keeping the neurons). With higher values, the models overfit a lot and the results on the validation and test sets are bad.

## Cite
If you use this implementation in your research, please cite the paper:
```
@article{xin2020graph,
  title={Graph Highway Networks},
  author={Xin, Xin and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M},
  journal={arXiv preprint arXiv:2004.04635},
  year={2020}
}
```
