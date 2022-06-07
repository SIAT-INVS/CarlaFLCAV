## FLResource: Federated Learning Resource Allocation
Federated Learning Resource Allocation (FLResource) is based on Carla Simulation and CVXPY (See https://www.cvxpy.org/). 
FLResource can reproduce results in the following papers:

## Install dependencies
```
pip3 install cvxpy
```

## Quick Start

1. Run Performance Predictor
```
python3 curve_fitting.py
```

2. Run Network Resource Allocation 
```
python3 resource_alloc_multitask.py
```

## Acknowledgement

* [CARLA](https://github.com/carla-simulator)
* [CVXPY](https://www.cvxpy.org/)

## Citation

```tex
@article{FLCAV,
  title={Federated deep learning meets autonomous vehicle perception: Design and verification},
  author={Shuai Wang and Chengyang Li and Qi Hao and Chengzhong Xu and Derrick Wing Kwan Ng and Yonina C. Eldar and H. Vincent Poor},
  journal={arXiv preprint arXiv:2206.01748},
  year={2022},
}
```
