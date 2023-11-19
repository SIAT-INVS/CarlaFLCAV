## FLPCDet: Federated Learning Point CLoud Detection
Federated Learning Point CLoud Detection (FLPCDet) is based on Carla Simulation and OpenPCDet (See README_PCDET.md). 

## Install FLPCDet
* FLPCDet is based on OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
* OpenPCDet is based on SPCONV (https://github.com/traveller59/spconv)
* Test Environment: Ubuntu 20.04 (Python 3.8) and CUDA 11.3 (Nvidia Driver 470)

1. Install dependencies, e.g.,
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install cvxpy
```

2. Install OpenPCDet
```
cd FLPCDet
pip3 install -r requirements.txt
```

3. Install SPCONV
For CUDA 11.3, use
```
pip3 install spconv-cu113
```

4. Install OPEN3D
```
pip3 install open3d
```

5. Build 
```
python setup.py develop
```

## Quick Start

1. Prepare dataset

Download our example dataset first and extract to `data` folder: [FLPCDet.tar.gz](https://hkustgz-my.sharepoint.com/:f:/g/personal/cli386_connect_hkust-gz_edu_cn/Em2lXlioF8dEjxLiAjH20bQBZwZaDTgfYT4QwHpD4sv5YA?e=vaNCl6)

```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/tesla339_dataset.yaml
```

1. Federated learning under resource contraints for SECOND
```
python flcav_second.py -w 4096 -l 4096 --batch_size=16 --epoch=20
```

1.  Model Folders
* Fedmodel: Cloud and edge federated models
* Output: Local models at each autonomous vehicles

1. Testing
```
cd tools;
python test.py --cfg_file cfgs/kitti_models/town05_test/vehicle1232.yaml --ckpt ../fedmodel/cloud/global.pth
```

## Acknowledgement

* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

