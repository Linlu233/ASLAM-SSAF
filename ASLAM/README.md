# ASLAM
The repository contains the code for paper "Link Prediction for Attribute and Structure Learning".

## Requirements:
* torch
* numpy
* torch_geometric
* sklearn
* scipy

## Runs:
The ASLAM code is contained in the code dir and the data are downloaded and stored in the data dir.
### Run ASLAM models:
  ```bash
  cd code
  python train.py --runs 10 --lr 0.001 --wd 5e-4 --epochs 401 --bs 32 --patience 50 --dynamic_train False --dynamic_val False --dynamic_test False --dataset disease --val_ratio 0.05 --test_ratio 0.10 --train_percent 1.0 --val_percent 1.0 --test_percent 1.0 --use_new_split False --use_feat False
  ```

## Reference:
If you find the code useful, please cite our paper:
```bash
....
```
