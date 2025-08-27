# 🎯 Train
## 🗂️ Data preparation
You can prepare training data in **one** of the following ways: 

- Download the processed data from [HuggingFace](https://huggingface.co/datasets/ZhengGuangze/LBM/tree/main/train), containing 11k sequences with 24 frames, ~128G.
- Or prepare [Kubric MOVi-F](https://github.com/google-research/kubric) from scratch, following [CoTracker](https://github.com/facebookresearch/co-tracker) and generate 11k sequences with 24 frames and annotations.

Please put the sequences under ```data/kubric_lbm``` directory. The expected file structure is as follows:
```
data/
├── kubric_lbm/
│   ├── 0000
|   |   ├── depths
|   |   |   ├── 000.png
|   |   |   ├── 001.png
|   |   |   ├── ...
|   |   |   └── 023.png
|   |   ├── frames
|   |   |   ├── 000.png
|   |   |   ├── 001.png
|   |   |   ├── ...
|   |   |   └── 023.png
|   |   └── 0000.npy
│   ├── 0001
│   ├── ...
│   └── 11000
```
## 📦 Dependency
In addition to the basic dependencies introduced in the [README](../README.md), we also recommend using the CUDA-compiled multi-scale deformable attention during AMP training to enhance training stability. Install ```mmcv``` following:
```bash
pip install openmim
mim install mmcv
```
**Recommandation** Please modify ```USE_MMCV = False``` to ```USE_MMCV = True``` in [lbm/models/transformer.py](../lbm/models/transformer.py) to enable ```mmcv```. Install and compile ```mmcv``` may take a long time, and the compiled attention is not supported when export to ONNX and TensorRT. Therefore, we remove the dependency of ```mmcv``` for demo, though CUDA-compiled attention may speed up inference.

## 📝 Script
### Train LBM
Train from scratch as follows:
```bash
GPUS_PER_NODE=4 # please change to fit your device
NNODES=1 # please change to fit your device
NODE_RANK=0
MASTER_ADDR=localhost 
MASTER_PORT=6005

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS main.py \
    --config_path "lbm/configs/default.yaml" \
    --movi_f_root "data/kubric_lbm" \ # path to your MOVi-F
    --tapvid_root "data/tapvid_davis/tapvid_davis.pkl" \ # path to your DAVIS
    --eval_dataset "davis" \
    --epoch_num 150 \
    --lr 5e-4 \
    --wd 1e-5 \
    --bs 4 \ # batch size for each device, bs=4 takes ~52GB each GPU
    --model_save_path "lbm"
  ```
Please note that the following default settings:
- ```--amp``` is on for FP16-mixed precision.
- ```--N``` is set 256 for each sequence, where 256 point trajectories are sampled.
- set ```--checkpoint_path``` to ```path/to/your/checkpoint``` if you want to resume training from a checkpoint.

(Optional) Find default settings in  ```lbm/configs/default.yaml```. Change the settings if you want.


## ⚡ Efficient training
Smaller model size of LBM brings higher training efficiency with faster training and less memory usage. Compared to SOTA methods:

| Model | Model size | Reported raining cost |
| ----- | ------------- | ------------------------ |
| LBM | 18M | 2 days with 4 H800 80GB GPUs |
| [DELTA](https://github.com/snap-research/DELTA_densetrack3d) | 59M | 2.5 days with 8 A100 80GB GPUs |
| [TrackOn](https://github.com/gorkaydemir/track_on) | 49M | 32 A100 64GB GPUs |
| [CoTracker3](https://github.com/facebookresearch/co-tracker) | 25M | 32 A100 80GB GPUs |

During training, the maximum allocated memory is 51.7G for LBM with ```batch_size=4``` for each GPU.