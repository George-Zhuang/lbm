# 📊 Evaluation
The evaluation of LBM include:
  - [2D point tracking](#2d-point-tracking) on TAP-Vid DAVIS, TAP-Vid Kinetics, and RoboTAP.
  - [2D object tracking](#2d-object-tracking) on TAO, OVT-B, and BFT.

## 🤗 Download checkpoints
Download the **pretrained weights** for demo and evaluation from [huggingface](https://huggingface.co/ZhengGuangze/LBM) and put them in ```checkpoints``` folder. For example:

```bash
huggingface-cli download ZhengGuangze/LBM lbm.pt --local-dir checkpoints
```

📌 Notice: If you want to evaluate your own trained checkpoints, please make sure to convert the checkpoint from DDP mode using: ```python tools/ckpt_converter.py -i output/lbm/checkpoint.pt -o checkpoints/lbm.pt```.

## 2D Point Tracking
1. Download [TAP-Vid DAVIS](https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip), [TAP-Vid Kinetics](https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip), and [RoboTAP](https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip) following [TAP-Vid](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md) benchmark. The expected file structure is as follows:
    ```
    data/
    ├── tapvid_kinetics/
    │   ├── from 0000_of_0010.pkl to 0009_of_0010.pkl (10 files)
    │   ├── tapvid_kinetics.csv
    │   ├── README.md
    │   ├── test.txt
    │   ├── train.txt
    │   └── val.txt
    ├── robotap/
    │   ├── from robotap_split0.pkl to robotap_split4.pkl (5 files)
    |   └── README.md
    └── tapvid_davis/
        ├── tapvid_davis.pkl
        ├── README.md
        └── SOURCES.md
    ```

2. Run evaluation on TAP-Vid benchmark:
    - If you have multiple GPUs, run ```tools/eval_pttrack_mp.py``` for faster evaluation with ```ray```:
      ```bash
      python tools/eval_pttrack_mp.py \
        --eval_dataset davis \
        --tapvid_root data/tapvid_davis/tapvid_davis.pkl \
        --checkpoint_path checkpoints/lbm.pt
      ```
    - Run evaluation on a single GPU:
      ```bash
      python tools/eval_pttrack.py \
        --eval_dataset davis \
        --tapvid_root data/tapvid_davis/tapvid_davis.pkl \
        --checkpoint_path checkpoints/lbm.pt
      ```
    Results will be saved to ```output/lbm_evaluate/davis/eval_results.log``` by default. Please note for 2D point tracking:
    - Choices of ```--eval_dataest```: 
        - ```davis```, 
        - ```kinetics```, 
        - ```robotap```.
    - Examples of corresponding ```--tapvid_root```: 
        - ```data/tapvid_davis/tapvid_davis.pkl```, 
        - ```data/tapvid_kinetics```, 
        - ```data/robotap```.


## 2D Object Tracking
1. Download TAO, OVT-B, and BFT dataset:
    - [TAO val](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) dataset and download the annotations [`tao_val_lvis_v1_classes.json`](https://huggingface.co/dereksiyuanli/masa/resolve/main/tao_val_lvis_v1_classes.json).
    - [OVT-B](https://drive.google.com/drive/folders/1Qfmb6tEF92I2k84NgrkjEbOKnFlsrTVZ) dataset and annotations.
    - [BFT test](https://huggingface.co/datasets/ZhengGuangze/BFT/tree/main) dataset and annotations.
2. Download public detections from [HuggingFace](https://huggingface.co/datasets/ZhengGuangze/OVMOT-detections) or using ```huggingface-cli``` as follows:
    ```bash
    huggingface-cli download anonymous1966/OVMOT-detections --repo-type dataset --local-dir data/public_detections
    ```
    Extract the public detections:
    ```bash
    python data/public_detections/extract.py
    ```
    The expected file structure is as follows:
    ```
    data/
    ├── bft/
    │   ├── annotaions/
    │   └── images/
    ├── ovt-b/
    │   ├── OVT-B/
    │   ├── ovtb_classname.py
    │   └── ovtb_ann_lvis_format.json
    ├── tao/
    │   ├── annotaions/tao_val_lvis_v1_classes.json
    │   └── images/val/
    ├── public_detections/
    │   ├── ovt-b/
    │   ├── ovtao/
    │   └── bft/
    ```
    Please note that the public detections are from [GLEE-Plus](https://github.com/FoundationVision/GLEE), as described in the paper.
3. Install evaluation metrics [TETA](https://github.com/SysCV/tet.git) and [TrackEval](https://github.com/JonathonLuiten/TrackEval):
    ```bash
    pip install git+https://github.com/SysCV/tet.git#subdirectory=teta
    pip install git+https://github.com/anonymous1966/OWTA.git
    ```
4. Run evaluation on TAO:
    ```bash
    python tools/eval_objtrack.py \
      --config_path lbm/configs/objtrack_tao.yaml \
      --checkpoint_path checkpoints/lbm.pt
    ```
    Run evaluation on OVT-B: 
    ```bash
    python tools/eval_objtrack.py \
      --config_path lbm/configs/objtrack_ovt-b.yaml \
      --checkpoint_path checkpoints/lbm.pt
    ```
    Run evaluation on BFT:
    ```bash
    python tools/eval_objtrack.py \
      --config_path lbm/configs/objtrack_bft.yaml \
      --checkpoint_path checkpoints/lbm.pt
    ```