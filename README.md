# Lattice Boltzmann Model
Like **tracking anything**? Want **efficiency**? Try LBM! 
<div align="center">
<img src="assets/demo.gif" width="640">
</div>

[Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity (NeurIPS 2025)](https://george-zhuang.github.io/lbm/)\
Guangze Zheng, Shijie Lin, Haobo Zuo, Si Si, Ming-Shan Wang, Changhong Fu, and Jia Pan\
<a href=""><img src="https://img.shields.io/badge/arXiv-2403.11186-b31b1b" alt="arXiv"></a>
<a href="https://george-zhuang.github.io/lbm/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/ZhengGuangze/LBM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>


The features of LBM inlcude:
- [x] **physics-inspired** by lattice Boltzmann method in fluid dynamics.
- [x] **online** in a frame-by-frame feed-forward manner.
- [x] **real-time** with ~50 FPS on NVIDIA Jetson Orin NX (TensorRT FP16).
- [x] **robust** against **detection failure** for 2d object tracking.

### 📌 News
- ```2025.09``` LBM is accpected by **NeurIPS 2025**.
- ```2025.06``` LBM TensorRT is available. LBM can also track 3D points now by lifting.
- ```2025.04``` LBM is proposed for **online** and **real-time** 2D point tracking and object tracking in dynamic scenes, with only 18M parameter and achieve SOTA performance.

### 📚 Tutorial
- [```Train```](assets/train.md) train LBM from scratch. About 2 days on 4 NVIDIA H800 GPUs.
- [```Eval```](assets/eval.md) eval LBM to reproduce results in the paper.
- [```TensorRT```](assets/tensorrt.md) run LBM on NVIDIA Jetson devices as fast as on RTX 4090! 49 FPS on NVIDIA Jetson Orin NX.

### 🛠️ Prepare

- **Clone** this repo:
    ```bash
    git clone https://github.com/George-Zhuang/lbm.git
    cd lbm
    ```

- **Basic** packages:
    ```bash
    conda create -n lbm python=3.10
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # please check your cuda version
    pip install -r requirements.txt
    ```
- [Optional] **Demo** with Ultralytics for detection in **2D object tracking**:
    ```bash
    pip install ultralytics
    pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git
    ```

- Download the **pretrained weights** for demo and evaluation from [HuggingFace](https://huggingface.co/ZhengGuangze/LBM) and put them in ```checkpoints``` folder. For example:
  ```bash
  huggingface-cli download ZhengGuangze/LBM lbm.pt --local-dir checkpoints
  ```

### ▶️ Demo

- **Click for point tracking**

  Simply run the following:
  ```bash
  python tools/demo_click.py --video_path data/demo.mp4
  ```
  The demo uses ```cv2``` for visualization. Please click a few points to track and press `q` to quit the ```cv2``` window.

- **Object tracking**
  
  This demo corresponds to Section 4.5 in the paper. Simply run the following and ultralytics will download YOLOE and MobileCLIP weights automatically:
  ```bash
  python tools/demo_box.py --video_path data/demo.mp4 --prompt bird
  ```

### 😊 Acknowledgements
Thanks to these great repositories: [Track-On](https://github.com/gorkaydemir/track_on), [CoTracker](https://github.com/facebookresearch/co-tracker), [DELTA](https://github.com/snap-research/DELTA_densetrack3d), [TAPNet](https://github.com/google-deepmind/tapnet), and many other inspiring works in the community.

### 🎫 License
The model is licensed under the [Apache 2.0 license](./LICENSE.txt).