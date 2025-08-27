# 🚀 TensorRT

LBM supports [TensorRT](https://github.com/NVIDIA/TensorRT) acceleration with with ​**FP16 quantization**. Tested on:

- [x] NVIDIA RTX 4090 on Windows 11 with: 
    - torch 2.6.0+cu124
    - onnx 1.18.0
    - onnxruntime-gpu 1.22.0
    - TensorRT 10.11.0.33

- [x] NVIDIA Jetson Orin NX with 
    - [JetPack 6.2](https://developer.nvidia.com/embedded/jetpack)
    - torch 2.6.0
    - onnx 1.18.0
    - onnxruntime-gpu 1.22.0
    - TensorRT 10.3.0

## 📦 Dependency
Please make sure you have following packages:
- On PC or server:
    - Create a conda environment and install packages:
        ```bash
        conda create -n lbm python=3.10
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        pip install -r requirements.txt 
        pip install onnx onnxruntime-gpu tensorrt pycuda
        ```
- On NVIDIA Jetson devices:
    - Create a conda environment:
        ```bash
        conda create -n lbm python=3.10
        ```
    - Download ```torch torchvision onnxruntime-gpu opencv-python pycuda``` from [Jetson AI Lab](https://pypi.jetson-ai-lab.dev/jp6/cu126) with corresponding JetPack and CUDA version. 
    - Install locally.
    - Install other packages using pip:
        ```bash
        pip install -r requirements.txt
        pip install onnx
        ```
    - Install TensorRT: copy all ```tensorrt``` packages from ```/usr/lib/python3.10/dist-packages/tensorrt*``` and paste in your conda environment, *e.g.*, ```~/anaconda3/envs/lbm/lib/python3.10/site-packages/```.

## 👣 Step-by-step tutorial
- Before start, please make sure you have a [LBM checkpoint]([huggingface](https://huggingface.co/anonymous1966/LBM)) at ```checkpoints/lbm.pt```.
- In this tutorial, we set ```max_point_num=10``` as the default settting. ```max_point_num``` significantly affects inference speed since TensorRT allocates corresponding memory before tracking. Please set ```max_point_num``` to optimize your practical applications.

### Export to ONNX

```bash
python tools/export_onnx.py \
    --config_path lbm/configs/demo.yaml \
    --checkpoint_path checkpoints/lbm.pt \
    --output_file checkpoints/lbm.onnx \
    --check
```
You will get an exported ONNX file at ```checkpoints/lbm.onnx```.

### Export to TensorRT
```bash
python tools/export_trt.py \
    --onnx checkpoints/lbm.onnx \
    --saveEngine checkpoints/lbm.engine \
    --max_point_num 10 \
    --fp16
```
You will get an exported TensorRT file at ```checkpoints/lbm.engine```.

## 🎬 Demo
Similar to the [click demo](../tools/demo_click.py) for point tracking, we provide a click demo for TensorRT point tracking.

```bash
python tools/demo_trt.py \
    --video_path data/demo.mp4 \
    --checkpoint_path checkpoints/lbm.engine \
    --max_point_num 10 \
    --save_dir tmp/output_pttrack_trt 
```
Click points on the cv2 window and press 'q' to quit. The tracking results is visualized on the cv2 window and also saved in ```tmp/output_pttrack_trt```.

## 📊 Benchmark speed
Similar to [Demo](), we provide a speed benchmark for your exported TensorRT model.

```bash
python tools/benchmark_trt.py \
    --video_path data/demo.mp4 \
    --checkpoint_path checkpoints/lbm.engine \
    --warmup 10 \
    --iteration 100 \
    --max_point_num 10 
```

For comparison, we provide a speed benchmark for the original pytorch model.

```bash
python tools/benchmark_torch.py \
    --config_path lbm/configs/demo.yaml \
    --video_path data/demo.mp4 \
    --checkpoint_path checkpoints/lbm.pt \
    --warmup 10 \
    --iterations 100 \
    --device cuda
```
The results are shown as follows:

| Platform | Model | max_point_num | Tracked points | Inference time | FPS |
| -------- | ----- | ------------- | ------------------------ | -------------- | --- |
| RTX 4090 | PyTorch | - | 10 | 20.50 | 48.78 |
| RTX 4090 | TensorRT | 10 | 10 | 3.84 | 260.37 |
| Orin NX  | PyTorch | - | 10 | 142.83 | 7.00 |
| Orin NX  | TensorRT | 10 | 10 | 20.38 | 49.06 |

- In other words, LBM reaches RTX 4090 inference speed on Orin NX with TensorRT acceleration. 
- The inference speed of LBM on Orin NX with PyTorch is slower than the speed report in the paper, because the reported version is accelerated with CUDA compiled ```MultiScaleDeformAttention``` by MMCV, following DeformableDETR and most works uses deformable attention, while TensorRT acceleration starts from native pytorch version of ```MultiScaleDeformAttention``` as shown in the table above.

## 💡 Tips
- 🤔 How to install ```tenosrrt``` on NVIDIA Jetson devices?
    - 😎 Copy all ```tensorrt``` packages from ```/usr/lib/python3.10/dist-packages/tensorrt*``` and paste in your conda environment, *e.g.*, ```~/anaconda3/envs/lbm/lib/python3.10/site-packages/```.
- 🤔 How to  on Jetson devices?
    - 😎 Download packages from [Jetson AI Lab](https://pypi.jetson-ai-lab.dev/jp6/cu126) with corresponding JetPack and CUDA version. Install locally.
- 🤔 Why TensorRT makes model slower than pytorch?
    - 😎 Please check if ```max_point_num``` is set properly during export and inference.

        | Model | max_point_num | Tracked points | Inference time | FPS |
        | ----- | ------------- | ------------------------ | -------------- | --- |
        | PyTorch | - | 10 | 20.50 | 48.78 |
        | TensorRT | 10 | 10 | 3.84 | 260.37 |
        | TensorRT | 100 | 10 | 4.81 | 207.98 |
        | TensorRT | 1000 | 10 | 20.97 | 47.69 |