# Freehand-Gensin-Diffusion
Transferring Gensin PV into freehand style with Diffusion Model.

# Plans
- [x] Inference codes of image-model
- [x] Pretrained weights for 480x320 resolution
- [ ] Pretrained weights for 1280x720 resolution
- [ ] Inference codes of video-model incorporating temporal module.
- [ ] Training scripts


# Examples 

Here are some results we generated, with the resolution of 480x320.

![](https://github.com/Kebii/Freehand-Gensin-Diffusion/blob/master/assets/Focalors_demo_480x320_wotpl.gif)

![](https://github.com/Kebii/Freehand-Gensin-Diffusion/blob/master/assets/Navia_demo_480x320_wotpl.gif)

# Limitations
We observe following shortcomings in current version:
1. The primary issue is the temporal inconsistency in the generated frames, which causes flickering and jittering in the video.
2. Training and inference for this model are inefficient, requiring substantial computational resources.

# Installation

## Build Environtment

We Recommend a python version `>=3.10` and cuda version `=11.7`. Then build environment as follows:

```shell
conda create -n Gensin python=3.10
conda activate Gensin
# Install requirements with pip:
pip install -r requirements.txt
```

## Download weights

You can download weights manually, which has some steps:

1. Download our trained weights from [BaiduDisk](https://), which include two parts: `denoising_unet.pth`, `reference_unet.pth`.

2. Download pretrained weight of based models and other components: 
    - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
    - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

3. Download dwpose weights (`dw-ll_ucoco_384.onnx`, `yolox_l.onnx`) following [this](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet).

Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- denoising_unet.pth
|-- reference_unet.pth
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

# Inference 

Here is the cli command for running inference scripts:

```shell
python -m scripts.paint_gensin --config ./configs/prompts/paint_gensin.yaml -W 480 -H 320
```

* You can refer the format of `paint_gensin.yaml` and modify the `input_video_path` to transfer other Gensin PVs in MP4 format.

# Disclaimer

This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

# Acknowledgements

This repository is build based on [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone). We thank them for their excellent work in releasing high-quality code.
