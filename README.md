<h1 align="center">No Longer Maintained</h1>
<p align="center">This project was originally just a proof of concept. With so many new optimization techniques, such as <a href="https://huggingface.co/blog/sdxl_ort_inference">Turbo</a>, this implementation is extremely slow by today's standard...</p>

# Stable Diffusion for Unity
This is an Unity project that allows you to run the Stable Diffusion pipeline within Unity, achieved by using `ONNX` version of the models and `Microsoft.ML.OnnxRuntime` to inference at runtime. 

## Requirement
- Install [`CUDA Toolkit`](https://developer.nvidia.com/cuda-toolkit-archive) and [`CUDA Deep Neural Network`](https://developer.nvidia.com/cudnn)
    - It is possible to run on CPU only, albeit at a significantly slower speed.
        - 20 Steps took: **~15sec** on `RTX 3060`; **~1min 10sec** on `i7-13700`
#### Supported Versions
- This project was originally built on [`ONNX Runtime 1.14`](https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html)
    - Tested on **`CUDA v11.6`** and **`CUDA v11.8`**
    - Tested on **`CUDNN v8.5.0.96`**

## Getting Started
1. Download the models from **[Release](https://github.com/Haoming02/stable-diffusion-for-unity/releases)**, and put them into the `StreamingAssets` folder.
2. There are 2 sample scenes included:
    - One is a simple UI where you can enter any prompt and adjust some settings to generate an image
    - Another is a 3D scene where you can generate a skybox and a painting at runtime

## Implemented Features
- Positive Prompt
- Steps
- `Euler a` and `LMS` sampling method
- CFG Scale
- Inference `fp16` Precision

# References
- https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html
- https://github.com/cassiebreviu/StableDiffusion
- https://onnxruntime.ai/docs/get-started/with-csharp.html
- https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html
