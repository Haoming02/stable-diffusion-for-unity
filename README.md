<h1 align="center">No Longer Maintained</h1>
<p align="center">This project was originally just a proof of concept. With so many new optimization techniques, such as <a href="https://huggingface.co/blog/sdxl_ort_inference">Turbo</a>, this implementation is extremely slow by today's standard...</p>

# Stable Diffusion for Unity
This is an Unity project that allows you to run the Stable Diffusion pipeline within Unity, 
achieved by using `ONNX` version of the models and `OnnxRuntime` to inference at runtime. 

## Requirements
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
- [CUDA Deep Neural Network](https://developer.nvidia.com/cudnn)
- [ONNX Runtime](https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html)
    > as shown in the `Plugins` folder

#### Supported Versions
- This project was built on **ONNX Runtime** `1.15.1`
    - Tested on **CUDA** `v11.6` and `v11.8`
    - Tested on **CUDNN** `v8.5.0.96`

## Getting Started
- Download the models from [**Release**](https://github.com/Haoming02/stable-diffusion-for-unity/releases), and put them into the `StreamingAssets` folder.
- 2 sample scenes are included:
    - One is a simple UI where you can enter any prompt and adjust some settings to generate an image
    - Another is a 3D scene where you can generate a skybox and a painting

## Implemented Features
- Positive Prompt
- Steps
- **Euler a** sampling method
- CFG Scale
- Inference `fp16` Precision

# References
- https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html
- https://github.com/cassiebreviu/StableDiffusion
- https://onnxruntime.ai/docs/get-started/with-csharp.html
- https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html
