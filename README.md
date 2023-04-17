# StableDiffusion for Unity
This was achieved by using `ONNX` version of the models, then use `Microsoft.ML.OnnxRuntime` to inference at runtime. 

### Requirement
- Download the [Plugins and Models](https://drive.google.com/file/d/1zA8TExTPkWq7sRDW6JMUGkDqhFvRpSlo/view?usp=share_link) first and put them into the project directory
  - The model is the official [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/onnx)
  - You can use other models as well so long as you convert it to `.onnx` format
- It is possible to run on CPU only. Though `CUDA` and `CUDNN` are highly recommanded
  - 20 Steps took: **~15sec** on `RTX 3060`; **~1min 10sec** on `i7-13700`
  - **Officially supported [versions:](https://onnxruntime.ai/docs/tutorials/csharp/csharp-gpu.html)** `CUDA: v11.6`, `CUDNN: v8.5.0.96` *(This project was built on `ONNX Runtime 1.14`)*

### Implemented Features
- Positive Prompt
- Steps
- `Euler a` and `LMS` sampling method
- CFG Scale

### References
- https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html
- https://github.com/cassiebreviu/StableDiffusion
- https://onnxruntime.ai/docs/get-started/with-csharp.html