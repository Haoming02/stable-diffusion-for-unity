using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using UnityEngine;

namespace StableDiffusion
{
    public static class Main
    {
        private const int EmbeddingSize = 59136; // 77 x 768

        private static DenseTensor<float> textEmbeddings;

        public delegate void pipelineLoaded();
        public static pipelineLoaded onReady;

        public static void Init(string unet, string textEncoder, string clipTokenizer, string vaeDecoder)
        {
            Unet.SetModel(unet);
            TextEncoder.SetModel(textEncoder);
            TextTokenizer.SetModel(clipTokenizer);
            VAE.SetModel(vaeDecoder);

            // int[] uncondInputTokens = CreateUncondInput();
            int[] negativeInputTokens = TextTokenizer.TokenizeText("low quality, worst quality, jpeg, nsfw");
            float[] uncondEmbedding = TextEncoder.Encode(negativeInputTokens).ToArray();

            textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });
            for (int i = 0; i < EmbeddingSize; i++)
                textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];

            onReady?.Invoke();
        }

        public static void Run(string prompt, int steps, float cfg, int seed, ref Texture2D output)
        {
            var textTokenized = TextTokenizer.TokenizeText($"high quality, best quality, {prompt}");
            float[] textPromptEmbeddings = TextEncoder.Encode(textTokenized).ToArray();

            for (int i = 0; i < EmbeddingSize; i++)
                textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];

            Unet.Inference(steps, textEmbeddings, cfg, seed, ref output);
        }
    }
}
