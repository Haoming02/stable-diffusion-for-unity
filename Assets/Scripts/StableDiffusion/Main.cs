using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using UnityEngine;

namespace StableDiffusion
{
    public class Main
    {
        private const int EmbeddingSize = 59136;    // 77 x 768

        private static DenseTensor<float> textEmbeddings;

        public delegate void pipelineLoaded();
        public static pipelineLoaded onReady;

        public static void Init(string unet, string textEncoder, string clipTokenizer, string extension, string vaeDecoder)
        {
            Unet.LoadModel(unet);
            TextEncoder.LoadModel(textEncoder);
            TextTokenizer.LoadModel(clipTokenizer, extension);
            VAE.LoadModel(vaeDecoder);

            int[] uncondInputTokens = TextTokenizer.CreateUncondInput();
            float[] uncondEmbedding = TextEncoder.Encode(uncondInputTokens).ToArray();

            textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });
            for (int i = 0; i < EmbeddingSize; i++)
                textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];

            onReady?.Invoke();
        }

        public static void Run(string prompt, int steps, float cfg, int seed, ref Texture2D output, bool useLMS)
        {
            var textTokenized = TextTokenizer.TokenizeText(prompt);
            float[] textPromptEmbeddings = TextEncoder.Encode(textTokenized).ToArray();

            for (int i = 0; i < EmbeddingSize; i++)
                textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];

            Unet.Inference(steps, textEmbeddings, cfg, seed, ref output, useLMS);
        }

        public static void Free()
        {
            Unet.Free();
            TextEncoder.Free();
            TextTokenizer.Free();
            VAE.Free();
        }
    }
}