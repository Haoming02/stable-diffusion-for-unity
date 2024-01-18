using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace StableDiffusion
{
    public static class VAE
    {
        private static string modelPath = null;
        public static void SetModel(string path) => modelPath = path;

        public static Tensor<float> Decoder(List<NamedOnnxValue> input)
        {
            using InferenceSession vaeDecoderModel = new InferenceSession(modelPath);

            var output = vaeDecoderModel.Run(input);
            return output.ToList().First().Value as Tensor<float>;
        }

        public static void ConvertToImage(ref Texture2D result, Tensor<float> output, int width, int height)
        {
            var pixels = new Color[width * height];

            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    pixels[x + y * width] = new Color(
                        output[0, 0, y, x] / 2 + 0.5f,
                        output[0, 1, y, x] / 2 + 0.5f,
                        output[0, 2, y, x] / 2 + 0.5f,
                        1.0f
                    );
                }
            });

            result.SetPixels(pixels);
            result.Apply();
        }
    }
}
