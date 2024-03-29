using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

namespace StableDiffusion
{
    public static class TextEncoder
    {
        private static string modelPath = null;
        public static void SetModel(string path) => modelPath = path;

        public static DenseTensor<float> Encode(int[] tokenizedInput)
        {
            using InferenceSession textEncoderModel = new InferenceSession(modelPath);

            var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Count() });
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };

            var encoded = textEncoderModel.Run(input);

            var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
            var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

            return lastHiddenStateTensor;
        }
    }
}
