using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

namespace StableDiffusion
{
    public class TextEncoder
    {
        private static InferenceSession textEncoderModel;

        public static void LoadModel(string path)
        {
            textEncoderModel = new InferenceSession(path);
        }

        public static void Free() { textEncoderModel.Dispose(); }

        public static DenseTensor<float> Encode(int[] tokenizedInput)
        {
            var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Count() });

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };

            var encoded = textEncoderModel.Run(input);

            var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
            var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

            return lastHiddenStateTensor;
        }
    }
}