using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

namespace StableDiffusion
{
    public static class TextTokenizer
    {
        private const int modelMaxLength = 77;
        private const int blankTokenValue = 49407;

        private static SessionOptions sessionOptions;

        private static string modelPath = null;
        public static void SetModel(string path)
        {
            modelPath = path;
            sessionOptions = new SessionOptions();
            sessionOptions.RegisterOrtExtensions();
        }

        public static int[] TokenizeText(string text)
        {
            using InferenceSession textTokenizerModel = new InferenceSession(modelPath, sessionOptions);

            var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };

            var tokens = textTokenizerModel.Run(inputString);

            var inputIds = (tokens.ToList().First().Value as IEnumerable<long>).ToArray();
            var InputIdsInt = inputIds.Select(x => (int)x).ToArray();

            if (InputIdsInt.Length < modelMaxLength)
            {
                var pad = Enumerable.Repeat(blankTokenValue, 77 - InputIdsInt.Length).ToArray();
                InputIdsInt = InputIdsInt.Concat(pad).ToArray();
            }

            return InputIdsInt;
        }

        public static int[] CreateUncondInput()
        {
            var inputIds = new List<int>() { 49406 };

            var pad = Enumerable.Repeat(blankTokenValue, modelMaxLength - inputIds.Count()).ToArray();
            inputIds.AddRange(pad);

            return inputIds.ToArray();
        }
    }
}
