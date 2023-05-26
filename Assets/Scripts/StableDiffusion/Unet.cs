using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace StableDiffusion
{
    public class Unet
    {
        private static InferenceSession unetEncoderModel;
        private static SchedulerBase scheduler;

        private const int batch_size = 1;
        private const int height = 512;
        private const int width = 512;

        private const float scale = 1.0f / 0.18215f;

        public static void LoadModel(string path)
        {
            unetEncoderModel = new InferenceSession(path, Options());
        }

        public static void Free() { unetEncoderModel.Dispose(); }

        public static void Inference(int steps, DenseTensor<float> textEmbeddings, float cfg, int seed, ref Texture2D result, bool useLMS)
        {
            scheduler = useLMS ? new LMSDiscreteScheduler() : new EulerAncestralDiscreteScheduler();
            var timesteps = scheduler.SetTimesteps(steps);

            var latents = GenerateLatentSample(batch_size, seed, scheduler.InitNoiseSigma);
            var input = new List<NamedOnnxValue>();

            for (int t = 0; t < steps; t++)
            {
                var latentModelInput = TensorHelper.Duplicate(latents.ToArray(), new[] { 2, 4, height / 8, width / 8 });

                latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);

                input = CreateUnetModelInput(textEmbeddings, latentModelInput, timesteps[t]);

                // Run Inference
                var output = unetEncoderModel.Run(input);
                var outputTensor = (output.ToList().First().Value as DenseTensor<float>);

                var splitTensors = TensorHelper.SplitTensor(outputTensor, new[] { 1, 4, height / 8, width / 8 });
                var noisePred = splitTensors.Item1;
                var noisePredText = splitTensors.Item2;

                noisePred = performGuidance(noisePred, noisePredText, cfg);

                latents = scheduler.Step(noisePred, timesteps[t], latents);
            }

            latents = TensorHelper.MultipleTensorByFloat(latents.ToArray(), scale, latents.Dimensions.ToArray());
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", latents) };

            VAE.ConvertToImage(ref result, VAE.Decoder(decoderInput), width, height);
        }

        public static Tensor<float> GenerateLatentSample(int batchSize, int seed, float initNoiseSigma)
        {
            Random.InitState(seed);
            var channels = 4;
            var latents = new DenseTensor<float>(new[] { batchSize, channels, height / 8, width / 8 });
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = Random.Range(0.0f, 1.0f); // Uniform(0,1) random number
                var u2 = Random.Range(0.0f, 1.0f); // Uniform(0,1) random number
                var radius = Mathf.Sqrt(-2.0f * Mathf.Log(u1)); // Radius of polar coordinates
                var theta = 2.0f * Mathf.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Mathf.Cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latentsArray[i] = (float)standardNormalRand * initNoiseSigma;
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;
        }

        public static List<NamedOnnxValue> CreateUnetModelInput(Tensor<float> encoderHiddenStates, Tensor<float> sample, long timeStep)
        {
            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timeStep }, new int[] { 1 }))
            };

            return input;
        }

        private static Tensor<float> performGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Dimensions[0]; i++)
                for (int j = 0; j < noisePred.Dimensions[1]; j++)
                    for (int k = 0; k < noisePred.Dimensions[2]; k++)
                        for (int l = 0; l < noisePred.Dimensions[3]; l++)
                            noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);

            return noisePred;
        }

        private static SessionOptions Options()
        {
            var sessionOptions = new SessionOptions();

            try
            {
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                sessionOptions.AppendExecutionProvider_CUDA();
            }
            //catch
            //{
            //    sessionOptions.EnableMemoryPattern = false;
            //    sessionOptions.AppendExecutionProvider_DML();
            //}
            finally { sessionOptions.AppendExecutionProvider_CPU(); }

            return sessionOptions;
        }
    }
}