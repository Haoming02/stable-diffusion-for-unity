using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;
using MathRandom = Unity.Mathematics.Random;

namespace StableDiffusion
{
    public class Unet
    {
        private static InferenceSession unetEncoderModel;
        private static SchedulerBase scheduler;

        private const int batch_size = 1;
        private const int height = 512;
        private const int width = 512;
        private const int channels = 4;

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
            var latents = new DenseTensor<float>(new[] { batchSize, channels, height / 8, width / 8 });
            float[] latentsArray = latents.ToArray();
            MathRandom random = new MathRandom((uint)seed);

            Parallel.For(0, latentsArray.Length, i =>
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                float u1 = random.NextFloat(); // Uniform(0,1) random number
                float u2 = random.NextFloat(); // Uniform(0,1) random number
                float radius = math.sqrt(-2.0f * math.log(u1)); // Radius of polar coordinates
                float theta = 2.0f * math.PI * u2; // Angle of polar coordinates
                float standardNormalRand = radius * math.cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latentsArray[i] = standardNormalRand * initNoiseSigma;
            });

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;
        }

        public static List<NamedOnnxValue> CreateUnetModelInput(Tensor<float> encoderHiddenStates, Tensor<float> sample, long timeStep)
        {
            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<float>(new float[] { timeStep }, new int[] { 1 }))
                // NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timeStep }, new int[] { 1 }))
            };

            return input;
        }

        private static Tensor<float> performGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            Parallel.For(0, noisePred.Dimensions[0], i =>
            {
                for (int j = 0; j < noisePred.Dimensions[1]; j++)
                    for (int k = 0; k < noisePred.Dimensions[2]; k++)
                        for (int l = 0; l < noisePred.Dimensions[3]; l++)
                            noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);
            });

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
            finally
            {
                sessionOptions.AppendExecutionProvider_CPU();
            }

            return sessionOptions;
        }
    }
}