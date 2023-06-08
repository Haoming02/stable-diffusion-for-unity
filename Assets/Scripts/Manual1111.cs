using UnityEngine;
using UnityEngine.UI;

public class Manual1111 : MonoBehaviour
{
    [SerializeField]
    private InputField promptField;

    [SerializeField]
    private Text stepsLabel;
    [SerializeField]
    private Text cfgLabel;
    [SerializeField]
    private Text schedulerLabel;

    private int steps = 20;
    private float cfg = 7;
    private bool useLMS = false;

    private const int resolution = 512;

    [SerializeField]
    private RawImage result;
    private Texture2D output;

    void Awake()
    {
        Application.targetFrameRate = 60;

        StableDiffusion.Main.Init(
            Application.streamingAssetsPath + "/unet/" + "model" + ".onnx",
            Application.streamingAssetsPath + "/text_encoder/" + "model" + ".onnx",
            Application.streamingAssetsPath + "/tokenizer/" + "cliptokenizer" + ".onnx",
            Application.streamingAssetsPath + "/tokenizer/" + "ortextensions" + ".dll",
            Application.streamingAssetsPath + "/vae_decoder/" + "model" + ".onnx"
        );

        output = new Texture2D(resolution, resolution, TextureFormat.RGBA32, false);
    }

    public void SetSteps(float s)
    {
        steps = (int)s;
        stepsLabel.text = $"{steps}";
    }

    public void SetGuidance(float g)
    {
        cfg = g;
        cfgLabel.text = $"{cfg}";
    }

    public void ToggleSampler()
    {
        useLMS = !useLMS;
        schedulerLabel.text = useLMS ? "LMS" : "Euler A";
    }

    public void Run()
    {
        StableDiffusion.Main.Run(promptField.text, steps, cfg, Random.Range(0, int.MaxValue), ref output, useLMS);
        result.texture = output;
    }

    void OnDisable()
    {
        StableDiffusion.Main.Free();
    }
}