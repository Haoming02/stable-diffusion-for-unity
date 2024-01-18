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

    private int steps = 16;
    private float cfg = 7.5f;

    private const int resolution = 512;

    [SerializeField]
    private RawImage result;
    private Texture2D output;

    private const string unetFolder = "/unet/";
    private const string encoderFolder = "/text_encoder/";
    private const string tokenizerFolder = "/tokenizer/";
    private const string vaeFolder = "/vae_decoder/";

    private const string extension = ".onnx";

    void Awake()
    {
        StableDiffusion.Main.Init(
            $"{Application.streamingAssetsPath}{unetFolder}model{extension}",
            $"{Application.streamingAssetsPath}{encoderFolder}model{extension}",
            $"{Application.streamingAssetsPath}{tokenizerFolder}cliptokenizer{extension}",
            $"{Application.streamingAssetsPath}{vaeFolder}model{extension}"
        );

        output = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);
    }

    public void SetSteps(float s)
    {
        steps = (int)s;
        stepsLabel.text = $"{steps}";
    }

    public void SetGuidance(float g)
    {
        cfg = g;
        cfgLabel.text = $"{Mathf.Round(cfg * 10.0f) / 10}";
    }

    public void Run()
    {
        StableDiffusion.Main.Run(promptField.text, steps, cfg, Random.Range(0, int.MaxValue), ref output);
        result.texture = output;
    }
}
