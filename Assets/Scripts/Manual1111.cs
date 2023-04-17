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
        StableDiffusion.Main.Init(
            Application.streamingAssetsPath + "/Models/" + "unet" + ".onnx",
            Application.streamingAssetsPath + "/Models/" + "encoder" + ".onnx",
            Application.streamingAssetsPath + "/Models/" + "token" + ".onnx",
            Application.streamingAssetsPath + "/Models/" + "ortextensions" + ".dll",
            Application.streamingAssetsPath + "/Models/" + "vae" + ".onnx"
            );
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

    public void UseLMS(bool v)
    {
        useLMS = v;
        schedulerLabel.text = v ? "LMS" : "Euler A";
    }

    public void Run()
    {
        output = new Texture2D(resolution, resolution, TextureFormat.RGBA32, false);
        StableDiffusion.Main.Run(promptField.text, steps, cfg, Random.Range(0, int.MaxValue), ref output, useLMS);
        result.texture = output;
    }

    void OnDisable()
    {
        StableDiffusion.Main.Free();
    }
}