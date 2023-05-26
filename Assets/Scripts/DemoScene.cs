using UnityEngine;

public class DemoScene : MonoBehaviour
{
    private const int steps = 4;
    private const float cfg = 4;
    private const bool useLMS = false;

    private const int resolution = 512;

    private Texture2D paintingTexture;
    private Texture2D skyboxTexture;

    [SerializeField]
    private Canvas loading;

    [SerializeField]
    private Renderer skybox;
    [SerializeField]
    private Renderer painting;

    private bool pipelineReady;
    private float xRotation = 0.0f;

    void Awake()
    {
        Application.targetFrameRate = -1;
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Start()
    {
        loading.enabled = true;
        pipelineReady = false;

        StableDiffusion.Main.onReady += OnReady;

        StableDiffusion.Main.Init(
            Application.streamingAssetsPath + "/unet/" + "model" + ".onnx",
            Application.streamingAssetsPath + "/text_encoder/" + "model" + ".onnx",
            Application.streamingAssetsPath + "/tokenizer/" + "cliptokenizer" + ".onnx",
            Application.streamingAssetsPath + "/tokenizer/" + "ortextensions" + ".dll",
            Application.streamingAssetsPath + "/vae_decoder/" + "model" + ".onnx"
        );

        paintingTexture = new Texture2D(resolution, resolution, TextureFormat.RGBA32, false);
        skyboxTexture = new Texture2D(resolution, resolution, TextureFormat.RGBA32, false);
    }

    private void OnReady()
    {
        loading.enabled = false;
        pipelineReady = true;
    }

    private void Update()
    {
        if (!pipelineReady)
            return;

        float x = Input.GetAxis("Mouse X");
        float y = Input.GetAxis("Mouse Y");


        xRotation = Mathf.Clamp(xRotation - y, -60.0f, 60.0f);

        Vector3 camRotation = transform.rotation.eulerAngles;
        camRotation.x = xRotation;
        camRotation.y += x;

        transform.rotation = Quaternion.Euler(camRotation);

        if (Input.GetKeyDown(KeyCode.Alpha1))
            GenSkybox();
        if (Input.GetKeyDown(KeyCode.Alpha2))
            GenPainting();
    }

    private const string promptPainting = "high quality, best quality, masterpiece, painting, davinci";
    private const string promptSkybox = "high quality, best quality, blue sky, cloud, sun, hdr";

    private void GenPainting()
    {
        StableDiffusion.Main.Run(promptPainting, steps, cfg, Random.Range(0, int.MaxValue), ref paintingTexture, useLMS);
        painting.material.mainTexture = paintingTexture;
    }

    private void GenSkybox()
    {
        StableDiffusion.Main.Run(promptSkybox, steps, cfg, Random.Range(0, int.MaxValue), ref skyboxTexture, useLMS);
        skybox.material.mainTexture = skyboxTexture;
    }

    void OnDisable()
    {
        StableDiffusion.Main.Free();
    }
}