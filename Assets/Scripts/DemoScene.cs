using UnityEngine;
using System.Collections;

public class DemoScene : MonoBehaviour
{
    private const int steps = 12;
    private const float cfg = 6.0f;

    private const int resolution = 512;

    private Texture2D paintingTexture;
    private Texture2D skyboxTexture;

    [SerializeField]
    private Canvas loading;
    [SerializeField]
    private Canvas generating;

    [SerializeField]
    private Renderer skybox;
    [SerializeField]
    private Renderer painting;

    private float xRotation = 0.0f;

    private const string unetFolder = "/unet/";
    private const string encoderFolder = "/text_encoder/";
    private const string tokenizerFolder = "/tokenizer/";
    private const string vaeFolder = "/vae_decoder/";

    private const string extension = ".onnx";

    void Awake()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        StableDiffusion.Main.onReady += OnReady;

        paintingTexture = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);
        skyboxTexture = new Texture2D(resolution, resolution, TextureFormat.RGB24, false);
    }

    void Start()
    {
        loading.enabled = true;
        generating.enabled = false;

        StableDiffusion.Main.Init(
            $"{Application.streamingAssetsPath}{unetFolder}model{extension}",
            $"{Application.streamingAssetsPath}{encoderFolder}model{extension}",
            $"{Application.streamingAssetsPath}{tokenizerFolder}cliptokenizer{extension}",
            $"{Application.streamingAssetsPath}{vaeFolder}model{extension}"
        );
    }

    private void OnReady()
    {
        loading.enabled = false;
        StartCoroutine(MainLoop());
    }

    private const float sensitivity = 50.0f;

    private IEnumerator MainLoop()
    {
        while (true)
        {
            float x = Input.GetAxis("Mouse X") * sensitivity * Time.deltaTime;
            float y = Input.GetAxis("Mouse Y") * sensitivity * Time.deltaTime;

            xRotation = Mathf.Clamp(xRotation - y, -60.0f, 60.0f);

            Vector3 camRotation = transform.rotation.eulerAngles;

            camRotation.x = xRotation;
            camRotation.y += x;

            transform.rotation = Quaternion.Euler(camRotation);

            if (Input.GetKeyDown(KeyCode.Alpha1))
            {
                generating.enabled = true;

                yield return null;

                GenSkybox();

                yield return null;

                generating.enabled = false;
            }

            if (Input.GetKeyDown(KeyCode.Alpha2))
            {
                generating.enabled = true;

                yield return null;

                GenPainting();

                yield return null;

                generating.enabled = false;
            }

            yield return null;
        }
    }

    private const string promptPainting = "a masterpiece painting of a flower vase, by davinci";
    private const string promptSkybox = "a dslr photo of blue sky with cloud and sun, hdr";

    private void GenPainting()
    {
        StableDiffusion.Main.Run(promptPainting, steps, cfg, Random.Range(0, int.MaxValue), ref paintingTexture);
        painting.material.mainTexture = paintingTexture;
    }

    private void GenSkybox()
    {
        StableDiffusion.Main.Run(promptSkybox, steps, cfg, Random.Range(0, int.MaxValue), ref skyboxTexture);
        skybox.material.mainTexture = skyboxTexture;
    }
}
