

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static System.Net.Mime.MediaTypeNames;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using System.Drawing;
using System.Drawing.Imaging;
using Emgu.CV;
using Emgu.CV.CvEnum;
using static Emgu.CV.VideoCapture;
using SixLabors.ImageSharp.Formats.Jpeg;
using Emgu.CV.Structure;

var modelPath = "C:\\Users\\sikor\\ground\\ML_playground\\segmentor.onnx";
var inputVideoPath = @"C:\Users\sikor\Documents\krk3.mp4";
var outputVideoPath = @"C:\Users\sikor\Documents\krk_masked.mp4";

var maskFpsCounter = 0;
Bitmap lastMask = default;

using var vw = new VideoWriter(outputVideoPath, 25, new System.Drawing.Size(1920, 1080), true);
using var session = new InferenceSession(modelPath);
using (var video = new VideoCapture(inputVideoPath, API.Any, new Tuple<CapProp, int>(CapProp.Fps, 1)))
using (var mat = new Mat())

while (video.Read(mat))
{
        var bitmap = mat.ToBitmap();

        var bitmapSize = (Width: bitmap.Width, Height: bitmap.Height);

        if (lastMask == default)
        {
            var input = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor("input.1", ImageAsNetResizedInput(bitmap)) };

            var outputTensor = session.Run(input).First().AsTensor<float>();
            var outputNumpy = np.array(outputTensor.AsEnumerable())
                .reshape(new Shape(outputTensor.Dimensions.ToArray(), outputTensor.Strides.ToArray()));
            var segmentationMap = np.argmax(outputNumpy, 1)[0];
            lastMask = new Bitmap(
                NumpyToBitmap(segmentationMap),
                bitmapSize.Width,
                bitmapSize.Height);
        }

        // Create masked image
        var originalImage = bitmap;
        Bitmap result = default; 
        using Image<Rgb24> imageReleaded = SixLabors.ImageSharp.Image.Load<Rgb24>(originalImage.ToArray(ImageFormat.Bmp));
        using (Image<Rgba32> outputImage = new Image<Rgba32>(bitmapSize.Width, bitmapSize.Height))
        {
            // take the 2 source images and draw them on the image
            outputImage.Mutate(o => o
                .DrawImage(imageReleaded, new SixLabors.ImageSharp.Point(0, 0), 1f)
                .DrawImage(SixLabors.ImageSharp.Image.Load(lastMask.ToArray(ImageFormat.Bmp)), new SixLabors.ImageSharp.Point(0, 0), 1f) // draw the second next to it
            );

            result = outputImage.ToBitmap();
        };
        maskFpsCounter++;

        if (maskFpsCounter >= 5)
        {
            lastMask = default;
            maskFpsCounter = 0;
        }

        vw.Write(result.ToMat());
}

    
Console.WriteLine();

static Bitmap NumpyToBitmap(NDArray array)
{
    var w = 256;
    var h = 256;
    Bitmap pic = new Bitmap(256, 256, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

    (int R, int G, int B) mapIndexToColor(int classIndex)
    {
        return classIndex switch
        {
            1 => (128, 64, 128),
            2 => (70, 70, 70),
            3 => (153, 153, 153),
            4 => (107, 142, 35),
            5 => (70, 130, 180),
            6 => (220, 20, 60),
            7 => (0, 0, 142),
            _ => (0, 0, 0)
        };
    }

    for (int x = 0; x < w; x++)
    {
        for (int y = 0; y < h; y++)
        {
            int arrayIndex = y * w + x;
            var color = mapIndexToColor(array[x, y]);

            var c = System.Drawing.Color.FromArgb(140, color.R, color.G, color.B);

            pic.SetPixel(y, x, c);
        }
    }

    return pic;
}

static Tensor<float> ImageAsNetResizedInput(Bitmap imageSource)
{
    var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imageSource.ToArray(ImageFormat.Bmp));
        
    image.Mutate(x => x.Resize(256, 256));

    Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 256, 256 });
    var mean = new[] { 0.485f, 0.456f, 0.406f };
    var stddev = new[] { 0.229f, 0.224f, 0.225f };

    image.ProcessPixelRows(accessor =>
    {
        for (int y = 0; y < accessor.Height; y++)
        {
            Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
            for (int x = 0; x < accessor.Width; x++)
            {
                input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
            }
        }
    });

    return input;
}
