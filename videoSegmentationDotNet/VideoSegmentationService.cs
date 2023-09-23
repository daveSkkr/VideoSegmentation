using Emgu.CV.CvEnum;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Emgu.CV.VideoCapture;
using System.Drawing;
using System.Drawing.Imaging;
using NumSharp;

namespace ConsoleApp1
{
    public delegate (int R, int G, int B) MapIndexToColor(int classIndex);

    public class VideoSegmentationService
    {
        private readonly ISegmentationModelInferenceService inferenceService;
        private readonly MapIndexToColor mapIndexToColor;

        public VideoSegmentationService(
            ISegmentationModelInferenceService inferenceService,
            MapIndexToColor mapIndexToColor)
        {
            this.inferenceService = inferenceService;
            this.mapIndexToColor = mapIndexToColor;
        }

        public void SegmentizeVideo(string inputPath, string outputPath)
        {
            Bitmap laskmask = default;
            var maskFpsCounter = 0;

            using var vw = new VideoWriter(outputPath, 25, new System.Drawing.Size(1920, 1080), true);
            using (var video = new VideoCapture(inputPath, API.Any, new Tuple<CapProp, int>(CapProp.Fps, 1)))
            using (var mat = new Mat())
            while (video.Read(mat))
            {
                var bitmap = mat.ToBitmap();

                var inputByteArray = mat.GetData();
                var inputByteNumpy = NDArray.FromMultiDimArray<byte>(inputByteArray);

                var originalBitmapSize = (bitmap.Width, bitmap.Height);

                if (laskmask == default)
                {
                    var outputBitmap = this.inferenceService.GetSegmentationMap(bitmap);
                    laskmask = new Bitmap(
                        NumpyToBitmap(outputBitmap),
                        originalBitmapSize.Width,
                        originalBitmapSize.Height);
                }

                var maskedImage = CreateMaskedImage(laskmask, bitmap, originalBitmapSize);

                maskFpsCounter++;

                if (maskFpsCounter >= 5)
                {
                    laskmask = default;
                    maskFpsCounter = 0;
                }

                vw.Write(maskedImage.ToMat());
            }
        }

        private Bitmap CreateMaskedImage(Bitmap laskmask, Bitmap originalImage, (int Width, int Height) originalBitmapSize)
        {
            Bitmap result = default;
            var imageReleaded = SixLabors.ImageSharp.Image.Load<Rgb24>(originalImage.ToArray(ImageFormat.Bmp));
            using (Image<Rgba32> outputImage = new Image<Rgba32>(originalBitmapSize.Width, originalBitmapSize.Height))
            {
                outputImage.Mutate(o => o
                    .DrawImage(imageReleaded, new SixLabors.ImageSharp.Point(0, 0), 1f)
                    .DrawImage(
                        SixLabors.ImageSharp.Image.Load(laskmask.ToArray(ImageFormat.Bmp)),
                        new SixLabors.ImageSharp.Point(0, 0), 1f)
                );

                result = outputImage.ToBitmap();
            };

            return result;
        }

        private Bitmap NumpyToBitmap(NDArray array)
        {
            var w = array.shape[0];
            var h = array.shape[1];
            Bitmap pic = new Bitmap(w, h, PixelFormat.Format32bppArgb);

            for (int x = 0; x < w; x++)
            {
                for (int y = 0; y < h; y++)
                {
                    var color = this.mapIndexToColor(array[x, y]);

                    var c = System.Drawing.Color.FromArgb(140, color.R, color.G, color.B);

                    pic.SetPixel(y, x, c);
                }
            }

            return pic;
        }
    }
}
