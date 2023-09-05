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
    public class VideoSementationService
    {
        private readonly ISegmentationModelInferenceService inferenceService;

        public VideoSementationService(
            ISegmentationModelInferenceService inferenceService)
        {
            this.inferenceService = inferenceService;
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
                var originalBitmapSize = (bitmap.Width, bitmap.Height);

                if (laskmask == default)
                {
                    var outputBitmap = this.inferenceService.GetSegmentationMap(bitmap);
                    laskmask = new Bitmap(
                        NumpyToBitmap(outputBitmap),
                        originalBitmapSize.Width,
                        originalBitmapSize.Height);
                }

                // Create masked image
                var originalImage = bitmap;
                Bitmap result = default;
                using Image<Rgb24> imageReleaded = SixLabors.ImageSharp.Image.Load<Rgb24>(originalImage.ToArray(ImageFormat.Bmp));
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
                maskFpsCounter++;

                if (maskFpsCounter >= 5)
                {
                    laskmask = default;
                    maskFpsCounter = 0;
                }

                vw.Write(result.ToMat());
            }


        }

        private Bitmap NumpyToBitmap(NDArray array)
        {
            var w = 256;
            var h = 256;
            Bitmap pic = new Bitmap(256, 256, PixelFormat.Format32bppArgb);

            for (int x = 0; x < w; x++)
            {
                for (int y = 0; y < h; y++)
                {
                    var color = MapIndexToColor(array[x, y]);

                    var c = System.Drawing.Color.FromArgb(140, color.R, color.G, color.B);

                    pic.SetPixel(y, x, c);
                }
            }

            return pic;
        }

        private (int R, int G, int B) MapIndexToColor(int classIndex)
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
    }
}
