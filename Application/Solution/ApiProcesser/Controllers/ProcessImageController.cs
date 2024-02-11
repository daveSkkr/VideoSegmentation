using ConsoleApp1;
using Emgu.CV.Structure;
using Microsoft.AspNetCore.Mvc;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
using System.Drawing;
using System.Reflection;

namespace ApiProcesser.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ProcessImageController : ControllerBase
    {
        private readonly ImageSegmentationService imageSegmentationService;

        public ProcessImageController(ImageSegmentationService imageSegmentationService)
        {
            this.imageSegmentationService = imageSegmentationService;
        }

        [HttpPost("ProcessForMask")]
        public IActionResult ProcessForMask(ImageToProcess imageToProcess)
        {
            byte[] result;

            using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(Convert.FromBase64String(imageToProcess.PayloadBase64)))
            using (var segmentationMap = imageSegmentationService.CreateSegmentationMapFor(image))
            {
                using (var ms = new MemoryStream())
                {
                    segmentationMap.Save(ms, new PngEncoder());
                    result = ms.ToArray();
                }
            }

            return Ok(result);
        }

        [HttpPost("ProcessApplyingMask")]
        public async IAsyncEnumerable<byte[]> ProcessApplyingMask(ImageToProcess imageToProcess, CancellationToken cancellationToken)
        {
            byte[] result;

            var sw = new Stopwatch();
            sw.Start();

            using (var image = SixLabors.ImageSharp.Image.Load<Rgb24>(Convert.FromBase64String(imageToProcess.PayloadBase64)))
            using (var segmentationMap = imageSegmentationService.CreateSegmentationMapFor(image))
            using (var maskedImage = CreateMaskedImage(segmentationMap, image))
            {
                using (var ms = new MemoryStream())
                {
                    maskedImage.Save(ms, new PngEncoder());
                    result = ms.ToArray();
                }
            }

            sw.Stop();


            while (!cancellationToken.IsCancellationRequested)
            {
                yield return result;
                await Task.Delay(1000);
            }
        }

        private Image<Rgb24> CreateMaskedImage(Image<Rgba32> landMask, Image<Rgb24> originalImage)
        {
            Image<Rgb24> outputImage = new Image<Rgb24>(originalImage.Width, originalImage.Height);
            
            outputImage.Mutate(o => o
                .DrawImage(originalImage, new SixLabors.ImageSharp.Point(0, 0), 1f)
                .DrawImage(landMask, new SixLabors.ImageSharp.Point(0, 0), 1f)
            );

            return outputImage;
        }
    }

    public class ImageToProcess
    {
        public string PayloadBase64 { get; init; }

        public string ImgFormat { get; init; }
    }
}