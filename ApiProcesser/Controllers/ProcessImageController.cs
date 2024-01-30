using ConsoleApp1;
using Microsoft.AspNetCore.Mvc;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using System.Diagnostics;
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

        [HttpPost()]
        public IActionResult Process(ImageToProcess imageToProcess)
        {

            var image = SixLabors.ImageSharp.Image.Load<Rgb24>(Convert.FromBase64String(imageToProcess.PayloadBase64));
            var segmentationMap = imageSegmentationService.CreateSegmentationMapFor(image);

            using (var ms = new MemoryStream())
            {
                segmentationMap.Save(ms, new PngEncoder());
                return Ok(new ResponseDto() { Payload = ms.ToArray() });
            }
        }
    }

    public class ImageToProcess
    {
        public string PayloadBase64 { get; init; }

        public string ImgFormat { get; init; }
    }

    public class ResponseDto
    {
        public byte[] Payload { get; init; }

        public string ImgFormat { get; init; }
    }
}