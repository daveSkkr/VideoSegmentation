using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public interface ISegmentationModelInferenceService : IDisposable
    {
        NDArray GetSegmentationMap(Bitmap bmpImg);
    }

    public class SegmentationModelInferenceService : ISegmentationModelInferenceService
    {
        private readonly InferenceSession inferenceSession;
        private readonly InferenceExpectedInputMetadata metadata;

        public SegmentationModelInferenceService(InferenceExpectedInputMetadata metadata)
        {
            this.metadata = metadata;
            this.inferenceSession = new InferenceSession(metadata.ModelPath);
        }

        public NDArray GetSegmentationMap(Bitmap bmpImg)
        {
            var input = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor(metadata.InputName, BitmapAsNetResizedInput(bmpImg)) };

            var outputTensor = this.inferenceSession.Run(input).First().AsTensor<float>();

            var outputNumpy = np.array(outputTensor.AsEnumerable())
                .reshape(new Shape(outputTensor.Dimensions.ToArray(), outputTensor.Strides.ToArray()));

            var segmentationMap = np.argmax(outputNumpy, 1)[0];

            return segmentationMap;
        }

        public Tensor<float> BitmapAsNetResizedInput(Bitmap imageSource)
        {
            var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imageSource.ToArray(ImageFormat.Bmp));

            image.Mutate(x => x.Resize(this.metadata.ImageDimensions.Width, this.metadata.ImageDimensions.Height));

            Tensor<float> input = new DenseTensor<float>(
                new[] { 1, this.metadata.Channels, this.metadata.ImageDimensions.Width, this.metadata.ImageDimensions.Height });

            var mean = this.metadata.MeanForChannels;
            var stddev = this.metadata.StdForChannels;

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

        public void Dispose()
        {
            this.inferenceSession.Dispose();
        }
    }
}
