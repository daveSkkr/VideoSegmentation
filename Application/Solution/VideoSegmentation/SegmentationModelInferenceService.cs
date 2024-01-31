using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using System.Drawing;

namespace ConsoleApp1
{
    public interface ISegmentationModelInferenceService : IDisposable
    {
        NDArray GetSegmentationMap(Image<Rgb24> image);
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

        public NDArray GetSegmentationMap(Image<Rgb24> image)
        {
            var input = new List<NamedOnnxValue>() { NamedOnnxValue.CreateFromTensor(metadata.InputName, BitmapAsNetResizedInput(image)) };

            var outputTensor = this.inferenceSession.Run(input).First().AsTensor<float>();
            var outputNumpy = np.array(outputTensor.AsEnumerable())
                .reshape(new Shape(outputTensor.Dimensions.ToArray(), outputTensor.Strides.ToArray()));
            var segmentationMap = np.argmax(outputNumpy, 1)[0];

            return segmentationMap;
        }

        public Tensor<float> BitmapAsNetResizedInput(Image<Rgb24> image)
        {
            Tensor<float> input = new DenseTensor<float>(
                new[] { 1, this.metadata.Channels, this.metadata.ModelExpectedImageDimensions.Height, this.metadata.ModelExpectedImageDimensions.Width });
            
            using (var imageCopy = image.Clone())
            {
                imageCopy.Mutate(x => x.Resize(this.metadata.ModelExpectedImageDimensions.Width, this.metadata.ModelExpectedImageDimensions.Height));

                var mean = this.metadata.MeanForChannels;
                var stddev = this.metadata.StdForChannels;

                imageCopy.ProcessPixelRows(accessor =>
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
            }

            return input;
        }

        public void Dispose()
        {
            this.inferenceSession.Dispose();
        }
    }
}
