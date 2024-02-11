using NumSharp;

namespace ConsoleApp1
{
    public delegate (int R, int G, int B) MapIndexToColor(int classIndex);

    public class ImageSegmentationService : IDisposable
    {
        private readonly ISegmentationModelInferenceService inferenceService;
        private readonly MapIndexToColor mapIndexToColor;

        public ImageSegmentationService(
            ISegmentationModelInferenceService inferenceService,
            MapIndexToColor mapIndexToColor)
        {
            this.inferenceService = inferenceService;
            this.mapIndexToColor = mapIndexToColor;
        }

        public Image<Rgba32> CreateSegmentationMapFor(Image<Rgb24> image)
        {
            var (width, height) = (image.Width, image.Height);
            var segmentationMap = this.inferenceService.GetSegmentationMap(image);
            var output = NumpyToImage(segmentationMap, alphaChannel: 140);

            output.Mutate(mutate => mutate.Resize(width, height));

            return output;
        }

        public void Dispose()
        {
            inferenceService.Dispose();
        }

        private Image<Rgba32> NumpyToImage(NDArray array, int alphaChannel)
        {
            var width = array.shape[0];
            var height = array.shape[1];

            var image = new Image<Rgba32>(width, height);

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgba32> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        var color = this.mapIndexToColor(array[y, x]);

                        pixelSpan[x] = new Rgba32((byte)color.R, (byte)color.G, (byte)color.B, (byte) 200);
                    }
                }
            });

            return image;
        }
    }
}
