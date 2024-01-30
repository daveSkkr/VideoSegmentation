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

        public Image<Rgb24> CreateSegmentationMapFor(Image<Rgb24> image)
        {
            var (width, height) = (image.Width, image.Height);
            var output = NumpyToImage(
                this.inferenceService.GetSegmentationMap(image), alphaChannel: 140);

            output.Mutate(mutate => mutate.Resize(width, height));

            return output;
        }

        public void Dispose()
        {
            inferenceService.Dispose();
        }

        private Image<Rgb24> NumpyToImage(NDArray array, int alphaChannel)
        {
            var width = array.shape[0];
            var height = array.shape[1];

            var image = new Image <Rgb24>(width, height);

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        var color = this.mapIndexToColor(array[y, x]);

                        pixelSpan[x] = new Rgb24((byte)color.R, (byte)color.G, (byte)color.B);
                    }
                }
            });

            return image;
        }
    }
}
