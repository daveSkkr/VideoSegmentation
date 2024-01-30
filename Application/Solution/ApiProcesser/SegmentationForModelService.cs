using ConsoleApp1;
using System.Reflection;

namespace ApiProcesser
{
    public static class SegmentationForModelServiceFactory
    {
        
        public static ImageSegmentationService CreateSegmentationForModelService()
        {
            string modelPath = Path.Combine(
                Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "segmentor.onnx");

            MapIndexToColor mapIndexToColor = (classIndex) =>
                classIndex switch
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


            var session = new SegmentationModelInferenceService(
                new InferenceExpectedInputMetadata(
                    modelPath,
                    "input.1",
                    3,
                    MeanForChannels: new[] { 0.485f, 0.456f, 0.406f },
                    StdForChannels: new[] { 0.229f, 0.224f, 0.225f },
                    ModelExpectedImageDimensions: (Width: 256, Height: 256)));

            return new ImageSegmentationService(session, mapIndexToColor);
        }
    }
}
