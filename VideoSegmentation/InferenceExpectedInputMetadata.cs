namespace ConsoleApp1
{
    public record InferenceExpectedInputMetadata(
        string ModelPath,
        string InputName,
        int Channels,
        float[] MeanForChannels,
        float[] StdForChannels,
        (int Width, int Height) ImageDimensions);
}
