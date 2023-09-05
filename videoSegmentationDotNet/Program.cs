using System.Drawing;
using ConsoleApp1;

var modelPath = "C:\\Users\\sikor\\ground\\ML_playground\\segmentor.onnx";
var inputVideoPath = @"C:\Users\sikor\Documents\krk3.mp4";
var outputVideoPath = @"C:\Users\sikor\Documents\krk_masked.mp4";

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


using var session = new SegmentationModelInferenceService(
    new InferenceExpectedInputMetadata(
        modelPath, 
        "input.1", 
        3,
        meanForChannels: new[] { 0.485f, 0.456f, 0.406f },
        stdForChannels: new[] { 0.229f, 0.224f, 0.225f },
        imageDimensions: (Width: 256, Height: 256)));

var videoSegmentationService = new VideoSementationService(session, mapIndexToColor);

videoSegmentationService.SegmentizeVideo(inputVideoPath, outputVideoPath);
    
Console.ReadLine();


