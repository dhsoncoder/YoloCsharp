using System;
using Microsoft.ML.OnnxRuntime;
using YoloCSharp;

class Program
{
    static void Main(string[] args)
    {
        // Đường dẫn tới mô hình ONNX
        string modelPath = "F:\\C#\\yolov8.onnx";
        // Đường dẫn tới ảnh đầu vào
        string imagePath = "F:\\C#\\nc.jpg";
        // Đường dẫn tới ảnh đầu ra
        string outputImagePath = "F:\\C#\\img_done.jpg";

        string yamlFilePath = "F:\\Job20.v1i.yolov5pytorch\\coco.yaml"; ;

        // Tạo một instance của YoloV8
        var predictor = new YoloV8(modelPath,yamlFilePath);

        // Chạy dự đoán
        predictor.Detect(imagePath, outputImagePath);
     
    }
}
