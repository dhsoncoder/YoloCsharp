using System;
using Microsoft.ML.OnnxRuntime;
using YoloCSharp;

class Program
{
    static void Main(string[] args)
    {
        // Đường dẫn tới mô hình ONNX
        string modelPath = "path_to_model";
        // Đường dẫn tới ảnh đầu vào
        string imagePath = "path_to_input_image";
        // Đường dẫn tới ảnh đầu ra
        string outputImagePath = "path_to_output_image";

        string yamlFilePath = "path_to_yaml"; ;

        // Tạo một instance của YoloV8
        var predictor = new YoloV8(modelPath,yamlFilePath);

        // Chạy dự đoán
        predictor.Detect(imagePath, outputImagePath);
     
    }
}
