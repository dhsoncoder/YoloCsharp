using System;
using Microsoft.ML.OnnxRuntime;
using YoloCSharp;

class Program
{
    static void Main(string[] args)
    {
        // Đường dẫn tới mô hình ONNX
        string modelPath = "G:\\HaiSon\\YoloC#\\dhsoncoder\\YoloCsharp\\bin\\cotdien.onnx";
        // Đường dẫn tới ảnh đầu vào
        string imagePath = "G:\\HaiSon\\YoloC#\\dhsoncoder\\YoloCsharp\\bin\\anhcd.JPG";
        // Đường dẫn tới ảnh đầu ra
        string outputImagePath = "G:\\HaiSon\\YoloC#\\dhsoncoder\\YoloCsharp\\bin\\output_ok.jpg";

        string yamlFilePath = "G:\\HaiSon\\YoloC#\\dhsoncoder\\YoloCsharp\\bin\\cotdien.yaml"; ;

        // Tạo một instance của YoloV8
        var predictor = new YoloV8(modelPath,yamlFilePath);

        // Chạy dự đoán
        predictor.Detect(imagePath, outputImagePath);
     
    }
}
