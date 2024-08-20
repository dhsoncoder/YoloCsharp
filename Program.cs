using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    static void Main(string[] args)
    {
        // Đường dẫn tới mô hình ONNX
        string modelPath = "D:\\YoloCsharp\\bin\\cotdien.onnx";
        // Đường dẫn tới ảnh đầu vào
        string imagePath = "D:\\YoloCsharp\\bin\\anhcd.JPG";
        // Đường dẫn tới ảnh đầu ra
        string outputImagePath = "D:\\YoloCsharp\\bin\\anhcd.JPGoutput_with_boxes.jpg";

        // Tạo một instance của InferenceSession
        using var session = new InferenceSession(modelPath);

        // Tiền xử lý ảnh
        var tensor = PreprocessImage(imagePath, 640, 640);

        // Tạo một Dictionary để chứa tensor đầu vào
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("images", tensor) };

        // Chạy mô hình và nhận kết quả
        using var results = session.Run(inputs);
        var outputTensor = results.First(x => x.Name == "output0").AsTensor<float>();

        // Chuyển đổi output tensor thành mảng và in ra từng giá trị
        var outputData = outputTensor.ToArray();
        for (int i = 0; i < 120; i += 12)
        {
            Console.WriteLine($"Block {i / 12}:");
            for (int j = 0; j < 12; j++)
            {
                Console.WriteLine($"Value[{i + j}] = {outputData[i + j]}");
            }
            Console.WriteLine();
        }


        // Xử lý đầu ra để vẽ bounding boxes

        var boxes = ExtractBoundingBoxes(outputData, 640, 640); // Extract bounding boxes



        // Vẽ bounding boxes lên ảnh
        DrawBoundingBoxes(imagePath, boxes, outputImagePath);
    }

    private static Tensor<float> PreprocessImage(string imagePath, int width, int height)
    {
        using var image = new Bitmap(imagePath);
        using var resized = new Bitmap(image, new Size(width, height));

        var data = new float[width * height * 3];

        // Lấy dữ liệu ảnh từ bitmap
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = resized.GetPixel(x, y);
                int index = (y * width + x) * 3;
                data[index] = pixel.R / 255.0f;
                data[index + 1] = pixel.G / 255.0f;
                data[index + 2] = pixel.B / 255.0f;
            }
        }

        // Tạo Tensor từ dữ liệu ảnh
        var tensor = new DenseTensor<float>(data, new int[] { 1, 3, height, width });
        return tensor;
    }


    private static (Rectangle, float)[] ExtractBoundingBoxes(float[] outputData, int width, int height)
    {
        // Đây là ví dụ đơn giản. Thực tế, bạn sẽ cần phải chuyển đổi đầu ra của mô hình thành các bounding box.
        // Giả sử đầu ra là các điểm tin cậy và bounding box ở định dạng [x_min, y_min, x_max, y_max].
        var boxes = new (Rectangle, float)[outputData.Length / 5];
        for (int i = 0; i < boxes.Length; i++)
        {
            // Giả định dữ liệu đầu ra là các giá trị cho x_min, y_min, x_max, y_max, confidence.
            int xMin = (int)(outputData[i * 5] * width);
            int yMin = (int)(outputData[i * 5 + 1] * height);
            int xMax = (int)(outputData[i * 5 + 2] * width);
            int yMax = (int)(outputData[i * 5 + 3] * height);
            float confidence = outputData[i * 5 + 4];

            var box = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin);
            boxes[i] = (box, confidence);
        }
        return boxes;
    }

    private static void DrawBoundingBoxes(string imagePath, (Rectangle, float)[] boxes, string outputImagePath)
    {
        using var image = new Bitmap(imagePath);
        using var graphics = Graphics.FromImage(image);
        var pen = new Pen(Color.Red, 2);

        foreach (var (box, confidence) in boxes)
        {
            graphics.DrawRectangle(pen, box);
            graphics.DrawString($"{confidence:0.00}", new Font("Arial", 12), Brushes.Red, box.X, box.Y - 20);
        }

        image.Save(outputImagePath, ImageFormat.Jpeg);
    }
}
