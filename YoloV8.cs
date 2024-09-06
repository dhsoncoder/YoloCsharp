using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using YamlDotNet.RepresentationModel;


namespace YoloCSharp
{
    public class YoloV8
    {
        private Dictionary<int, string> classNames;
        private static int test = 0;
        private static int  imgWidth,imgHeight;
        private string _modelPath;
        private InferenceSession session;
        private List<int> _modelInput = new List<int>();
        private List<int> _modelOutput = new List<int>();
        private float _confidenceThreshold;
        private float _iouThreshold;

        public YoloV8(string modelPath, string yamlFilePath, float confidenceThreshold = 0.3f, float iouThreshold = 0.4f)
        {
            _modelPath = modelPath;
            session = new InferenceSession(modelPath);
            foreach (var input in session.InputMetadata)
            {
                var inputInfo = input.Value;
                var inputShape = inputInfo.Dimensions.Select(d => d.ToString()).ToArray(); // Chuyển thành mảng để dễ xử lý

                // Duyệt qua từng phần tử của mảng inputShape và thêm vào list
                foreach (var shapeElement in inputShape)
                {
                    if (int.TryParse(shapeElement, out int dimension))
                    {
                        _modelInput.Add(dimension);
                    }
                }
            }
            foreach (var output in session.OutputMetadata)
            {
                var outputInfo = output.Value;
                var outputShape = outputInfo.Dimensions.Select(d => d.ToString()).ToArray();
                // Duyệt qua từng phần tử của mảng inputShape và thêm vào list
                foreach (var shapeElement in outputShape)
                {
                    if (int.TryParse(shapeElement, out int dimension))
                    {
                        _modelOutput.Add(dimension);
                      
                    }
                }
            }
            // Tải các tên class từ file YAML
            classNames = LoadClassNamesFromYaml(yamlFilePath);
            _confidenceThreshold = confidenceThreshold;
            _iouThreshold = iouThreshold;
        }

        public YoloV8()
        {
        }

        public void Detect(string imagePath, string outputImagePath)
        {
            // Tiền xử lý ảnh
            int inputWidth = _modelInput[_modelInput.Count - 2];
            int inputHeight = _modelInput[_modelInput.Count - 1];
            int outputdata1 = _modelOutput[0];
            int outputdata2 = _modelOutput[1];
            int outputdata3 = _modelOutput[2];
            var ndArray = PreprocessImage(imagePath, inputWidth, inputHeight);

            
            // Chuyển NDArray thành DenseTensor<float>
            var tensor = ConvertToTensor(ndArray);

            // Tạo một Dictionary để chứa tensor đầu vào
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("images", tensor) };

            // Chạy mô hình và nhận kết quả
            using var results = session.Run(inputs);

            Tensor<float> outputTensor = results[0].AsTensor<float>();

            // Chuyển đổi sang ndarray sử dụng NumSharp
            var npOutput = np.array(outputTensor).reshape(outputdata1, outputdata2, outputdata3);

            // Thực hiện squeeze để loại bỏ chiều đầu tiên (1)
            var squeezedOutput = np.squeeze(npOutput);

            // Thực hiện transpose để hoán đổi trục 
            var transposedOutput = np.transpose(squeezedOutput);


            // Gọi hàm Postprocess với dữ liệu đầu ra từ mô hình
            Postprocess(imagePath, transposedOutput, imgWidth, imgHeight, 0.5f, 0.45f, outputImagePath);
        }

        private static DenseTensor<float> ConvertToTensor(NDArray ndArray)
        {
            var shape = ndArray.shape;
            var data = ndArray.Data<float>();  // Lấy dữ liệu float từ NDArray

            // Chuyển NDArray thành DenseTensor<float> với kích thước tương ứng
            return new DenseTensor<float>(data.ToArray(), shape);
        }


        private static NDArray PreprocessImage(string imagePath, int inputWidth, int inputHeight)
        {
            Mat img = new Mat();
            img = CvInvoke.Imread(imagePath);
            // Lấy kích thước ban đầu của ảnh
            imgWidth = img.Width;
            imgHeight = img.Height;


            // Chuyển đổi không gian màu từ BGR sang RGB (tương đương cv2.cvtColor)
            Mat imgRgb = new Mat();
            CvInvoke.CvtColor(img, imgRgb, Emgu.CV.CvEnum.ColorConversion.Bgr2Rgb);

            // Thay đổi kích thước ảnh theo input shape (tương tự cv2.resize)
            Mat resizedImage = new Mat();
            CvInvoke.Resize(imgRgb, resizedImage, new Size(inputWidth, inputHeight));

            // Tạo NDArray để lưu dữ liệu ảnh
            var imageData = np.zeros(new int[] { inputHeight, inputWidth, 3 }, np.float32);
            var imgdata = resizedImage.ToImage<Rgb, byte>();
            // Duyệt qua các pixel và chuyển đổi BGR sang RGB
            for (int j = 0; j < inputHeight; j++)
            {
                for (int i = 0; i < inputWidth; i++)
                {
                    // lấy giá trị pixel tại vị trí (i, j)
                    var pixel = imgdata[i, j];
                    imageData[i, j, 0] = pixel.Red / 255.0f;
                    imageData[i, j, 1] = pixel.Green / 255.0f;
                    imageData[i, j, 2] = pixel.Blue / 255.0f;

                }
            }

            // Transpose ảnh để đưa kênh màu lên đầu, giống với cách xử lý của OpenCV
            imageData = np.transpose(imageData, new int[] { 2, 0, 1 });  // Channel first

            // Thêm chiều mới ở vị trí đầu tiên để phù hợp với input (expand_dims)
            imageData = np.expand_dims(imageData, axis: 0).astype(np.float32);

            // In hình dạng của dữ liệu đầu vào (giống với Python)
            Console.WriteLine("Input shape: " + string.Join(", ", imageData.shape));

            return imageData;
        }

        private void Postprocess(string imagePath, NDArray outputData, int imgWidth, int imgHeight, float confidenceThres, float iouThres, string outputImagePath)
        {
            // Tính toán hệ số tỷ lệ cho tọa độ hộp giới hạn
            int inputWidth = 640; // Chiều rộng của ảnh đầu vào cho mô hình
            int inputHeight = 640; // Chiều cao của ảnh đầu vào cho mô hình

            float xFactor = (float)imgWidth / inputWidth;
            float yFactor = (float)imgHeight / inputHeight;

            var boxes = new List<Rectangle>();


            var scores = new List<float>();
            var classIds = new List<int>();

            // Xác định số lượng hàng và cột của dữ liệu đầu ra
            var shape = outputData.shape;
            int rows = shape[0];
            int cols = shape[1];
            int count = 0;
            for (int i = 0; i < rows; i++)
            {
                // Extract the current row
                var row = outputData[i, ":"];

                // Confidence score
                float confidence = row[4].GetValue<float>();


                // Find class with the highest score
                float maxScore = 0;
                int classId = -1;

                for (int j = 4; j < cols; j++)
                {
                    float score = row[j].GetValue<float>();
                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = j - 4;
                    }
                }

                if (maxScore >= 0.5f)
                {
                    count++;
                    // Compute bounding box coordinates
                    float x = row[0].GetValue<float>();
                    float y = row[1].GetValue<float>();
                    float w = row[2].GetValue<float>();
                    float h = row[3].GetValue<float>();

                    int left = (int)((x - w / 2) * xFactor);
                    int top = (int)((y - h / 2) * yFactor);
                    int width = (int)(w * xFactor);
                    int height = (int)(h * yFactor);

                    boxes.Add(new Rectangle(left, top, width, height));
                    scores.Add(maxScore);

                    classIds.Add(classId);


                }

            }
            Console.WriteLine("So object: " + count);

            // Áp dụng Non-Maximum Suppression (NMS)
            var indices = DnnInvoke.NMSBoxes(boxes.ToArray(), scores.ToArray(), confidenceThres, iouThres);

            // Chuyển đổi chỉ số thành danh sách các hộp giới hạn
            var finalBoxes = new List<Rectangle>();
            foreach (var index in indices)
            {
                if (index >= 0 && index < boxes.Count)  // Đảm bảo chỉ số hợp lệ
                {
                    finalBoxes.Add(boxes[index]);
                }
            }

            // Vẽ các hộp giới hạn lên ảnh
            DrawBoundingBoxes(imagePath, finalBoxes, scores, classIds, outputImagePath);
        }

        private void DrawBoundingBoxes(string imagePath, List<Rectangle> boxes, List<float> scores, List<int> classIds, string outputImagePath)
        {
            // Tạo đối tượng Bitmap từ ảnh đầu vào
            using var image = new Bitmap(imagePath);
            using var graphics = Graphics.FromImage(image);

            // Định nghĩa font chữ và cọ để vẽ text
            var font = new Font("Arial", 18);

            // Tạo một đối tượng Random để tạo màu ngẫu nhiên
            var random = new Random();

            // Duyệt qua các hộp giới hạn và điểm số tương ứng
            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];
                var score = scores[i];
                var classId = classIds[i];

                // Tạo màu ngẫu nhiên cho mỗi hộp giới hạn
                Color randomColor = Color.FromArgb(random.Next(256), random.Next(256), random.Next(256));
                var pen = new Pen(randomColor, 6);
                var brush = new SolidBrush(randomColor);

                // Vẽ hộp giới hạn lên ảnh
                graphics.DrawRectangle(pen, box);

                // Lấy tên class dựa trên classId
                string className = classNames.ContainsKey(classId) ? classNames[classId] : "Unknown";

                // Vẽ điểm số tin cậy và tên class lên ảnh
                string text = $"{className}: {score:0.00}";

                // Tính toán kích thước của phần text
                SizeF textSize = graphics.MeasureString(text, font);

                // Tạo nền cho text bằng cách vẽ một hình chữ nhật nhỏ
                var backgroundBrush = new SolidBrush(Color.FromArgb(150, 0, 0, 0)); // Màu đen với độ trong suốt
                graphics.FillRectangle(backgroundBrush, box.X, box.Y - 20, textSize.Width, textSize.Height);

                // Vẽ chữ lên ảnh, với brush đã tạo trước
                graphics.DrawString(text, font, Brushes.White, box.X, box.Y - 20); // Dùng màu trắng để chữ nổi bật
            }

            // Lưu ảnh đã được vẽ hộp giới hạn
            image.Save(outputImagePath, ImageFormat.Jpeg);
        }
        private Dictionary<int, string> LoadClassNamesFromYaml(string yamlFilePath)
        {
            var classNames = new Dictionary<int, string>();

            // Đọc nội dung file YAML
            var yamlStream = new YamlStream();
            using (var reader = new StreamReader(yamlFilePath))
            {
                yamlStream.Load(reader);
            }

            // Lấy root node
            var root = (YamlMappingNode)yamlStream.Documents[0].RootNode;

            // Lấy phần 'names' từ file YAML
            var namesNode = (YamlSequenceNode)root.Children[new YamlScalarNode("names")];

            // Lặp qua các phần tử trong 'names' và thêm vào dictionary
            for (int i = 0; i < namesNode.Children.Count; i++)
            {
                classNames.Add(i, namesNode.Children[i].ToString());
            }

            return classNames;
        }


    }
}
