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
    public class YoloV5
    {
        private Dictionary<int, string> classNames;
        private static int imgWidth, imgHeight;
        private string _modelPath;
        private InferenceSession session;
        private List<int> _modelInput = new List<int>();
        private List<int> _modelOutput = new List<int>();
        private float _confidenceThreshold;
        private float _iouThreshold;

        public YoloV5(string modelPath, string yamlFilePath, float confidenceThreshold = 0.3f, float iouThreshold = 0.4f)
        {
            _modelPath = modelPath;
            session = new InferenceSession(modelPath);
            foreach (var input in session.InputMetadata)
            {
                var inputShape = input.Value.Dimensions.Select(d => d.ToString()).ToArray();
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
                var outputShape = output.Value.Dimensions.Select(d => d.ToString()).ToArray();
                foreach (var shapeElement in outputShape)
                {
                    if (int.TryParse(shapeElement, out int dimension))
                    {
                        _modelOutput.Add(dimension);
                    }
                }
            }
            classNames = LoadClassNamesFromYaml(yamlFilePath);
            _confidenceThreshold = confidenceThreshold;
            _iouThreshold = iouThreshold;
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
        private static DenseTensor<float> ConvertToTensor(NDArray ndArray)
        {
            var shape = ndArray.shape;
            var data = ndArray.Data<float>();  // Lấy dữ liệu float từ NDArray

            // Chuyển NDArray thành DenseTensor<float> với kích thước tương ứng
            return new DenseTensor<float>(data.ToArray(), shape);
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
            var npOutput = np.array(outputTensor).reshape(outputdata2, outputdata3);

        

            // Gọi hàm Postprocess với dữ liệu đầu ra từ mô hình
            Postprocess(imagePath, npOutput, imgWidth, imgHeight, 0.5f, 0.45f, outputImagePath);
        }

        private void Postprocess(string imagePath, NDArray outputData, int imgWidth, int imgHeight, float confidenceThres, float iouThres, string outputImagePath)
        {
            int inputWidth = _modelInput[_modelInput.Count - 2];
            int inputHeight = _modelInput[_modelInput.Count - 1];

            float xFactor = (float)imgWidth / inputWidth;
            float yFactor = (float)imgHeight / inputHeight;

            var boxes = new List<Rectangle>();
            var scores = new List<float>();
            var classIds = new List<int>();

            int rows = outputData.shape[0]; // 25200 rows
            for (int i = 0; i < rows; i++)
            {
                var row = outputData[i, ":"];

                float confidence = row[4].GetValue<float>();
                float maxScore = 0;
                int classId = -1;

                for (int j = 5; j < row.size; j++) // From index 5 onwards are class scores
                {
                    float score = row[j].GetValue<float>();
                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = j - 5;
                    }
                }

                if (confidence > confidenceThres)
                {
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

            var indices = DnnInvoke.NMSBoxes(boxes.ToArray(), scores.ToArray(), confidenceThres, iouThres);
            var finalBoxes = new List<Rectangle>();
            var finalScores = new List<float>();
            var finalClassIds = new List<int>();

            foreach (var index in indices)
            {
                if (index >= 0 && index < boxes.Count)
                {
                    finalBoxes.Add(boxes[index]);
                    finalScores.Add(scores[index]);
                    finalClassIds.Add(classIds[index]);
                }
            }

            DrawBoundingBoxes(imagePath, finalBoxes, finalScores, finalClassIds, outputImagePath);
        }

        private void DrawBoundingBoxes(string imagePath, List<Rectangle> boxes, List<float> scores, List<int> classIds, string outputImagePath)
        {
            using var image = new Bitmap(imagePath);
            using var graphics = Graphics.FromImage(image);

            var font = new Font("Arial", 18);
            var random = new Random();

            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];
                var score = scores[i];
                var classId = classIds[i];

                Color randomColor = Color.FromArgb(random.Next(256), random.Next(256), random.Next(256));
                var pen = new Pen(randomColor, 6);

                graphics.DrawRectangle(pen, box);

                string className = classNames.ContainsKey(classId) ? classNames[classId] : "Unknown";
                string text = $"{className}: {score:0.00}";

                SizeF textSize = graphics.MeasureString(text, font);
                var backgroundBrush = new SolidBrush(Color.FromArgb(150, 0, 0, 0));
                graphics.FillRectangle(backgroundBrush, box.X, box.Y - 20, textSize.Width, textSize.Height);
                graphics.DrawString(text, font, Brushes.White, box.X, box.Y - 20);
            }

            image.Save(outputImagePath, ImageFormat.Jpeg);
        }

        private Dictionary<int, string> LoadClassNamesFromYaml(string yamlFilePath)
        {
            var classNames = new Dictionary<int, string>();

            var yamlStream = new YamlStream();
            using (var reader = new StreamReader(yamlFilePath))
            {
                yamlStream.Load(reader);
            }

            var root = (YamlMappingNode)yamlStream.Documents[0].RootNode;
            var namesNode = (YamlSequenceNode)root.Children[new YamlScalarNode("names")];

            for (int i = 0; i < namesNode.Children.Count; i++)
            {
                classNames.Add(i, namesNode.Children[i].ToString());
            }

            return classNames;
        }
    }


}
