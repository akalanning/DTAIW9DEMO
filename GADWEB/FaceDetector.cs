extern alias cc;
using ccsd=cc::System.Drawing;

using Accord.Vision.Detection;
using Accord.Vision.Detection.Cascades;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using SixLabors.ImageSharp.Processing.Processors;
//using System.Windows.Forms;
using Accord.Imaging.Filters;
using Accord.Collections;

//偵測人臉的程式是複製自:
//https://devindeep.com/face-mask-detection-using-ml-net-model-builder-and-c/

namespace GADWEB {

    class ImageProcessor {
        private ccsd.Bitmap _bitmap;
        public ccsd.Bitmap Result { get => _bitmap; }
        public ImageProcessor(ccsd.Bitmap bitmap) {
            _bitmap = bitmap;
            }
        internal ImageProcessor Grayscale() {
            var grayscale = new Grayscale(0.2125, 0.7154, 0.0721);
            _bitmap = grayscale.Apply(_bitmap);
            return this;
            }

        internal ImageProcessor EqualizeHistogram() {
            HistogramEqualization filter = new HistogramEqualization();
            filter.ApplyInPlace(_bitmap);
            return this;
            }

        internal ImageProcessor Resize(Size size) {
            _bitmap = new ccsd.Bitmap(_bitmap, size);
            return this;
            }
        }

    //---------------------------------------------------------------------------
    class FaceDetectorParameters {
        public float ScalingFactor { get; private set; }
        public ObjectDetectorScalingMode ScalingMode { get; private set; }
        public ObjectDetectorSearchMode SearchMode { get; private set; }
        public bool UseParallelProcessing { get; private set; }
        public int MinimumSize { get; private set; }

        public bool IsValid { get; private set; }

        private FaceDetectorParameters(float scalingFactor, int minimumSize, ObjectDetectorScalingMode objectDetectorScalingMode,
            ObjectDetectorSearchMode objectDetectorSearchMode, bool useParallelProcessing, bool isValid) {
            ScalingFactor = scalingFactor;
            MinimumSize = minimumSize;
            ScalingMode = objectDetectorScalingMode;
            SearchMode = objectDetectorSearchMode;
            UseParallelProcessing = useParallelProcessing;
            IsValid = isValid;
            }

        public static FaceDetectorParameters Create(float scalingFactor, int minimumSize, ObjectDetectorScalingMode objectDetectorScalingMode,
            ObjectDetectorSearchMode objectDetectorSearchMode, bool useParallelProcessing) =>
                new FaceDetectorParameters(scalingFactor, minimumSize, objectDetectorScalingMode, objectDetectorSearchMode, useParallelProcessing, true);
        }
    
    //------------------------------------------------------------------------
    class Face {
        public Rectangle rect;

        public Face(Rectangle x) {
            rect = x;
            }

        public Rectangle FaceRectangle { get => rect; }

        public override string ToString() =>
            $"X: {rect.X}, Y: {rect.Y}, Width: {rect.Width}, Height: {rect.Height}";
        }

    //------------------------------------------------------------------------
    class FaceDetector {
        private HaarObjectDetector _detector;

        public FaceDetector() {
            _detector = new HaarObjectDetector(new FaceHaarCascade());
            }

        internal IEnumerable<Face> ExtractFaces(ccsd.Bitmap picture, FaceDetectorParameters faceDetectorParameters) =>
            picture == null ?
            Enumerable.Empty<Face>() :
            ProcessFrame(picture, faceDetectorParameters).Select(rec => new Face(rec));

        private IEnumerable<Rectangle> ProcessFrame(ccsd.Bitmap picture, FaceDetectorParameters faceDetectorParameters) {
            _detector.MinSize = new Size(faceDetectorParameters.MinimumSize, faceDetectorParameters.MinimumSize);
            _detector.ScalingFactor = faceDetectorParameters.ScalingFactor;
            _detector.ScalingMode = faceDetectorParameters.ScalingMode;
            _detector.SearchMode = faceDetectorParameters.SearchMode;
            _detector.UseParallelProcessing = faceDetectorParameters.UseParallelProcessing;
            _detector.MaxSize = new Size(600, 600);
            _detector.Suppression = 1;
            return _detector.ProcessFrame(picture);//stackoverflow.com/questions/55402352/using-accord-imaging-in-uwp
            }
        }

    class FaceTester {
        private static FaceDetector _faceDetector = new FaceDetector();

        public static float ScaleFactor { get; set; }                   = 1.1f;//1.1f;
        public static int MinSize { get; set; }                         = 1;//5;
        public static ObjectDetectorScalingMode ScaleMode { get; set; } = ObjectDetectorScalingMode.GreaterToSmaller;
        public static ObjectDetectorSearchMode SearchMode { get; set; } = ObjectDetectorSearchMode.NoOverlap;//.Single;//.Average;
        public static bool Parallel { get; set; }                       = true;

        public static (int,int,int,int) Run(ccsd.Image img) {

            MinSize=img.Width/6;//人臉約佔寬度的 1/3，這裡指的寬度是指 yolo 抓下來的 person 區塊，但因為我們有放大，所以設為 1/6

            var pic=(ccsd.Bitmap)img;

            var faces=_faceDetector.ExtractFaces(
                    new ImageProcessor(pic).Grayscale().EqualizeHistogram().Result,
                    FaceDetectorParameters.Create(ScaleFactor, MinSize, ScaleMode, SearchMode, Parallel));

            //foreach(var face in faces) {
            //    }
            
            if (faces.Count()!=0) {
                Face face=faces.First();

                (int x,int y,int w,int h)=Utils.getBiggerBox(
                                            face.rect.X,
                                            face.rect.Y,
                                            face.rect.Width,
                                            face.rect.Height,
                                            img.Width,
                                            img.Height,
                                            1.7f,
                                            2.1f
                                            );
                

                return (x,y,w,h);
                }
            else
                return (0,0,0,0);
            }
        }





}