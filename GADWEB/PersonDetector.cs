using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using sis=SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;

using System.Reflection;
using System.Runtime.InteropServices;
using Keras.Utils;

//偵測人體的部分使用 yolo:
//https://github.com/WongKinYiu/yolov7

namespace GADWEB {

    public class Program {
	    public static bool SystemRunning=true;
        //public static BrowserForm myBrowserForm=null;
	    }
    
    public static class Utils {
        public static sis.Image<Rgb24> ResizeImage(sis.Image img,int targetWidth,int targetHeight) {
            var iw=img.Width;
            var ih=img.Height;
            var w=targetWidth;
            var h=targetHeight;

            float ratioW=(float)w/iw;
            float ratioH=(float)h/ih;
            float scale=Math.Min(ratioW,ratioH);

            var nw=(int)(iw*scale);
            var nh=(int)(ih*scale);

            var pad_dims_w=(w-nw)/2;
            var pad_dims_h=(h-nh)/2;

            // Resize image using default bicubic sampler 
            var image=img.Clone(x => x.Resize(nw,nh));

            var clone=new sis.Image<Rgb24>(w,h);
            clone.Mutate(i => i.Fill(sis.Color.Black));
            clone.Mutate(o => o.DrawImage(image,new sis.Point(pad_dims_w,pad_dims_h),1f)); // draw the first one top left

            return clone;
            }

        //parallel version
        public static Tensor<float> ExtractPixels(sis.Image<Rgb24> img) {
            Tensor<float> input=new DenseTensor<float>(new[] {1,3,img.Height,img.Width});

            img.Mutate(c => c.ProcessPixelRowsAsVector4((row,p) => {
                int y=p.Y;

                for (int x=0; x<row.Length; x++) {
                    //Debug.WriteLine(y+":"+x+"::");
                    var rr=row[x];
                    input[0,0,y,x]=rr.Z;//B
                    input[0,1,y,x]=rr.Y;//G
                    input[0,2,y,x]=rr.X;//R
                    }
                }));
            
            return input;
            }

        
        public static (int,int,int,int) getBiggerBox(int x,int y,int w,int h,int wtotal,int htotal,float wscale,float hscale) {
            int centerX=(x+w/2);
            int centerY=(y+h/2);

            int biggerW=(int)(w*wscale);
            int biggerH=(int)(h*hscale);
            int bx=centerX-biggerW/2;
            int by=centerY-biggerH/2;

            if (bx<0)
                bx=0;
            if (by<0)
                by=0;
            if ((bx+biggerW)>=wtotal)
                biggerW=wtotal-bx;
            if ((by+biggerH)>=htotal)
                biggerH=htotal-by;

            return (bx,by,biggerW,biggerH);
            }
        }

    public enum YoloLabelKind {
        Generic,
        IstanceSeg,
        }

    public class YoloLabel {
        public int              Id      {get;set;}
        public string?          Name    {get;set;}
        public YoloLabelKind    Kind    {get;set;}
        }

    public class YoloPrediction {
        public YoloLabel?       Label               {get;set;}
        public float            Score               {get;set;}=0;
        //---------------------------------------------------------
        public float            RectX               {get;set;}=0;
        public float            RectY               {get;set;}=0;
        public float            RectWidth           {get;set;}=0;
        public float            RectHeight          {get;set;}=0;
        public bool             hasFace             {get;set;}=false;//是否存在臉
        public string           faceuid             {get;set;}="";
        public int              faceX               {get;set;}=0;
        public int              faceY               {get;set;}=0;
        public int              faceWidth           {get;set;}=0;
        public int              faceHeight          {get;set;}=0;

        public YoloPrediction(YoloLabel label, float confidence) {
            Label=label;
            Score=confidence;
            }
        }

	public class YoloModel {
        public int Width    {get;set;}
        public int Height   {get;set;}
        public int Depth    {get;set;}

        public int              Dimensions      {get;set;}
        public string[]         Outputs         {get;set;}
        public List<YoloLabel>  Labels          {get;set;}=new List<YoloLabel>();
        public bool             UseDetect       {get;set;}
        }


    public  class Yolov7man : IDisposable {
        public readonly InferenceSession    Session;
        public readonly YoloModel           Model=new YoloModel();  
        public static float                 ScoreShreshold=0.48f;//設定 score 必須為 0.48 以上

        public Yolov7man(string ModelPath) {
            Microsoft.ML.OnnxRuntime.SessionOptions opts=new Microsoft.ML.OnnxRuntime.SessionOptions();
            opts.GraphOptimizationLevel=GraphOptimizationLevel.ORT_ENABLE_ALL;
            Session=new InferenceSession(ModelPath,opts);

            get_input_details();
            get_output_details();
            }

        public void SetupLabels(string[] labels) {
            labels.Select((s,i) => new {i,s}).ToList()
                  .ForEach(item => {
                        Model.Labels.Add(new YoloLabel { Id=item.i, Name=item.s });
                        });
            }

        public void SetupYoloDefaultLabels() {
            var s=new string[80];
            SetupLabels(s);
            }

        public List<YoloPrediction> Predict(sis.Image image) {
            var infResult=Inference(image)[0];
            return ParseDetect(infResult,image);
            }

        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, sis.Image image) {
            var result=new ConcurrentBag<YoloPrediction>();

            var w=image.Width;
            var h=image.Height;
            var xGain=Model.Width/(float)w;
            var yGain=Model.Height/(float)h;
            var gain=Math.Min(xGain,yGain);

            var xPad=(Model.Width-w*gain)/2;
            var yPad=(Model.Height-h*gain)/2;

            Parallel.For(0, output.Dimensions[0], (i) => {
                int idx=(int)output[i,5];
                if (idx!=0)
                    return;

                var label=Model.Labels[idx];
                var pred=new YoloPrediction(label,output[i,6]);
                if (pred.Score<ScoreShreshold)
                    return;

                var xMin=(output[i, 1] - xPad) / gain;
                var yMin=(output[i, 2] - yPad) / gain;
                var xMax=(output[i, 3] - xPad) / gain;
                var yMax=(output[i, 4] - yPad) / gain;

                pred.RectX=xMin;
                pred.RectY=yMin;
                pred.RectWidth=xMax-xMin;
                pred.RectHeight=yMax-yMin;

                var tmpv=Utils.getBiggerBox(
                                (int)pred.RectX,
                                (int)pred.RectY,
                                (int)pred.RectWidth,
                                (int)pred.RectHeight,
                                w,
                                h,
                                1.2f,
                                1.3f
                                );

                pred.RectX=tmpv.Item1;
                pred.RectY=tmpv.Item2;
                pred.RectWidth=tmpv.Item3;
                pred.RectHeight=tmpv.Item4;

                result.Add(pred);
                });

            return result.ToList();
            }

        private DenseTensor<float>[] Inference(sis.Image img) {
            var resized=Utils.ResizeImage(img,Model.Width,Model.Height);
            var inputs=new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("images", Utils.ExtractPixels(resized))
                };

            //----------------------------------------------
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result=Session.Run(inputs); // run inference

            var output=new List<DenseTensor<float>>();
            foreach(var item in Model.Outputs) {//add outputs for processing
                output.Add(result.First(x => x.Name==item).Value as DenseTensor<float>);
                };
            return output.ToArray();
            }

        private void get_input_details() {
            Model.Height=Session.InputMetadata["images"].Dimensions[2];
            Model.Width=Session.InputMetadata["images"].Dimensions[3];
            }

        private void get_output_details() {
            Model.Outputs=Session.OutputMetadata.Keys.ToArray();
            Model.Dimensions=Session.OutputMetadata[Model.Outputs[0]].Dimensions[1];
            Model.UseDetect=!(Model.Outputs.Any(x => x=="score"));
            }

        public void Dispose() {
            Session.Dispose();
            }
        }


    class PersonDetector {
        public static Yolov7man y7;

        public static void Init() {
            y7=new Yolov7man(@"assets\yolov7-tiny.onnx");
            y7.SetupYoloDefaultLabels();
            }

        public static List<YoloPrediction> Run(sis.Image Img) {
            return y7.Predict(Img);
            }

        public static List<YoloPrediction> RunForThreshold(sis.Image Img,float v) {
            var predset=Run(Img);
            return predset.Where(x => x.Score>=v).ToList();
            }
        }


    
}
