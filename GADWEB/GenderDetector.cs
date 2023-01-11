//social.msdn.microsoft.com/Forums/en-US/9cd8f782-9ff8-4f87-871a-06aad06c313c/the-type-bitmap-exists-in-both-corecompatsystemdrawing-and-systemdrawingcommon?forum=aspdotnetcore
//stackoverflow.com/questions/3672920/two-different-dll-with-same-namespace
extern alias cc;
using ccsd=cc::System.Drawing;

using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.PreProcessing.Image;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using K=Keras.Backend;
using static System.Net.Mime.MediaTypeNames;

//偵測性別的部分，模型來自我們自己的訓練：
//https://colab.research.google.com/drive/18JmJY5HzU1B3codl3FCrBN2yWJ_MnhtB?usp=sharing

namespace GADWEB {

    class GenderDetector {
        public static BaseModel model=null;
        public static Thread thr=null;
        public static string TargetPH="";
        public static int TargetGender=-1;//性別：男女
        public static int TargetGenderCode=-1;//性別＋口罩
        public static object locker=new object();

        public static void Init() {
            thr=new Thread(new ThreadStart(doInit));
            thr.Start();
            }

        public static void doInit() {
            
            Keras.Keras.DisablePySysConsoleLog=true;

            Debug.WriteLine(@"genderdetector model read");
            model=Sequential.LoadModel(@"assets\colab.w9.genderdetector.h5");
            Debug.WriteLine(@"genderdetector model is ready");
            
            while(Program.SystemRunning) {
                SpinWait.SpinUntil(()=>false,1);

                lock(locker) {
                    if (TargetPH!="") {
                        int v=predictImage(TargetPH);
                        switch(v) {
                            case 0://FEMALE.MASK
                            case 1://FEMALE.NOMASK
                                TargetGender=0;
                                break;
                            case 2://MALE.MASK
                            case 3://MALE.NOMASK
                                TargetGender=1;
                                break;
                            }

                        TargetGenderCode=v;

                        //Debug.WriteLine(TargetGenderCode+"::"+TargetPH);
                        TargetPH="";
                        }
                    }
                }

            model.Dispose();
            }

        public static int predictImage(string imgPH) {//看圖來預測性別
            var img=ImageUtil.LoadImg(
                path:imgPH
                );
            
            NDarray x=ImageUtil.ImageToArray(img);
            x=x.reshape(1,x.shape[0],x.shape[1],x.shape[2]);
            x=x.astype(np.float32);
            x /= 255;

            var y=model.Predict(x);
            var yargmax=y.argmax();
            int index=yargmax.asscalar<int>();

            return index;
            }
        }
}