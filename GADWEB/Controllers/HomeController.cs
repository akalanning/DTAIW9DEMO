extern alias cc;
using ccsd=cc::System.Drawing;

using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using System.Text.Json;
using System.Diagnostics;
using Newtonsoft.Json;
using sis=SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;
//using coih=CoenM.ImageHash;
using System.Security.Cryptography;
using GADWEB;
//using static System.Resources.ResXFileRef;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.IO.Packaging;
//using myDB;

using Microsoft.AspNetCore.Mvc;
//using System.Diagnostics;
//using static IronSoftware.Drawing.AnyBitmap;
using System.Drawing;

namespace GADWEB.Controllers {


	public class HomeController:Controller {
		
		public HomeController(ILogger<HomeController> logger) {
			}

		public IActionResult Index() {
			return PartialView();
			}

		
		//========
		public static byte[] BitmapToByteArray(Bitmap bitmap) {

            using(MemoryStream ms=new MemoryStream()) {
                bitmap.Save(ms,ImageFormat.Bmp);
                return ms.ToArray();
                }
            }

        public static bool ready=true;

        public static string grayscaleImage(string PH) {//將彩色圖變成灰階：stackoverflow.com/questions/2265910/convert-an-image-to-grayscale
            string newPH="";
            using(Bitmap c=new Bitmap(PH)) {

                // Loop through the images pixels to reset color.
                for (int x=0;x<c.Width;x++) {
                    for (int y=0;y<c.Height;y++) {
                        Color oc=c.GetPixel(x, y);
                        int grayscale=(int)((oc.R*0.3)+(oc.G*0.59)+(oc.B*0.11));//greyscale
                        Color newC=Color.FromArgb(grayscale,grayscale,grayscale);
                        c.SetPixel(x,y,newC);
                        }
                    }
                
                newPH=PH+".grayscale.png";
                c.Save(newPH);
                }

            return newPH;
            }

        public static (int,int) askAndWaitGenderDetectorCheck(string PH) {
            lock(GenderDetector.locker) {
                GenderDetector.TargetGender=-1;
                GenderDetector.TargetGenderCode=-1;
                GenderDetector.TargetPH=PH;
                }

            SpinWait.SpinUntil(()=>GenderDetector.TargetGender>=0,100000000);
            return (GenderDetector.TargetGender,GenderDetector.TargetGenderCode);
            }

        public IActionResult detectGender(string IMGDATA,long BEGT) {
            if (!ready)
                return Json(new {CODE=-1});//rear
            ready=false;

            long t1=DateTime.Now.Ticks;
            if (IMGDATA==null) {
                ready=true;
                return Json(new {CODE=1});//rear
                }

            //----------------------------------
            byte[] A=Convert.FromBase64String(IMGDATA.Substring(23));//read from pos=23 =? remove header "data:image/jpeg;base64,"


            //step1. 偵測人體 (可能會有多個)
            //----------------------------------
            sis.Image<Rgba32> sisImg=sis.Image.Load<Rgba32>(A);
            var predset=PersonDetector.RunForThreshold(sisImg,0.4f);
            if (predset.Count==0) {//若分數太低則放棄
                ready=true;
                return Json(new {CODE=2});
                }

            //偵測人臉 (對每個人體偵測臉的部分)
            //----------------------------------
            bool existFace=false;
            Image Img;
            using (var ms=new MemoryStream(A)) {
                Img=Image.FromStream(ms);

                long tick=DateTime.Now.Ticks;
                Img.Save($@"c:\tmp\gad\gad.{tick}.png");
                
                int faceid=0;
                foreach(var pred in predset) {
                    pred.faceuid=$@"{tick}-{faceid}";

                    Bitmap bmp=(Bitmap)Img;
                    Rectangle tmpRect=new Rectangle(
                                            (int)pred.RectX,
                                            (int)pred.RectY,
                                            (int)pred.RectWidth,
                                            (int)pred.RectHeight
                                            );
                    Bitmap personbmp=bmp.Clone(tmpRect,bmp.PixelFormat);
                    var personBA=BitmapToByteArray(personbmp);

                    //personbmp.Save($@"c:\tmp\gad\gad.{pred.faceuid}.person.png");

                    //----------------------------------
                    //detect face
                    int x=0,y=0,w=0,h=0;
                    using (var pms=new MemoryStream(personBA)) {
                        ccsd.Image img=ccsd.Image.FromStream(pms);
                        (x,y,w,h)=FaceTester.Run(img);
                        }

                    pred.hasFace=(w!=0);//是否存在人臉
                    if (pred.hasFace) {
                        Rectangle cloneRect=new Rectangle(x,y,w,h);
                        Bitmap cloneBitmap=personbmp.Clone(cloneRect,personbmp.PixelFormat);
                        cloneBitmap.Save($@"c:\tmp\gad\gad.{pred.faceuid}.face.png");

                        pred.faceX=(int)(100f*(pred.RectX+x)/Img.Width);
                        pred.faceY=(int)(100f*(pred.RectY+y)/Img.Height);
                        pred.faceWidth=(int)(100f*w/Img.Width);
                        pred.faceHeight=(int)(100f*h/Img.Height);

                        existFace=true;
                        }

                    faceid++;
                    }
                }

            //---------------------------------
            if (!existFace) {//沒有任何人臉則放棄
                ready=true;
                cleanFiles();
                return Json(new {CODE=3});
                }

            //抓出面積最大的那個=>最接近鏡頭
            var best=predset.Where(x => x.hasFace)//必須有偵測到臉
                            //.OrderByDescending(x => x.Score)
                            .OrderByDescending(x => x.RectWidth*x.RectHeight)
                            .FirstOrDefault();
            
            //用我們的模型偵測其性別
            //---------------------------------
            string PH=$@"c:\tmp\gad\gad.{best.faceuid}.face.png";
            string gPH=grayscaleImage(PH);
            (int gender,int gendercode)=askAndWaitGenderDetectorCheck(gPH);
            Debug.WriteLine($@"gender={gender} (code={gendercode}) :: {gPH}");

            cleanFiles();

            ready=true;

            return Json(new {
                        CODE=0,
                        GENDER=gender,
                        GENDERCODE=gendercode,
                        FACEX=best.faceX,
                        FACEY=best.faceY,
                        FACEWIDTH=best.faceWidth,
                        FACEHEIGHT=best.faceHeight 
                        });
            }

        public static void cleanFiles() {
            foreach(string imgph in Directory.EnumerateFiles($@"c:\tmp\gad","gad.*.png")) {
                //if (!imgph.Contains("grayscale"))
                    System.IO.File.Delete(imgph);
                }
            }






		
		}
}