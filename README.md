# DTAIW9DEMO

本作業使用 Keras.NET，但由於 Keras.NET 太老舊，它只能接受以下的套件版本，請如下安裝：

pip install tensorflow-cpu==2.6.0

pip install keras==2.6.0

pip install pillow

接著，請將 colab.w9.genderdetector.h5 檔放到 assets 目錄下，並且點開 VS2022，點開該檔的 properties，確認 [Copy To Output Directory] 欄位設定為 "Copy Always"

接著，請建立一個 c:\tmp\gad 目錄，用於儲存暫時性的檔案


接著，按 F5 執行


第一次執行會很慢，因為讀取 colab.w9.genderdetector.h5 會花掉大約 30 秒，請先等待一下再按 [Start Detect] 按鈕。


---------------------

藍色框代表偵測到男生、黃色框代表偵測到女生，實線代表沒戴口罩，虛線代表有戴口罩

![圖示](https://i.imgur.com/TiRdaUm.png)

![圖示](https://i.imgur.com/ZXLccW1.png)







 
