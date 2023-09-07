# Jetson_Nano_SmokingBehavior_Detection
This is an individual project using deep learning algorithm to detect smoking behavior in the public on the platform of Jetson Nano。
I use Jetson Nano b01 as the research platform with a 64GB TF card, a wireless network adapter, a usb webcam and a CSI camera. I install ubuntu 18.04 LTS onto Jetson Nano.

I choose PaddlePaddle as the deep-learning platform, PaddlePaddle is the first independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. The more information you could see via this link: https://www.paddlepaddle.org.cn/en. My project is based on PaddleDetection, an Object Detection toolkit based on PaddlePaddle. It supports object detection, instance segmentation, multiple object tracking and real-time multi-person keypoint detection. https://github.com/PaddlePaddle/PaddleDetection 


Firstly, I download archiconda to build the conda environment, see this link https://github.com/Archiconda/build-tools/releases , because archiconda is available on Jetson ARM structure. Due to Jetpack=4.6, python 3.6 would be the best option to use. After successfully installation, use the terminal to command:

conda create -n env_name python==3.6  # create a new environment with python3.6

conda activate env_name   # enter the new environment

Then, download the paddlepaddle installation for Jetson via the link https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html#python . Command in the terminal：pip install Jetpack4.6: nv-jetson-cuda10.2-cudnn8.2-trt8.0(nano). Wait here until succeed.

Afterwards, download the dataset from public resource. For my project, I aim to detect the smoking behavior so I select to download PP_smoke dataset from here: https://aistudio.baidu.com/datasetdetail/94796. This dataset includes 783 pics of human smoking and 783 labels of segmentation. And git-clone my file to your jetson.

Finally, please look through my file #detectionPaddle.ipynb. I recommend to install the Jupyter Notebook on your Jeston Nano (pip install Jupyter Notebook). And follow my instructions to train the dataset and output the best model and save it (change my route and dir in the file to suit your Jetson!!!). I also suggest to use PC or server with high performance GPU to train the dataset, which is sufficient, accurate and time-saving. When the training process is done, export the trained model to test it on the inference.

For example, the inference on a image is shown in below:

If you try to infer on the video, please open the terminal and command like this:

 cd paddle_smoking_detection
 
 python demo_yuan/infer_video.py

If you would like to infer in the camera, you could command like this in the terminal:
 
 python demo_yuan/infer_cv.py

Here is my video to show my project, you could click it if you would like to see:
https://www.bilibili.com/video/BV1uh4y1P78T/?spm_id_from=333.337.search-card.all.click 
