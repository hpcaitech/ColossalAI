# Overview

Image segmentation aims to segment the picture into different blocks according to the content. Compared with image classification and detection, segmentation is a finer task because each pixel needs to be classified. Similar to the detection model, the semantic segmentation model can be established based on the classification model. For example, we can use the CNN network to extract features for classification. However, due to the existence of hollow convolution, the computational complexity of DeepLab is higher. A more lightweight segmentation model is needed to meet the needs of low latency scenarios, such as autonomous vehicles.

This is the main repository for training DeepLabv3+ based on the CamVid data set for segmentation, which starts the training process using Colossal-AI. The main body of the Encoder in DeepLabv3+ is DCNN with hole convolution. For traditional classification networks such as ResNet, and the spatial pyramid pooling module with hole convolution (Atrous Spatial Pyramid Pooling, ASPP)), both can be used mainly to introduce multi-scale information.

## Requirements
```
python=3.7.11
cuda11.3
cudnn8.0
pytorch 1.10.1
```

## Data
This work supports `from torch.utils.data import Dataset as BaseDataset` as the data feed.

The following data is provided as training examples:
Camvid([Training](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/))

The code snippet is shown below:
```
os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
```
Firstly, you need to set the path to store the dataset by setting the environment variable 
```
os.environ['DATA']= './data'
```
The data are downloaded automatically into the specified directory by
```
os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
```

## Training
There is an example for training deeplabv3 on the CamVid dataset. 
The experiments can also be conducted on other datasets.
Please refer to: 
```
python ./cityscapes/download/downloader.py
```
for cityscapes data set. And then do this in training script:
```
DATA_DIR = './data/cityscapes/'
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')
```
Simply run
```
python ./train_mytest.py
```
to train the model.

## Experiment results
The evaluation of the performance are as follows.

![results](results/accuracy.png)
![results](results/loss.png)

The accuracy curve initially shows an obvious increase. After 200 to 300 epochs, the curve fluctuates within a fixed value range, representing the tend to be converged. Due to the difficulty in large-batch training, there is a decline during the middle of the training process. The training loss decreases as the number of epochs increases. The learning rate and the batch size are set to be 0.0001 and 256, respectively, which can be further finetuned for better convergence speed and training efficiency.
