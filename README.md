# ViPeR: Vision-based Surgical Phase Recognition
This repository contains code for the paper

# Abstract 
Surgical phase recognition poses a significant challenge in computer vision, with promising applications such as automated surgery procedure training and workflow optimization. However, designing an appropriate model is crucial for tackling this task, especially given the lack of suitable medical datasets. To that end, in this work, we introduce UroSlice, a new complex dataset of nephrectomy surgeries. To address the task of phase recognition in these videos, we propose a novel model named ‘ViPeR’ (Vision-based Surgical Phase Recognition). Our model incorporates hierarchical dilated temporal convolution layers and inter-layer residual connections to capture temporal correlations at both fine and coarse granularities. Experimental results demonstrate that our approach achieves state-of-the-art performance on both the publicly available Cholec80 and in-house UroSlice datasets (89.8% and 66.1% accuracy, respectively), thereby validating its effectiveness.

![alt text](https://github.com/soumyadeepchandra/ViPeR/blob/main/model_pipeline.jpg?raw=true)
![alt text](https://github.com/soumyadeepchandra/ViPeR/blob/main/enc_dec.jpg?raw=true)

## Enviroment
Pytorch == 2.0.1, 
torchvision == 0.15.2, 
python == 3.11, 
CUDA == 11.7

Install the required dependencies: 
```python
pip install -r environment.txt
```
## Datasets used
1. Cholec80: [http://camma.u-strasbg.fr/datasets]
2. UroSlice: [https://tinyurl.com/232s2j6m]

## Reproduce our results
1. Download the dataset data.zip (Cholec80 / any other dataset). 

2. Unzip the data.zip file to the current folder (./Dataset)

3. The dataset should look like:
Annotation folder: (./Dataset/annotations). Each row of .csv files contain start_time, stop_time, task_id
Video Folder: (./Dataset/videos)
Features Folder: (./Dataset/features)
Mapping File: (./Dataset/mapping.txt). File contains list of all tasks with their respective task_id
Split Folder: (./Dataset/splits) contains train and test split of the dataset

4. Run the script to read each .csv file and convert into .txt file. Each line of .txt file represent the task of each frame.
```python
python ./Dataset/csv_to_txt_gt.py
```
Ground Truth Folder: (./Dataset/groundTruth)

5. To train the Feature Extractor.
```python
Run python ./Feature_extractor/resnet_feature.py
```
The model weight is saved into ./models_feature/*.pth

6. To use the saved_model to generate the .npy feature of each video in [Video Folder]: (./ViTALS/Dataset/videos).
```python
 python ./Feature_extractor/datasets_npy_using_resnet.py
```

## Train your own model
7. To train model:
   ```python
   python ./Action_Segmnt/main.py --action train --split 1
   ```

## Use Pretrained weights
8. To generate predicted results for split:
   ```python
   python ./Action_Segment/main.py --action predict --split 1
   ```

## Evaluating the model
9. To evaluate the performance:
   ``` python
   python ./Action_Segment/eval.py --split 1
   ```


Feel free to raise an issue if you have trouble with our code.
