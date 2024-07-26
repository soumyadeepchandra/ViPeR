# ViTALS: Vision Transformer for Action Localization in Surgical Nephrectomy
This repository contains code for the paper

# Abstract 
Surgical action localization is a challenging computer vision problem. While it has promising applications including automated training of surgery procedures, surgical workflow optimization, etc., appropriate model design is pivotal to accomplishing this task. Moreover, the lack of suitable medical datasets adds an additional layer of complexity. To that effect, we introduce a new complex dataset of nephrectomy surgeries called UroSlice. To perform the action localization from these videos, we propose a novel model termed as ‘ViTALS’ (Vision Transformer for Action Localization in Surgical Nephrectomy). Our model incorporates hierarchical dilated temporal convolution layers and inter-layer residual connections to capture the temporal correlations at finer as well as coarser granularities. The proposed approach achieves state-of-the-art performance on Cholec80 and UroSlice datasets (89.8% and 66.1% accuracy, respectively), validating its effectiveness.


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
2. UroSlice: Will be released post-acceptance

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

5. Run python ./Feature_extractor/resnet_feature.py to train the Feature Extractor. 
The model weight is saved into ./models_feature/*.pth

6. Run python ./Feature_extractor/datasets_npy_using_resnet.py to use the saved_model to generate the .npy feature of each video in [Video Folder]: (./ViTALS/Dataset/videos). 

## Train your own model
7. Run "python ./Action_Segment/main.py --action train --split 1"  to train model.

## Use Pretrained weights
8. Run "python ./Action_Segment/main.py --action predict --split 1" to generate predicted results for split.

## Evaluating the model
9. Run "python ./Action_Segment/eval.py --split 1 to evaluate the performance.


Feel free to raise a issue if you got trouble with our code.
