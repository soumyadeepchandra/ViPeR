
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pandas as pd

def convert_csv_to_txt(csv_file, gt_txt_file):
  
    task_list = ['0 Preparation', '1 CalotTriangleDissection', '2 ClippingCutting', '3 GallbladderDissection', '4 GallbladderPackaging', '5 CleaningCoagulation', '6 GallbladderRetraction']  

    annotation_data = pd.read_csv(open(csv_file, 'r'), header=None)

    with open(gt_txt_file, 'w') as txt_file:
        
        all_ranges = []
        for step in annotation_data.values:
            start = step[0]
            end = step[1]
            diff = step[1] - step[0]
            row = step[2].split(' ')
            task = ' '.join(row[1:])
            all_ranges.append((start,end))
            for i in range(diff):
                txt_file.write(''.join(task) + '\n')
            

def batch_convert_csv_to_txt(folder_path, target_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in csv_files:
        base_name = os.path.splitext(csv_file)[0]
        file_path = base_name + '.txt'
        csv_file_path = os.path.join(folder_path, csv_file)
        txt_file_path = os.path.join(target_path, file_path)
        print(base_name +'.csv')
        convert_csv_to_txt(csv_file_path, txt_file_path)

# Usage example
folder_path = './ViTALS/Dataset/annotations/'  # Specify the path to the folder containing the .csv files
target_path = './ViTALS/Dataset/groundTruth/'  # Specify the path to the folder saving the .txt files
batch_convert_csv_to_txt(folder_path, target_path)
