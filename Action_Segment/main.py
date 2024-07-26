import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random
import logger as logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19970928 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')
parser.add_argument('--log_dir', default='log')
args = parser.parse_args()
 
num_epochs = 200

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
# batch_input_dim = 196608   # (h w c) = (224*224*3) = 196608

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 10  # videos are at @25 fps. Taking 5 frames per sec for memory constraints


vid_list_file = "./Dataset/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./Dataset/splits/test.split"+args.split+".bundle"

features_path = "./Dataset/features_npy/" # features of dimension [2048]
features_path2 = "./Dataset/features_full/" #features of dimension [h,w,c] for evaluation graphs
gt_path = "./Dataset/groundTruth/"
anno_path = "./Dataset/annotations/"

mapping_file = "./Dataset/mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)
results_dir = "./{}/".format(args.result_dir)
log_dir = "./{}/".format(args.log_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

'''
Logging Setup
'''
logger = logging.get_logger(__name__)
logging.setup_logging(output_dir=log_dir)

logger.critical("Video_train_list: {}".format(vid_list_file))
logger.critical("Video_test_list: {}".format(vid_list_file_tst))
logger.critical("Feature path: {}".format(features_path))
logger.critical("GT path: {}".format(gt_path))
logger.critical("Mapping_file path: {}".format(mapping_file))
logger.critical("Model save dir: {}".format(model_dir))
logger.critical("Result save dir: {}".format(results_dir))
logger.critical("Log save dir: {}".format(log_dir))



'''
Read the mapping file and create and action dictionary
'''
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    row = a.split(' ')
    task = ' '.join(row[1:])
    actions_dict[task] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)

logger.critical("Actions_dict: {}".format(actions_dict))
logger.critical("num_classes: {}".format(num_classes))

logger.critical("\n########################## Model initialisation: ")
trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
if args.action == "train":
    logger.critical("\n####################### Video batch generation: ")
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, features_dim)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, features_dim)
    batch_gen_tst.read_data(vid_list_file_tst)

    logger.critical("\n################ Training: ")
    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if args.action == "predict":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, features_dim)
    batch_gen.read_data(vid_list_file)
    trainer.predict(model_dir, results_dir, features_path, features_path2, anno_path, batch_gen, num_epochs, actions_dict, sample_rate)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate, features_dim)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, features_path2, anno_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

