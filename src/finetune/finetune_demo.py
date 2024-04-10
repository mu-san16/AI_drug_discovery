# import module
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import argparse
import logging
import re
from src.finetune.dti_model import MbertPcnnModel

# setting logging
logging.basicConfig(level = logging.INFO)
# setting parser
parser = argparse.ArgumentParser(description='Training parser')
# setting args
args = parser.parse_args()

# define comandline
parser.add_argument('--dataset_name', type=str, default="Binding_DB", help='dataset_name')
parser.add_argument('--data_path', type=str, default="file_path")
parser.add_argument('--model_version', type=str, default="11", choices=["1", "2", "3", "4", "11", "14"])
parser.add_argument('--learning_rate', type=float, default = 0.0001, help = 'learning_rate')   # may change parameters
parser.add_argument('--fold', default=0, choices=[0,1,2,3,4], type=int)

# k1, k2, k3の調査

output_dir = "%s/%s/mbert_cnn_v%s_lr%.4f_k%d_k%d_k%d_fold%d/" % (args.data_path, args.dataset_name, args.model_version, args.learning_rate, args.k1, args.k2, args.k3, args.fold)


def info_scores(current_step, min_mse_step, min_mse_dev, max_ci_dev, mse_tst, ci_tst, prefix='', checkpoint_time=0, output_dir='', dataset_name='', model_version='', 
                learning_rate=0, k1=0, k2=0, k3=0, num_train_steps=0):
    line1a = '************************** [%s-V%s-lr(%.4f)-f(%d,%d,%d)step(%d/%d)] ***************************' % \
            (dataset_name, model_version, learning_rate, k1, k2, k3, current_step, num_train_steps)
    line1b = '**************************  %s Best @ [%d]  ***************************' % (prefix, min_mse_step)
    line2 = '********** [dev]\tmse:\t%f\tci\t%f **********' % (min_mse_dev, max_ci_dev)
    line3 = '********** [tst]\tmse:\t%f\tci\t%f **********' % (mse_tst, ci_tst)
    line4 = '********** [time]\t%ds **********' % checkpoint_time
    line5 = '********************************************************************'
    logging.info(line1a)
    logging.info(line1b)
    logging.info(line2)
    logging.info(line3)
    if checkpoint_time>0:
        logging.info(line4)
    logging.info(line5)

    # write file
    with open(f'{output_dir}/{prefix}_status.txt', 'wt') as handle:
        handle.write(line1a+'\n')
        handle.write(line1b + '\n')
        handle.write(line2+'\n')
        handle.write(line3+'\n')
        if checkpoint_time > 0:
            handle.write(line4+'\n')
        handle.write(line5+'\n')

def restore_best_scores(current_step, best_model_dir_mse, best_model_dir_ci, estimator, input_fn_dev, input_fn_tst):
    minmse_step=0
    minmse_dev=10000
    minmse_ci_dev=0
    minmse_mse_tst=10000
    minmse_ci_tst=0
    maxci_step=0
    maxci_dev=0
    maxci_mse_dev=10000
    maxci_mse_tst=10000
    maxci_ci_tst=0

    # mse
    checkpoint_file = "%s/checkpoint" % best_model_dir_mse
    if os.path.isfile(checkpoint_file):    #file_pathが存在するかどうかの確認
        with open(checkpoint_file, 'rt') as handle:
            line = handle.readline()    #fileの最初の行の読み取り
            best_model_prefix = re.findall(r'"(.*?)"', line)[0]    #クウォート内文章を読み取れる
        
        checkpoint_path = "%s/%s" % (best_model_dir_mse, best_model_prefix)
        eval_results = e


    # ci
    checkpoint_file = "%s/checkpoint" % best_model_dir_ci
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'rt') as handle:
            line = handle.readline()
            best_model_prefix = re.findall(r'"(.*?)"', line)[0]


def main(argv):
    del argv

    model = MbertPcnnModel()