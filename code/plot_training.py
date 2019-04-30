import matplotlib.pyplot as plt
import argparse
import pickle
import os
from operator import add
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hist_dir', dest='hist_dir', type=str, default='./training_history', required=False)
    return parser.parse_args()

def extract_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        v2v_hist = pickle.load(f)
        labeled_hist = pickle.load(f)
    return v2v_hist, labeled_hist 
    
def plot_hist(hist):
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

def plot_compare(hist_v2v, hist_labeled):
    plt.plot(hist_v2v)
    plt.plot(hist_labeled)
    plt.title('Test loss of progress')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim((0.3,0.7))
    plt.legend(['word2vec', 'labeled'], loc='upper right')
    plt.show()

def average(hist_dir):
    dir_list = os.listdir(hist_dir)
    ave_v2v = np.array([])
    ave_labeled = np.array([])
    for file in dir_list:
        filepath = os.path.join(hist_dir, file)
        v2v, labeled = extract_pickle(filepath)
        v2v_loss = np.array(v2v['val_loss'])
        labeled_loss = np.array(labeled['val_loss'])
        if ave_v2v.size == 0:
            ave_v2v = v2v_loss
            ave_labeled = labeled_loss
        else:
            ave_v2v = np.add(ave_v2v,v2v_loss)
            ave_labeled = np.add(ave_labeled,labeled_loss)
    ave_v2v /= len(dir_list)
    ave_labeled /= len(dir_list)
    return ave_v2v, ave_labeled

if __name__ == "__main__":
    args = get_args()
    v2v_hist, labeled_hist = average(args.hist_dir)
    plot_compare(v2v_hist,labeled_hist)
