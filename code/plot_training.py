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

def plot_compare(hist_v2v, hist_labeled, title,y_range):
    plt.plot(hist_v2v)
    plt.plot(hist_labeled)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(y_range)
    plt.legend(['word2vec', 'labeled'], loc='upper right')
    plt.show()

def average(hist_dir,hist_type):
    dir_list = os.listdir(hist_dir)
    ave_v2v = np.array([])
    ave_labeled = np.array([])
    for file in dir_list:
        filepath = os.path.join(hist_dir, file)
        v2v, labeled = extract_pickle(filepath)
        v2v_loss = np.array(v2v[hist_type])
        labeled_loss = np.array(labeled[hist_type])
        if ave_v2v.size == 0:
            ave_v2v = v2v_loss
            ave_labeled = labeled_loss
        else:
            ave_v2v = np.add(ave_v2v,v2v_loss)
            ave_labeled = np.add(ave_labeled,labeled_loss)
    ave_v2v /= len(dir_list)
    ave_labeled /= len(dir_list)
    return ave_v2v, ave_labeled

def average_accuary(hist_dir):
    dir_list = os.listdir(hist_dir)
    ave_v2v = 0
    ave_labeled = 0
    for file in dir_list:
        filepath = os.path.join(hist_dir, file)
        v2v, labeled = extract_pickle(filepath)
        ave_v2v += v2v['val_acc'][-1]
        ave_labeled += labeled['val_acc'][-1]
    ave_v2v /= len(dir_list)
    ave_labeled /= len(dir_list)
    return ave_v2v, ave_labeled

if __name__ == "__main__":
    args = get_args()
    v2v_loss, labeled_loss = average(args.hist_dir,'val_loss')
    v2v_acc, labeled_acc = average(args.hist_dir,'val_acc')
    ave_acc_v2v, ave_acc_labeled = average_accuary(args.hist_dir)
    print('History of the first experiment-------------------------')
    print("Average acc. of word2vec: {}".format(ave_acc_v2v))
    print("Average acc. of labeled: {}".format(ave_acc_labeled))
    plot_compare(v2v_loss,labeled_loss,'Loss of progress',(0.3,0.7))
    plot_compare(v2v_acc,labeled_acc,'Test accuracy of progress',(0.4,1))
   # print('History of a specific experiment experiment-------------------------')
   # v, l = extract_pickle('./training_history/training_hist0.pkl')
   # plot_compare(v['val_loss'],l['val_loss'],'Test loss of progress',(0.3,0.7))
   # plot_compare(v['val_acc'],l['val_acc'],'Test accuracy of progress',(0.4,1))
