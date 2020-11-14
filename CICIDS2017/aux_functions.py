import tensorflow as tf
import numpy as np
import os
import sys
import time
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Model
import pandas as pd



class DataHandler():
    def __init__(self,dataset,labels,weights,batch_size = 64,shuffle=False):
        self.dataset = dataset
        self.current = 0
        self.len = len(dataset)
        self.batch_size = batch_size
        self.labels = labels
        self.do_shuffle = shuffle
        self.inds = np.arange(self.len)
        self.weights = weights
        assert self.len>=batch_size
        assert len(labels)==len(dataset)
        if self.do_shuffle:
            self.shuffle()
    def shuffle(self):
        if self.do_shuffle:
            p = np.random.permutation(len(self.dataset))
            self.inds = p
    def next_batch(self,batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        current_inds = self.inds[self.current:self.current+batch_size]
        batch = self.dataset[current_inds]
        y_batch = self.labels[current_inds]
        if self.weights is not None:
            w_batch = self.weights[current_inds]
        self.current +=batch_size
        if self.current>=self.len:
            new_inds = self.inds[:batch_size-len(batch)]
            batch = np.concatenate((batch,self.dataset[new_inds]))
            y_batch = np.concatenate((y_batch,self.labels[new_inds]))
            if self.weights is not None:
                w_batch =  np.concatenate((w_batch,self.weights[new_inds]))
            self.current=0
            self.shuffle()
        if self.weights is not None:
            return batch,y_batch,w_batch
        return batch,y_batch
    
class PacketModel(Model):
    def __init__(self):
        super(PacketModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(2048, activation='relu')
        self.d2 = Dense(2)

    def call(self, x):
        x = self.flatten(x)
        dense1_out = self.d1(x)
        dense2_out = self.d2(dense1_out)
        return dense2_out
    
def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    optimizer.lr.assign(lr)
    
class Solver():
    def __init__(self,optimizer,net,base_lr):
        self.iters = 0
        self.optimizer = optimizer
        self.net = net
        self.base_lr = base_lr
    def update_lr(self):
        adjust_learning_rate_inv(self.base_lr,self.optimizer,self.iters)
        

def get_files(day,prefix = '../datasets/CICIDS2017_packet-based/'):
    all_files = []
    prefix = prefix+day
    for file in os.listdir(prefix):
        if file.endswith(".npy") and file.startswith('part'):
            all_files.append(os.path.join(prefix, file))
    all_files = sorted(all_files)
    return all_files


def preproces_dataset(src_day,trg_day,dataset_path):
    
    def get_ds(day):
        timesteps = 20
        num_input = 29
        all_files = get_files(day)
        x_test = []
        for f in all_files:
            print (f)
            x_test.append(np.load(f))
        x_test = np.concatenate(x_test,axis=0)

        a = timesteps -  len(x_test) % timesteps
        temp = x_test[:a]
        x_test = np.concatenate((x_test,temp),axis=0)
        x_test = x_test.reshape(-1,timesteps*num_input)
        x_test = x_test.astype(np.float32)
        yt = np.load(dataset_path+day+'/labels.npy')
        a = timesteps -  len(yt) % timesteps
        temp = yt[:a]
        y_test = np.concatenate((yt,temp),axis=0)
        y_test = y_test.reshape(-1,timesteps)
        y_test = y_test[:,-1]
        return x_test,y_test

    x_src, y_src = get_ds(src_day)
    x_trg, y_trg = get_ds(trg_day)
    
    real_labels = y_src!=-1
    x_src = x_src[real_labels]
    y_src = y_src[real_labels]

    real_labels = y_trg!=-1
    x_trg = x_trg[real_labels]
    y_trg = y_trg[real_labels]
    
    x_all = np.concatenate((x_src,x_trg),axis=0)
    train_min = np.min(x_all,axis=0)
    train_max = np.max(x_all,axis=0)

    x_src  = (x_src - train_min)/(train_max - train_min + 1e-6)
    x_trg  = (x_trg - train_min)/(train_max - train_min + 1e-6)
    
    y_src[y_src!=0] = 1
    y_trg[y_trg!=0] = 1
    
    return x_src,y_src,x_trg,y_trg,train_min,train_max

