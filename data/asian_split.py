'''
"Leave-one-subject-out" ten-fold cross validation.
30 videos by 10 actors in Asian-TUG dataset, each actor 3.
For each fold, use 3 videos from one actor as testing set, the others as training set.
The raw skeleton data is arranged by actors, sequentially.
'''


import torch
import numpy as np

skeleton = torch.load(r'...\TUG-EDGCN\data\Asian_TUG\raw\skeleton.npy')
labels = torch.load(r'...\TUG-EDGCN\data\Asian_TUG\raw\label.pkl')
frames_count = torch.load(r'...\TUG-EDGCN\data\Asian_TUG\raw\framescount.pkl')

idx = np.arange(30)

for i in range(0,10):
    startidx = i * 3
    endidx = startidx + 3
    train_index = np.concatenate((idx[0:startidx], idx[endidx:]))
    test_index = idx[startidx:endidx]

    skeleton_train = skeleton[train_index,:,:,:]
    skeleton_test = skeleton[test_index,:,:,:]
    label_train = labels[train_index,:]
    label_test = labels[test_index,:]
    fc_train = frames_count[train_index]
    fc_test = frames_count[test_index]

    torch.save(skeleton_train,r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold\fold{0}\train_skeleton_fold{1}.npy'.format(i,i))
    torch.save(skeleton_test,r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold\fold{0}\test_skeleton_fold{1}.npy'.format(i,i))
    torch.save(label_train,r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold\fold{0}\train_label_fold{1}.pkl'.format(i,i))
    torch.save(label_test,r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold\fold{0}\test_label_fold{1}.pkl'.format(i,i))
    torch.save(fc_train,r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold\fold{0}\train_fc_fold{1}.pkl'.format(i,i))
    torch.save(fc_test,r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold\fold{0}\test_fc_fold{1}.pkl'.format(i,i))


print("========================Finished========================")