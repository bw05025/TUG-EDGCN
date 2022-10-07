# Split data for five-fold cross validation

import torch
import numpy as np


skeleton = torch.load(r'...\TUG-EDGCN\data\STS\raw\skeleton.npy')
labels_fine = torch.load(r'...\TUG-EDGCN\data\STS\raw\labels_fine.pkl')
labels_coarse = torch.load(r'...\TUG-EDGCN\data\STS\raw\labels_coarse.pkl')
frames_count = torch.load(r'...\TUG-EDGCN\data\STS\raw\frame_count.pkl')

idx = np.arange(217)
np.random.shuffle(idx)

for i in range(0,5):
    startidx = i * 38
    endidx = startidx + 65
    train_index = np.concatenate((idx[0:startidx], idx[endidx:]))
    test_index = idx[startidx:endidx]

    skeleton_train = skeleton[train_index,:,:,:]
    skeleton_test = skeleton[test_index,:,:,:]
    label_fine_train = labels_fine[train_index,:]
    label_fine_test = labels_fine[test_index,:]
    label_coarse_train = labels_coarse[train_index,:]
    label_coarse_test = labels_coarse[test_index,:]
    fc_train = frames_count[train_index]
    fc_test = frames_count[test_index]

    torch.save(skeleton_train,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\train_skeleton_fold{1}.npy'.format(i,i))
    torch.save(skeleton_test,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\test_skeleton_fold{1}.npy'.format(i,i))
    torch.save(label_fine_train,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\train_label_fine_fold{1}.pkl'.format(i,i))
    torch.save(label_fine_test,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\test_label_fine_fold{1}.pkl'.format(i,i))
    torch.save(label_coarse_train,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\train_label_coarse_fold{1}.pkl'.format(i,i))
    torch.save(label_coarse_test,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\test_label_coarse_fold{1}.pkl'.format(i,i))
    torch.save(fc_train,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\train_fc_fold{1}.pkl'.format(i,i))
    torch.save(fc_test,r'...\TUG-EDGCN\data\STS\Five_fold\fold{0}\test_fc_fold{1}.pkl'.format(i,i))


print("========================Finished========================")