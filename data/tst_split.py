# Split data for five-fold cross validation

import torch
import numpy as np

TSTframenum = torch.tensor([425,322,322,384,367,364,355,333,320,406,
            430,412,406,356,414,328,328,359,327,361,
            336,437,411,412,300,312,304,418,422,384,
            512,421,363,402,328,344,312,301,297,452,
            455,442,377,354,353,251,269,268,371,388,
            367,332,359,323,360,368,358,411,384,388])

skeleton = torch.load(r'...\TUG-EDGCN\data\TST_TUG\raw\skeleton.npy')
labels = torch.load(r'...\TUG-EDGCN\data\TST_TUG\raw\labels.pkl')

idx = np.arange(60)
np.random.shuffle(idx)

for i in range(0,5):
    startidx = i * 10
    endidx = startidx + 18
    train_index = np.concatenate((idx[0:startidx], idx[endidx:]))
    test_index = idx[startidx:endidx]

    skeleton_train = skeleton[train_index,:,:,:]
    skeleton_test = skeleton[test_index,:,:,:]
    label_train = labels[train_index,:]
    label_test = labels[test_index,:]
    fc_train = TSTframenum[train_index]
    fc_test = TSTframenum[test_index]

    torch.save(skeleton_train,r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold{0}\train_skeleton_fold{1}.npy'.format(i,i))
    torch.save(skeleton_test,r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold{0}\test_skeleton_fold{1}.npy'.format(i,i))
    torch.save(label_train,r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold{0}\train_label_fold{1}.pkl'.format(i,i))
    torch.save(label_test,r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold{0}\test_label_fold{1}.pkl'.format(i,i))
    torch.save(fc_train,r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold{0}\train_fc_fold{1}.pkl'.format(i,i))
    torch.save(fc_test,r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold{0}\test_fc_fold{1}.pkl'.format(i,i))


print("========================Finished========================")