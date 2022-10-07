import torch
import numpy as np


skeleton = torch.load(r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold0\test_skeleton_fold0.npy')
label = torch.load(r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold0\test_label_fold0.pkl')
frames_count = torch.load(r'...\TUG-EDGCN\data\TST_TUG\Five_fold\fold0\test_fc_fold0.pkl')

test_sk = skeleton[0,:,0,0].numpy()
test_lb = label[0,:].numpy()
test_fc = frames_count[0].numpy()

print("end")

