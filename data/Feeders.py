import os
import torch

TSTframenum = torch.tensor([425,322,322,384,367,364,355,333,320,406,
            430,412,406,356,414,328,328,359,327,361,
            336,437,411,412,300,312,304,418,422,384,
            512,421,363,402,328,344,312,301,297,452,
            455,442,377,354,353,251,269,268,371,388,
            367,332,359,323,360,368,358,411,384,388])

class TST_Feeder_5F(torch.utils.data.Dataset):
    def __init__(self, datapath, datatype, fold):
        self.data_path = datapath
        self.data_type = datatype
        self.fold = fold
        self.load_data()

    def load_data(self):
        if self.data_type == 'train':
            self.skeleton = torch.load(os.path.join(self.data_path,r'fold{0}\train_skeleton_fold{0}.npy'.format(self.fold)))
            self.label = torch.load(os.path.join(self.data_path,r'fold{0}\train_label_fold{0}.pkl'.format(self.fold)))
            self.framescount = torch.load(os.path.join(self.data_path,r'fold{0}\train_fc_fold{0}.pkl'.format(self.fold)))
        elif self.data_type == 'val':
            self.skeleton = torch.load(os.path.join(self.data_path,r'fold{0}\test_skeleton_fold{0}.npy'.format(self.fold)))
            self.label = torch.load(os.path.join(self.data_path,r'fold{0}\test_label_fold{0}.pkl'.format(self.fold)))
            self.framescount = torch.load(os.path.join(self.data_path,r'fold{0}\test_fc_fold{0}.pkl'.format(self.fold)))

        self.N = self.skeleton.size(dim=0)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, item):
        skeleton = self.skeleton[item,:,:,:]
        labels = self.label[item,:]
        frames_count = self.framescount[item]

        # skeleton normalization
        # take middle point between left and right shoulders at the first frame as origin for each example
        left_shoulder_beacon = skeleton[0,4,:]
        right_shoulder_beacon = skeleton[0,8,:]
        beacon = (left_shoulder_beacon + right_shoulder_beacon) / 2

        skeleton = skeleton - beacon

        # data augmentation
        # IMPLEMENT HERE

        return skeleton, labels, frames_count


class STS_Feeder_5F(torch.utils.data.Dataset):
    def __init__(self, datapath, datatype, fold):
        self.data_path = datapath
        self.data_type = datatype
        self.fold = fold
        self.load_data()

    def load_data(self):
        if self.data_type == 'train':
            self.skeleton = torch.load(os.path.join(self.data_path,r'fold{0}\train_skeleton_fold{0}.npy'.format(self.fold)))
            self.labels_fine = torch.load(os.path.join(self.data_path,r'fold{0}\train_label_fine_fold{0}.pkl'.format(self.fold)))
            self.labels_coarse = torch.load(os.path.join(self.data_path, r'fold{0}\train_label_coarse_fold{0}.pkl'.format(self.fold)))
            self.frames_count = torch.load(os.path.join(self.data_path, r'fold{0}\train_fc_fold{0}.pkl'.format(self.fold)))
        elif self.data_type == 'val':
            self.skeleton = torch.load(os.path.join(self.data_path,r'fold{0}\test_skeleton_fold{0}.npy'.format(self.fold)))
            self.labels_fine = torch.load(os.path.join(self.data_path,r'fold{0}\test_label_fine_fold{0}.pkl'.format(self.fold)))
            self.labels_coarse = torch.load(os.path.join(self.data_path, r'fold{0}\test_label_coarse_fold{0}.pkl'.format(self.fold)))
            self.frames_count = torch.load(os.path.join(self.data_path, r'fold{0}\test_fc_fold{0}.pkl'.format(self.fold)))

        self.N = self.skeleton.size(dim=0)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, item):
        skeleton = self.skeleton[item,:,:,:]
        labels_fine = self.labels_fine[item,:]
        labels_coarse = self.labels_coarse[item,:]
        frames_count = self.frames_count[item]

        labels = torch.zeros([2,labels_coarse.size(0)])
        labels[0,:] = labels_fine
        labels[1,:] = labels_coarse

        # skeleton normalization
        neck_beacon = skeleton[0,2,:]
        skeleton = skeleton - neck_beacon

        return skeleton, labels, frames_count


class Asian_Feeder_10F(torch.utils.data.Dataset):
    def __init__(self, datapath, datatype, fold):
        self.data_path = datapath
        self.data_type = datatype
        self.fold = fold
        self.load_data()

    def load_data(self):
        if self.data_type == 'train':
            self.skeleton = torch.load(os.path.join(self.data_path,r'fold{0}\train_skeleton_fold{0}.npy'.format(self.fold)))
            self.labels = torch.load(os.path.join(self.data_path,r'fold{0}\train_label_fold{0}.pkl'.format(self.fold)))
            self.frames_count = torch.load(os.path.join(self.data_path, r'fold{0}\train_fc_fold{0}.pkl'.format(self.fold)))
        elif self.data_type == 'val':
            self.skeleton = torch.load(os.path.join(self.data_path,r'fold{0}\test_skeleton_fold{0}.npy'.format(self.fold)))
            self.labels = torch.load(os.path.join(self.data_path,r'fold{0}\test_label_fold{0}.pkl'.format(self.fold)))
            self.frames_count = torch.load(os.path.join(self.data_path, r'fold{0}\test_fc_fold{0}.pkl'.format(self.fold)))
        self.N = self.skeleton.size(dim=0)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, item):
        skeleton = self.skeleton[item,:,:,:]
        labels = self.labels[item,:]
        frames_count = self.frames_count[item]

        # skeleton normalization
        spine_beacon = skeleton[0,9,:]
        skeleton = skeleton - spine_beacon

        return skeleton, labels, frames_count


'''=============================================    OLD    ============================================='''
class TST_Feeder(torch.utils.data.Dataset):
    def __init__(self, datapath, datatype):
        self.data_path = datapath
        self.data_type = datatype
        self.load_data()

    def load_data(self):
        if self.data_type == 'train':
            self.skeleton = torch.load(os.path.join(self.data_path,'train_skeleton.npy'))
            self.label = torch.load(os.path.join(self.data_path,'train_label.pkl'))
        elif self.data_type == 'val':
            self.skeleton = torch.load(os.path.join(self.data_path,'test_skeleton.npy'))
            self.label = torch.load(os.path.join(self.data_path,'test_label.pkl'))
        else:
            self.skeleton = torch.load(os.path.join(self.data_path,'skeleton.npy'))
            self.label = torch.load(os.path.join(self.data_path,'label.pkl'))

        self.N = self.skeleton.size(dim=0)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, item):
        skeleton = self.skeleton[item,:,:,:]
        labels = self.label[item,:]

        if self.data_type == 'train':
            frames_count = TSTframenum[item]
        elif self.data_type == 'val':
            frames_count = TSTframenum[item+45]

        # skeleton normalization
        # take middle point between left and right shoulders at the first frame as origin for each example
        left_shoulder_beacon = skeleton[0,4,:]
        right_shoulder_beacon = skeleton[0,8,:]
        beacon = (left_shoulder_beacon + right_shoulder_beacon) / 2

        skeleton = skeleton - beacon

        # data augmentation
        # IMPLEMENT HERE

        return skeleton, labels, frames_count



class STS_Feeder(torch.utils.data.Dataset):
    def __init__(self, datapath, datatype):
        self.data_path = datapath
        self.data_type = datatype
        self.load_data()

    def load_data(self):
        if self.data_type == 'train':
            self.skeleton = torch.load(os.path.join(self.data_path,'skeleton_train.npy'))
            self.labels_fine = torch.load(os.path.join(self.data_path,'labels_fine_train.pkl'))
            self.labels_coarse = torch.load(os.path.join(self.data_path, 'labels_coarse_train.pkl'))
            self.frames_count = torch.load(os.path.join(self.data_path, 'frames_count_train.pkl'))
        elif self.data_type == 'val':
            self.skeleton = torch.load(os.path.join(self.data_path,'skeleton_val.npy'))
            self.labels_fine = torch.load(os.path.join(self.data_path,'labels_fine_val.pkl'))
            self.labels_coarse = torch.load(os.path.join(self.data_path, 'labels_coarse_val.pkl'))
            self.frames_count = torch.load(os.path.join(self.data_path, 'frames_count_val.pkl'))

        self.N = self.skeleton.size(dim=0)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, item):
        skeleton = self.skeleton[item,:,:,:]
        labels_fine = self.labels_fine[item,:]
        labels_coarse = self.labels_coarse[item,:]
        frames_count = self.frames_count[item]

        labels = torch.zeros([2,labels_coarse.size(0)])
        labels[0,:] = labels_fine
        labels[1,:] = labels_coarse

        # skeleton normalization
        neck_beacon = skeleton[0,2,:]
        skeleton = skeleton - neck_beacon

        return skeleton, labels, frames_count


class Asian_Feeder(torch.utils.data.Dataset):
    def __init__(self, datapath, datatype):
        self.data_path = datapath
        self.data_type = datatype
        self.load_data()

    def load_data(self):
        if self.data_type == 'train':
            self.skeleton = torch.load(os.path.join(self.data_path,'skeleton_train.npy'))
            self.labels = torch.load(os.path.join(self.data_path,'label_train.pkl'))
            self.frames_count = torch.load(os.path.join(self.data_path, 'framescount_train.pkl'))
        elif self.data_type == 'val':
            self.skeleton = torch.load(os.path.join(self.data_path,'skeleton_test.npy'))
            self.labels = torch.load(os.path.join(self.data_path,'label_test.pkl'))
            self.frames_count = torch.load(os.path.join(self.data_path, 'framescount_test.pkl'))
        self.N = self.skeleton.size(dim=0)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, item):
        skeleton = self.skeleton[item,:,:,:]
        labels = self.labels[item,:]
        frames_count = self.frames_count[item]

        # skeleton normalization
        spine_beacon = skeleton[0,9,:]
        skeleton = skeleton - spine_beacon

        return skeleton, labels, frames_count