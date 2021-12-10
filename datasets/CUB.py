from .bases import BaseImageDataset
import os.path as osp
import glob
import re
import os

class CUB(BaseImageDataset):
    def __init__(self, data_dir = 'data_dir', verbose = True):
        super(CUB, self).__init__()
        self.root = data_dir
        self.dataset_dir = data_dir
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))

        cluster_txt_file = open(os.path.join(self.root, 'cluster_id.txt'))

        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        data_len=None

        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])

        # for cub CLASS ID-1
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)

        cluster_list = []
        for line in cluster_txt_file:
            cluster_list.append(int(line[:-1].split(' ')[-1]))

        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        dataset_train=[]
        dataset_test=[]


        train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
        train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]

        train_cluster = [x for x in cluster_list][:data_len]


        for idx in range(len(train_label)):
            dataset_train.append((train_img[idx],train_label[idx]))


        test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
        test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

        # retaining the dataset format
        test_cluster = test_label

        for idx in range(len(test_label)):
            dataset_test.append((test_img[idx], test_label[idx]))

        self.train = dataset_train
        self.test = dataset_test

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

        if verbose:
            print("successful load CUB dataset!!")



    def _process_dir_old(self, data_dir, relabel=True):
        img_paths = glob.glob(osp.join(data_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #assert 0 <= pid <= 2501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset