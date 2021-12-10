from .bases import BaseImageDataset
import os.path as osp
import glob
import re
import os

class NAbirds(BaseImageDataset):
    def __init__(self, data_dir = 'data_dir', verbose = True):
        super(NAbirds, self).__init__()
        self.root = data_dir
        self.dataset_dir = data_dir
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))



        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))

        data_len = None
        '''
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ').split('/')[-1])
        '''

        img_list = []
        label_list = []
        for line in label_txt_file:
            tmp_name =line[:-1].split(' ')[0]
            img_list.append(tmp_name.replace('-',''))
            label_list.append(int(line[:-1].split(' ')[-1]))


        train_test_list = []
        for line in train_val_file:
            # maintaining the img name in traintest list is consistency with the img_list
            train_test_list.append(int(line[:-1].split(' ')[-1]))


        train_file_list = [x for i, x in zip(train_test_list, img_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_list) if not i]

        dataset_train=[]
        dataset_test=[]


        train_img = [os.path.join(self.root, 'images_all', train_file+'.jpg') for train_file in
                              train_file_list[:data_len]]
        train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]


        ### prepossessing labels
        pre_labels = []
        for idx in range(len(train_label)):
            pre_labels.append((train_img[idx],train_label[idx]))

        self.num_train_pids, self.num_train_imgs, pid_hash = self.get_imagedata_info_discrete(pre_labels)


        ## end of prepossessing labels

        for idx in range(len(train_label)):
            dataset_train.append((train_img[idx],pid_hash[train_label[idx]]))


        test_img = [os.path.join(self.root, 'images_all', test_file+'.jpg') for test_file in
                             test_file_list[:data_len]]
        test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

        # retaining the dataset format


        for idx in range(len(test_label)):
            dataset_test.append((test_img[idx], pid_hash[test_label[idx]]))

        self.train = dataset_train
        self.test = dataset_test

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

        if verbose:
            print("successful load NABirds dataset!!")
            print("Using transformed labels!")



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