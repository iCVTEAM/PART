from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reading all dataset
    """

    def get_imagedata_info(self, data):
        pids = []
        for _, pid in data:
            pids += [pid]

        pids = set(pids)


        num_pids = len(pids)
        num_imgs = len(data)
        return num_pids, num_imgs

    def get_imagedata_info_discrete(self, data):
        pids = []
        for _, pid in data:
            pids += [pid]

        pids = set(pids)


        max_pid = max(pids)
        pids_hash = np.zeros((max_pid+1,),dtype =np.int)

        num_pids = len(pids)
        num_imgs = len(data)

        count = 0
        for idx in pids:
            pids_hash[idx] = count
            count= count+1


        return num_pids, num_imgs,pids_hash



    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of Imagereading dataset
    """

    def print_dataset_statistics(self, train,val):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)
        num_val_pids, num_val_imgs = self.get_imagedata_info(val)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # classes | # images ")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  val    | {:5d} | {:8d}".format(num_train_pids, num_val_imgs))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, img_path
