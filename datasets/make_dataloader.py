from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .UnifiedLoader import UnifiedLoader
from .CUB import CUB
from .NAbirds import NAbirds
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler,ClusterIdentitySampler


def train_collate_fn(batch):
    """
    collate_fn for training input
    """
    imgs, labels, _, = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)

    return torch.stack(imgs, dim=0), labels



def val_collate_fn(batch):
    """
        collate_fn for validation input
    """

    imgs, labels, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), labels,  img_paths

def make_dataloader(cfg):

    """
    # the data augmentation are not carefully modified, other hyper-params may lead to higher performance

    uncomment line for other dataset

    dataset = CUB(data_dir=cfg.DATA_DIR, verbose=True)

    #dataset = UnifiedLoader(dataset_name='Aircraft',data_dir=None,verbose=True)

    #dataset = NAbirds(data_dir='/media/space/ZYF/Dataset/Other/NAbirds/', verbose=True)
    """

    train_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),


        T.RandomCrop([448, 448]),
        T.RandomRotation(12, resample=Image.BICUBIC, expand=False, center=None),
        #T.ColorJitter(brightness=0.2, contrast=0.2), # used in other code

        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0),
                        T.RandomAffine(degrees=0, translate=None, scale=[0.8, 1.2], shear=15, \
                                       resample=Image.BICUBIC, fillcolor=0)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.CenterCrop([448, 448]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = cfg.DATALOADER_NUM_WORKERS



    #
    dataset = CUB(data_dir=cfg.DATA_DIR, verbose=True)
    #dataset = UnifiedLoader(dataset_name='StanfordCars',data_dir=None,verbose=True)
    #dataset = NAbirds(data_dir='/media/space/ZYF/Dataset/Other/NAbirds/', verbose=True)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.group_wise:
        print('using multi-attention training')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.BATCH_SIZE,
                                  num_workers=num_workers,
                                  sampler=RandomIdentitySampler(dataset.train, cfg.BATCH_SIZE, cfg.NUM_IMG_PER_ID),
                                  collate_fn=train_collate_fn  # customized batch sampler
                                  )
    elif cfg.group_wise==False:
        print('using baseline training')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  sampler=None,
                                  collate_fn=train_collate_fn,  # customized batch sampler
                                  drop_last=True
                                  )
    else:
        print('unsupported training strategy!   got {} for co-attention training'.format(cfg.CO_ATT))

    val_set = ImageDataset(dataset.test, val_transforms)
    val_loader = DataLoader(val_set,
                            batch_size=cfg.TEST_IMS_PER_BATCH,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=val_collate_fn
                            )
    return train_loader, val_loader, len(dataset.test), num_classes
