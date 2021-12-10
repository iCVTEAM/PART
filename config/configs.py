from .default import DefaultConfig


class Config(DefaultConfig):
    """
    must set data dir here

    self.DATA_DIR = 'path to your dir'

    self.PRETRAIN_PATH = 'path to your pretrained resnet 50 model'

    self.CLASSNUM  training class number, default set CUB as 200

    self.TEST_WEIGHT  testing weight set here

    self.TRAIN_BN_MOM BatchNorm momentum

    self.HARD_FACTOR not used

    self.LOSS_TYPE set your loss function here, default set softmax means softmax cross entropy

    """
    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'model1'
        self.DATA_DIR = '/media/space/zhaoyf/FGVC/Data/CUB_200_2011'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = '/root/.torch/models/resnet50-19c8e357.pth'

        #self.LOSS_TYPE = 'triplet+softmax+center'
        self.LOSS_TYPE = 'softmax'
        self.TEST_WEIGHT = '/media/space/zhaoyf/FGVC/TIP/TransPart-Res50models/4-part-final-cub-89.6/resnet50_250-CUB-89.6+.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.TRAIN_BN_MOM = 0.1
        self.RERANKING = True


