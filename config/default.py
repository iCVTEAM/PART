# coding=utf-8
"""
It is NOT RECOMMENDED for developers to modify the base class directly.
Instead, they should re derive a new configuration class in configs.py
"""
class DefaultConfig:
    """ Base configuration class for perparameter settings.
    All new configuration should be derived from this class.

    Note that these hyper parameters and not carefully tuned and can be used for all datasets. simply tuning them would lead to higher performance

    we report the mean acc but not the highest values during testing for its stability

    PROJECT_NAME: project name

    LOG_DIR: log directory

    OUTPUT_DIR: saved model directory

    DEVICE_ID: GPU IDs, i.e. "0,1,2" for multiple GPUs

    LOG_PERIOD = 50 : iteration of displaying training log

    CHECKPOINT_PERIOD  : saving model period

    EVAL_PERIOD = 10 : validation period

    MAX_EPOCHS = 200 :max training epochs

    DATA_DIR : dataset path

    DATALOADER_NUM_WORKERS : number of dataloader workers

    GROUP_SAMPLING = 'on'  : enable for the group-wise learning

    SAMPLER = 'triplet'： batch sampler, option: 'triplet','softmax'

    BATCH_SIZE  : MxN, M: number of classes, N: number of images of per class

    NUM_IMG_PER_ID : N, number of images of per class

    INPUT_SIZE ： HxW set as 600*600 in other work for high performance

    MODEL_NAME ：backbone name, option: 'resnet50',

    LAST_STRIDE : the stride of the last layer of resnet50

    PRETRAIN_CHOICE : 'imagenet', load image net pretraining  model

    PRETRAIN_PATH : pretrained weight path, should be automatically downloaded if not specify

    LOSS_TYPE: option: 'triplet+softmax','softmax+center','triplet+softmax+center'

    LOSS_LABELSMOOTH : using labelsmooth, option: 'on', 'off'

    COS_LAYER : using cosface for learning default False

    OPTIMIZER = SGD / Adam

    BASE_LR :base learning rate

    CE_LOSS_WEIGHT :weight of softmax loss

    TRIPLET_LOSS_WEIGHT: setting for improve performance default:unused

    CENTER_LOSS_WEIGHT : setting for improve performance default:unused


    CENTER_LR： learning rate for the weights of center loss,setting for improve performance default:unused

    MARGIN ： triplet loss margin, setting for improve performance default:unused

    self.STEPS ：learning rate decay steps, can be modified

    self.GAMMA ： decay factor of learning rate

    self.WARMUP_EPOCHS ： warm up epochs, can be disabled

    self.WARMUP_METHOD ： option: 'linear','constant'
    """

    def __init__(self):
        self.MODE = "validation_debug"  # option validation_debug, customized_test

        self.PROJECT_NAME = 'Transpart'  # project name
        self.LOG_DIR = "./log"  # log directory
        self.OUTPUT_DIR = "./output"  # saved model directory
        self.DEVICE_ID = "0,1"  # GPU IDs, i.e. "0,1,2" for multiple GPUs

        self.LOG_PERIOD = 50  # iteration of displaying training log
        self.CHECKPOINT_PERIOD = 10  # saving model period
        self.EVAL_PERIOD = 10  # validation period
        self.MAX_EPOCHS = 250  # max training epochs

        # data
        self.DATA_DIR = "None"  # dataset path
        self.DATALOADER_NUM_WORKERS = 4  # number of dataloader workers
        self.SAMPLER = 'triplet'
        self.BATCH_SIZE = 16#16
        self.NUM_IMG_PER_ID = 4#4

        # model
        self.INPUT_SIZE = [512, 512]  # HxW set as 600*600 NTS
        self.MODEL_NAME = "resnet101"  # backbone name, option: 'resnet50',
        self.LAST_STRIDE = 1  # the stride of the last layer of resnet50
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = "/xxx/pretrained_model/resnet50-19c8e357.pth"  # pretrained weight path

        # loss
        self.LOSS_TYPE = 'softmax'
        self.LOSS_LABELSMOOTH = 'on' #'on'  # using labelsmooth, option: 'on', 'off'
        self.COS_LAYER = False

        # solver
        self.group_wise = True
        #self.OPTIMIZER = 'Adam'  # optimizer

        self.OPTIMIZER = 'SGD'
        self.BASE_LR = 0.0008  # 0.0008  # base learning rate
        self.CE_LOSS_WEIGHT = 1.0  # weight of softmax loss
        self.TRIPLET_LOSS_WEIGHT = 1.0  # weight of triplet loss
        self.CENTER_LOSS_WEIGHT = 0.0005  # weight of center loss

        self.HARD_FACTOR = 0.0  # harder example mining

        self.WEIGHT_DECAY = 0.0005  # 0.0005
        self.BIAS_LR_FACTOR = 1.0
        self.WEIGHT_DECAY_BIAS = 0.0005  # 0.0005
        self.MOMENTUM = 0.9
        self.CENTER_LR = 0.5  # learning rate for the weights of center loss
        self.MARGIN = 0.3  # triplet loss margin

        # self.STEPS = [30, 60, 80] # for NAbirds
        self.STEPS = [60, 120, 160]  # learning rate decay epochs


        self.NUM_SELECT_PART = 4 # select how many parts?
        self.NUM_PART_STACK = 64 # select how many candidates for part selection

        self.GAMMA = 0.1  # decay factor of learning rate
        self.WARMUP_FACTOR = 0.01
        self.WARMUP_EPOCHS = 10  # warm up epochs
        self.WARMUP_METHOD = "linear"  # option: 'linear','constant'

        # configuration for test script
        self.TEST_IMS_PER_BATCH = 16
        self.FEAT_NORM = "yes"
        # self.TEST_WEIGHT = './output/resnet50_180.pth'
        self.FLIP_FEATS = 'off'  # using fliped feature for testing, option: 'on', 'off'

