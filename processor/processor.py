import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from utils.meter import AverageMeter
from utils.metrics import R1_mAP
from model.sync_batchnorm.replicate import *

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):


    """

    Args:
        training function inputs

        cfg: configuration file, passed from /config/configs.py or /default.py

        model: initialized deep model

        center_criterion: could be enabled if using center loss, implemented for further updating

        train_loader: training dataloader

        val_loader: validation dataloader

        optimizer: SGD or ADAM optimizer

        optimizer_center: SGD or ADAM optimizer  for center loss

        scheduler: updating scheduler

        loss_fn: loss function for learning

        num_query: number of learning samples

    Returns:

    """



    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            patch_replication_callback(model)
        model.to(device)

    Best_acc = 0
    Best_epoch = -1
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)


            cls_g,  cls_1, cls_2, cls_3, cls_4 = model(img, target,mode='train')

            #cls_g = model(img, target)

            # code for debug, could be modified latter
            cls_g = cls_g.to(device)
            cls_1 = cls_1.to(device)
            cls_2 = cls_2.to(device)
            cls_3 = cls_3.to(device)
            cls_4 = cls_4.to(device)

            loss_g = loss_fn(cls_g, cls_g, target)
            loss_1 = loss_fn(cls_1, cls_1, target)
            loss_2 = loss_fn(cls_2, cls_2, target)
            loss_3 = loss_fn(cls_3, cls_3, target)
            loss_4 = loss_fn(cls_4, cls_4, target)

            alpha = ((epoch + 1) / cfg.MAX_EPOCHS) * 1

            loss_p = (loss_1 + loss_2 + loss_3 + loss_4) / 10
            loss = loss_g + loss_p #* alpha
            loss.backward()



            #score, feat = model(img, target)
            #loss = loss_fn(score, feat, target)


            optimizer.step()
            if 'center' in cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (cls_g.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid,_) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)

                    cls_score= model(img,mode='test')
                    #cls_score = model(img, target)
                    #score, feat = model(img)
                    evaluator.update((cls_score, vid))

            if cfg.MODE == "validation_debug":
                # enable this if we have validation set or the test set can be taken as validation
                if acc >= Best_acc:
                    Best_epoch = epoch
                    Best_acc = acc
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_bestval.pth'))
            elif cfg.MODE == "customized_test":
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_current.pth'))


            logger.info("no enhancement Validation Results - Epoch: {}".format(epoch))
            logger.info("Using Validation samples: {}".format(count))
            logger.info("validation acc: {:.1%}".format(acc))



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing, using the original code without reconstruction")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       )
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model_dict = torch.load(cfg.TEST_WEIGHT, map_location=device)
    model.load_state_dict(model_dict,strict=False)


    model.eval()
    img_path_list = []
    for n_iter, (img, pid, imgpath) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img = img.to(device)

            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f= model(img,mode='test')
                    feat = feat + f
            else:
                feat = model(img,mode='test')

            evaluator.update((feat, pid))
            img_path_list.extend(imgpath)


    acc = evaluator.compute_acc()
    logger.info("acc: {:.1%}".format(acc))

