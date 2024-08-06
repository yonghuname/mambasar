import numpy as np
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from osgeo import gdal
import wandb
import logging

from Utils.path_hyperparameter import ph

def save_model(model, path, epoch, mode, optimizer=None):
    """
    当在评估中出现最佳指标时保存最佳模型
    或者在评估中每个指定的间隔保存检查点n

    参数:
        model(class): 我们构建的神经网络
        path(str): 模型保存路径
        epoch(int): 模型保存时的训练轮次
        mode(str): 确保是保存最佳模型还是检查点，应为 checkpoint、loss 或 f1score
        optimizer(class, optional): 训练的优化器，在保存检查点时需要

    返回:
        无返回值
    """
    # 创建保存模型的目录
    Path(path).mkdir(parents=True, exist_ok=True)
    # 获取当前的本地时间并将其转换为一个可读的字符串格式
    localtime = time.asctime(time.localtime(time.time()))
    if mode == 'checkpoint':  # 如果是 checkpoint 模式
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict, str(path + f'checkpoint_epoch{epoch}_{localtime}.pth'))
    else:  # 保存最佳模型
        torch.save(model.state_dict(), str(path + f'best_{mode}_epoch{epoch}_{localtime}.pth'))
    logging.info(f'best {mode} model {epoch} saved at {localtime}!')

def train_val_test(
        mode, dataset_name,
        dataloader, device, log_wandb, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_loss_model_path=None, best_f1_model_path=None,non_improved_epoch=None
):
    """
    在指定数据集上进行训练或评估，
    注意参数 [warmup_lr, grad_scaler] 在训练中是必需的，
    参数 [best_metrics, checkpoint_path, best_loss_model_path, non_improved_epoch] 在评估中是必需的

    参数:
        mode(str): 确保是训练还是评估，应为 train 或 val
        dataset_name(str): 指定数据集的名称
        dataloader(class): 对应模式和指定数据集的数据加载器
        device(str): 模型运行设备
        log_wandb(class): 用于记录超参数、指标和输出的类
        net(class): 我们构建的神经网络
        optimizer(class): 训练的优化器
        total_step(int): 训练步骤
        lr(float): 学习率
        criterion(class): 损失函数
        metric_collection(class): 指标计算器
        to_pilimg(function): 将数组转换为图像的函数
        epoch(int): 训练轮次
        warmup_lr(list, optional): 在预热阶段对应步骤的学习率
        grad_scaler(class, optional): 使用混合精度时缩放梯度
        best_metrics(list, optional): 评估中的最佳指标
        checkpoint_path(str, optional): 检查点保存路径
        best_loss_model_path(str, optional): 最佳损失模型保存路径
        non_improved_epoch(int, optional): 最佳指标未提高的持续轮次

    返回:
        在不同模式下返回不同的修改后的输入参数，
        当 mode = train 时，
        返回 log_wandb, net, optimizer, grad_scaler, total_step, lr
        当 mode = val 时，
        返回 log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch
    """
    # 确保 mode 参数的值为 'train' 或 'val'
    assert mode in ['train', 'val'], 'mode should be train, val'
    epoch_loss = 0
    if mode == 'train':
        net.train()  # 设置模型为训练模式
    else:
        net.eval()  # 设置模型为评估模式
    logging.info(f'SET model mode to {mode}!')
    batch_iter = 0

    # 使用 tqdm 显示进度条
    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter)

    # 定义 sample_name 以记录样本图像名称
    sample_name = None

    for i, (image, labels, name) in enumerate(tbar):
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
        batch_iter = batch_iter + ph.batch_size
        total_step += 1

        if mode == 'train':
            optimizer.zero_grad()
            # if total_step < ph.warm_up_step:     需要预热的话，，删掉注释
            #     for g in optimizer.param_groups:
            #         g['lr'] = warmup_lr[total_step]

        image = image.float().to(device)
        labels = labels.float().to(device)

        b, c, h, w = image.shape
        # 下采样图像和标签
        image = F.interpolate(image, size=(h // ph.downsample_raito, w // ph.downsample_raito), mode='bilinear', align_corners=False)
        labels = F.interpolate(labels.unsqueeze(1), size=(h // ph.downsample_raito, w // ph.downsample_raito), mode='bilinear', align_corners=False).squeeze(1)

        if mode == 'train':
            with torch.cuda.amp.autocast():
                preds = net(image)
                #loss_change, diceloss, bceloss = criterion(preds, labels)
                bceloss = criterion(preds, labels)
            #cd_loss = loss_change.mean()
            cd_loss = bceloss.mean()
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds = net(image)
            #loss_change, diceloss, bceloss = criterion(preds, labels)
            bceloss = criterion(preds, labels)
            cd_loss = bceloss.mean() # 是当前批次的损失，2张图片，总的评价指标不看这个

        epoch_loss += cd_loss
        preds = torch.sigmoid(preds)

        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=image.shape[0])
            t1_img_log = torch.round(image[sample_index]).cpu().clone().float()
            label_log = torch.round(labels[sample_index]).cpu().clone().float()
            pred_log = torch.round(preds[sample_index]).cpu().clone().float()
            sample_name = name[sample_index]  # 记录当前样本图像的名称

        batch_metrics = metric_collection(preds.float(), labels.int().unsqueeze(1))

        # 记录批次的度量指标
        log_wandb.log({
            f'{mode} loss': cd_loss,
            f'{mode} accuracy': batch_metrics['accuracy'],
            f'{mode} precision': batch_metrics['precision'],
            f'{mode} recall': batch_metrics['recall'],
            f'{mode} f1score': batch_metrics['f1score'],
            f'{mode} miou': batch_metrics['miou'],
            'learning rate': optimizer.param_groups[0]['lr'],
           # f'{mode} loss_dice': diceloss,
            f'{mode} loss_bce': bceloss,
            'step': total_step,
            'epoch': epoch
        })

        del image, labels

    epoch_metrics = metric_collection.compute()
    epoch_loss /= n_iter

    # 打印并记录每个 epoch 的度量指标(这个是我们需要关注的)
    print(f"{mode} miou: {epoch_metrics['miou']}")
    print(f"{mode} accuracy: {epoch_metrics['accuracy']}")
    print(f"{mode} precision: {epoch_metrics['precision']}")
    print(f"{mode} recall: {epoch_metrics['recall']}")
    print(f"{mode} f1score: {epoch_metrics['f1score']}")
    print(f'{mode} epoch loss: {epoch_loss}')

    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{str(k)}': epoch_metrics[k], 'epoch': epoch})
    metric_collection.reset()
    log_wandb.log({f'epoch_{mode}_loss': epoch_loss, 'epoch': epoch})

    # 在这里记录图像名称
    log_wandb.log({
        f'{mode} t1_images {sample_name}': wandb.Image(t1_img_log),  # 记录样本图像名称
        f'{mode} masks': {
            f'label {sample_name}': wandb.Image(to_pilimg(label_log)),
            f'pred {sample_name}': wandb.Image(to_pilimg(pred_log)),
        },
        'epoch': epoch
    })

    if mode == 'val':  # 验证集下
        if epoch_loss < best_metrics['lowest_loss']:  # 如果当前 epoch 的损失低于之前的最佳损失，则更新最佳损失
            best_metrics['lowest_loss'] = epoch_loss  # 更新 best_metrics 中的 lowest loss
            best_metrics['best_epoch_loss'] = epoch  # 记录最佳 epoch
            if ph.save_best_model:  # 如果设置了保存最佳模型，则调用 save_model 函数保存当前模型
                save_model(net, best_loss_model_path, epoch, 'loss')

        # 根据 F1 分数来保存最佳模型
        if epoch_metrics['f1score'] > best_metrics['highest_f1']:  # 如果当前 epoch 的 F1 分数高于之前的最佳 F1 分数，则更新
            best_metrics['highest_f1'] = epoch_metrics['f1score']  # 更新 best_metrics 中的最高 F1 分数
            best_metrics['best_epoch_f1'] = epoch  # 记录最佳 F1 分数的 epoch
            if ph.save_best_model:  # 如果设置了保存最佳模型，则调用 save_model 函数保存当前模型
                save_model(net, best_f1_model_path, epoch, 'f1score')

        # else:  # 如果当前 epoch 的性能没有改善，增加 non_improved_epoch 的计数   需要预热的话，，删掉注释
        #     non_improved_epoch += 1
        #     if non_improved_epoch == ph.patience:  # 类似早停策略
        #         lr *= ph.factor  # 将学习率乘以一个因子 ph.factor，减小学习率
        #         for g in optimizer.param_groups:
        #             g['lr'] = lr
        #         non_improved_epoch = 0

        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)

    if mode == 'train':
        return log_wandb, net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        return log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch
