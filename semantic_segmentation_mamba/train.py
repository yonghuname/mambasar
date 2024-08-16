import sys  # 系统特定参数和函数
import time  # 时间相关函数
# import ipdb  # 交互式 Python 调试器
import torch  # PyTorch 主库
import numpy as np  # 数值计算库
from torch import optim  # PyTorch 的优化器模块
import torchvision.transforms as T  # 计算机视觉数据处理库
from torch.utils.data import DataLoader  # 数据加载和批处理工具
from Utils.data_loading import BasicDataset  # 数据集加载工具
from Utils.path_hyperparameter import ph  # 路径和超参数配置
from Utils.losses import FCCDN_loss_without_seg  # 损失函数
import os  # 操作系统相关函数
import logging  # 日志记录模块
import random  # 随机数生成模块
import wandb  # Weights and Biases，实验跟踪和可视化工具
from rs_mamba_ss import RSM_SS  # 变化检测模型（mamba模型 ）
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, JaccardIndex  # 评估指标工具
from Utils.utils import train_val_test # 训练和验证函数
from puremambaunet2 import mygoUNet2,RSM_SS2hw
from BaseUnet import UNet
from mambaunet5deep import  mambaunet5deepnet
# from wyxmambaunet import wyxMambaUetwithattention
from res4deeppuremambaunet import res4deepMambaunetv1
from res5deepmambaunet import  res5deepMambaunetv1
from res4mambaunetmore import res4deepMambaunetv2,res4deepMambaunetwithmultihead

# 使用随机种子可以确保每次运行代码时生成的随机数序列相同，从而得到相同的结果
def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 设置随机种子并启动训练过程，并且在用户中断时进行适当的处理
def auto_experiment():
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)

# 训练和验证模型的主要函数
def train_net(dataset_name):
    # 创建训练和验证数据集
    train_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/train/image/',
                                 labels_dir=f'{ph.root_dir}/{dataset_name}/train/label/',
                                 train=True)
    val_dataset = BasicDataset(images_dir=f'{ph.root_dir}/{dataset_name}/val/image/',
                               labels_dir=f'{ph.root_dir}/{dataset_name}/val/label/',
                               train=False)

    # 标记数据集大小
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 创建数据加载器
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, batch_size=2, **loader_args)

    # 初始化日志记录
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    log_wandb = wandb.init(project=ph.log_wandb_project, resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=hyperparameter_dict, mode='offline')
    os.environ["WANDB_DIR"] = f"./{ph.log_wandb_project}"

    # 记录关键配置信息
    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
    ''')

    # 设置模型、优化器、预热调度器、学习率调度器、损失函数和其他东西
    # net= mygoUNet2(dims=ph.dims)
    # net = RSM_SS2hw(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,
    #              ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version,
    #              patchembed_version=ph.patchembed_version)
    # net = mambaunet5deepnet(depths=[3, 4, 4,4 ,4],      dims=[96, 192, 384, 768,1536], ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,
    #              ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version,
    #              patchembed_version=ph.patchembed_version)
    # net = res5deepMambaunetv1(depths=[3, 4, 4,4 ,4],      dims=[96, 192, 384, 768,1536], ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,
    #              ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version,
    #              patchembed_version=ph.patchembed_version)

    # net = wyxMambaUetwithattention(  num_classes=1,   in_channels=4,     img_size=256,      embed_dim=768,      depth=12,     patch_size=16  )
    #
    # net = res4deepMambaunetv1(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,  ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version,     patchembed_version=ph.patchembed_version)
    # net = res4deepMambaunetv2(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,  ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version, patchembed_version=ph.patchembed_version)
    net = res4deepMambaunetwithmultihead(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,  ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version, patchembed_version=ph.patchembed_version)

    # net = RSM_SS(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank,  ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version,  patchembed_version=ph.patchembed_version)
    # net = UNet(n_channels=4,n_classes=1,bilinear=True)  # 使用 BaseUNet（经典Unet模型）






    net = net.to(device=device)
    # optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate, weight_decay=ph.weight_decay) # AdamW 优化器

    optimizer = optim.Adam(net.parameters(), lr=ph.learning_rate)

    # warmup_lr = np.arange(1e-7, ph.learning_rate, (ph.learning_rate - 1e-7) / ph.warm_up_step) # 预热（建议不适用）
    grad_scaler = torch.cuda.amp.GradScaler()

    # 使用余弦退火学习率调度器（推荐）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ph.epochs, eta_min=ph.learning_rate * 0.01)

    # 加载模型和优化器
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0
    lr = ph.learning_rate
    criterion = FCCDN_loss_without_seg

    # 初始化最佳指标
    best_metrics = {
        'lowest_loss': float('inf'),
        'best_epoch_loss': -1,
        'highest_f1': 0.0,  # 初始化最高 F1 分数为 0
        'best_epoch_f1': -1  # 初始化最佳 F1 分数的 epoch
    }
    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device),
        'miou': JaccardIndex(num_classes=2).to(device=device)
    })

    to_pilimg = T.ToPILImage()  # 用于将张量（Tensor）或 ndarray（NumPy 数组）转换为 PIL 图像。在media中可以查看

    # 模型保存路径
    checkpoint_path = f'./{ph.project_name}_checkpoint/'
    best_loss_model_path = f'./{ph.project_name}_best_loss_model/'
    best_f1_model_path = f'./{ph.project_name}_best_f1_model/'
    non_improved_epoch = 0

    for epoch in range(ph.epochs):
        print('Start Train!')

        # 训练阶段
        log_wandb, net, optimizer, grad_scaler, total_step, lr = train_val_test(
            mode='train', dataset_name=dataset_name,
            dataloader=train_loader, device=device, log_wandb=log_wandb, net=net,
            optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
            metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
            warmup_lr=None, grad_scaler=grad_scaler
        )

        # 更新学习率
        scheduler.step()



        # 验证阶段
        if (epoch + 1) >= ph.evaluate_epoch and (epoch + 1) % ph.evaluate_inteval == 0:
            print('Start Validation!')

            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch  = train_val_test(
                    mode='val', dataset_name=dataset_name,
                    dataloader=val_loader, device=device, log_wandb=log_wandb, net=net,
                    optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                    metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                    best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                    best_loss_model_path=best_loss_model_path,best_f1_model_path=best_f1_model_path,
                    non_improved_epoch=non_improved_epoch
                )

    # 在最后一次训练后保存最佳模型
    print(f'Best model at epoch {best_metrics["best_epoch"]} with lowest loss {best_metrics["lowest_loss"]}')
    wandb.finish()

if __name__ == '__main__':
    auto_experiment()
