class Path_Hyperparameter:
    random_seed = 42  # 随机种子

    # 训练超参数
    epochs: int = 200  # 训练轮数
    batch_size: int =  16  # 批次大小
    inference_ratio = 1 # 验证和测试时的批次大小为训练批次大小的倍数
    learning_rate: float = 1e-4  # 学习率
    factor = 0.1  # 学习率衰减因子
    patience = 12  # 学习率调度器的耐心值
    warm_up_step = 1000  # 预热步数。通过在训练初期逐步增加学习率，可以提高训练的稳定性和模型的收敛速度，这里原来是是1000
    weight_decay: float = 1e-3  # AdamW优化器的权重衰减
    amp: bool = True  # 是否使用混合精度，以加速训练过程并减少 GPU 内存占用
    load: str = None #'./SAR_BaseUnet_train_Baseline_newdata_256_0.0001_best_loss_model/best_loss_epoch24_Mon Aug  5 13:15:38 2024.pth'  # 从 .pth 文件加载模型和/或优化器，便在测试或继续训练时恢复之前的状态
    max_norm: float = 20  # 梯度裁剪的最大范数。限制梯度的最大范数，以防止梯度爆炸

    # 评估和测试超参数
    evaluate_epoch: int = 0  # 训练多少轮后开始评估。即前面不做任何的评估，  默认 0
    evaluate_inteval: int = 1  # 每多少轮进行一次评估
    test_epoch: int = 101  # 训练多少轮后开始测试
    stage_epoch = [0, 0, 0, 0, 0]  # 每个阶段后调整学习率
    save_checkpoint: bool = False  # 是否保存模型检查点
    save_interval: int = 50  # 每多少轮保存一次检查点50
    save_best_model: bool = True  # 是否保存最佳模型

    # 模型超参数（mamba模型的）
    # RSM_SS tiny
    drop_path_rate = 0  # Drop path 比率。通过随机地丢弃网络中的一些路径，可以使模型在每次前向传播中使用不同的路径组合，从而迫使模型学习更鲁棒的特征
    # dims = 96  # 维度？
    dims = [96, 192, 384, 768]
    depths = [2, 2, 9, 2]  # 每个阶段的深度,原本是2292，加深后效果增加不显著并且训练机器极其慢
    ssm_d_state = 16  # SSM 状态维度
    ssm_dt_rank = "auto"  # SSM Dt 排名。Dt 矩阵的秩决定了状态转移过程中可用的独立信息量。用于指定或自动确定状态空间模型中 Dt 矩阵的秩
    ssm_ratio = 2.0  # SSM 比率。用于控制模型中不同部分的比例关系，可以增强模型的表示能力，同时控制计算复杂度
    mlp_ratio = 4.0  # MLP 比率。多层感知器（MLP）是由多个全连接层（也称为线性层）组成的。mlp_ratio 参数用于控制隐藏层的宽度（神经元数量）相对于输入层宽度的比例
    downsample_version = "v3"
    patchembed_version = "v2" #指将输入图像划分为若干小块（patches），然后将这些小块映射到高维空间

    # 数据参数
    image_size = 256  # 图像大小
    downsample_raito = 1  # 通过下采样来减少图像的分辨率，从而降低计算复杂度和内存占用。downsample_raito 为 1 表示不进行下采样
    dataset_name = 'SAR_Dataset_CD'  # 数据集名称
    root_dir = '.'  # 数据集的根目录（当前路径）

    # 推理参数
    log_path = './log_feature/'  # 日志路径n

    # log wandb 超参数
    # 在终端中运行这段代码，可以在浏览器中查看相关的训练指标图像等： wandb sync <wandb_directory>
    # log_wandb_project: str = 'train_whu_cd'  # wandb 项目名称
    log_wandb_project: str = 'SAR_OSMUnet_train_Baseline_newdata'  # wandb 项目名称（Weights and Biases，录模型的超参数、训练过程中的指标（如损失、准确率）、模型权重和输出日志）

    project_name = f'{log_wandb_project}_{image_size}_{learning_rate}'  # 项目名称

    #将 Path_Hyperparameter 类的所有属性及其值提取出来，并返回一个字典。这种方法常用于保存和加载模型的状态
    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}
        # 返回一个字典，包含所有不以'_'开头的属性及其值

ph = Path_Hyperparameter()
# 创建 Path_Hyperparameter 类的实例
