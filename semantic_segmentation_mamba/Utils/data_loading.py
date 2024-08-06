import logging  # 用于日志记录
from os import listdir  # 用于文件操作
from os.path import splitext  # 用于文件操作
from pathlib import Path  # 用于处理文件路径
import random  # 用于随机操作
import numpy as np  # 用于数组操作
from osgeo import gdal  # 用于处理图像文件
from torch.utils.data import Dataset  # pytorch中的基本数据集类
import albumentations as A  # 用于数据增强
from albumentations.pytorch import ToTensorV2  # 将图像转换为张量


class BasicDataset(Dataset):  # BasicDataset 类用于训练、评估和测试的数据集
    """ Basic dataset for train, evaluation and test.

    Attributes:
        images_dir(str): path of images. 图像路径
        labels_dir(str): path of labels. 标签路径
        train(bool): ensure creating a train dataset or other dataset. 确定是创建训练数据集还是其他数据集
        ids(list): name list of images. 图像名称列表
        train_transforms_all(class): data augmentation applied to image and label. 应用于图像和标签的数据增强
    """

    def __init__(self, images_dir: str, labels_dir: str, train: bool):
        """ Init of basic dataset.

        Parameter:
            images_dir(str): path of images.
            labels_dir(str): path of labels.
            train(bool): ensure creating a train dataset or other dataset.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # image name without suffix (通过遍历 images_dir 目录下的所有文件，获取去除文件扩展名后的文件名列表)
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()  # 对文件名列表进行排序

        # 如果图像ID列表为空，抛出一个运行时错误，提示用户在指定的目录中没有找到图像文件
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # 定义数据增强和转换
        #self.train_transforms_all = A.Compose([
        #    A.Flip(p=0.5),  # 以50%的概率水平翻转图像
        #    A.Transpose(p=0.5),  # 以50%的概率转置图像
        #], additional_targets={'image1': 'image'})  # 不仅对主图像进行相应的增强操作还要对其对应的标签图像进行一样的操作.

        # 定义了一个标准化操作，将图像的像素值从[0, 255]范围转换为更适合用于训练神经网络的范围。修改输入图片通道数在这里改
        self.normalize = A.Compose([
            A.Normalize(mean=(0.0, 0.0, 0.0,0.0), std=(1.0, 1.0, 1.0,1.0))
        ])

        # 将图像数据转换为适合用于 PyTorch 深度学习模型的输入，即(C,H,W)->(通道数,高度,宽度)
        self.to_tensor = A.Compose([
            ToTensorV2()
        ])

    # 返回数据集的长度，即数据集中样本的数量
    def __len__(self):
        """ Return length of dataset."""
        return len(self.ids)  # self.ids 是一个包含所有图像文件名（去掉后缀）的列表

    # label 数组中所有非零的元素都设置为1
    @classmethod  # @classmethod 使得方法可以在类本身上调用，而不需要实例化类。通常命名为 cls
    def label_preprocess(cls, label):
        """ Binaryzation label."""
        label[label != 0] = 1
        return label

    # 这段代码定义了一个类方法 load，用于打开图像文件并将其转换为 NumPy 数组
    @classmethod
    def load(cls, filename):
        """Open image and convert image to array using GDAL."""
        try:
            filename = str(filename)  # 确保 filename 是一个字符串
            dataset = gdal.Open(filename)  # 打开图像文件
            if dataset is None:
                raise ValueError(f"Cannot open image: {filename}")

            bands = []
            for band_index in range(1, dataset.RasterCount + 1):  # 获取所有波段的数据
                band = dataset.GetRasterBand(band_index)
                band_data = band.ReadAsArray()
                bands.append(band_data)

            img = np.stack(bands, axis=-1)  # 将所有波段数据沿新的最后一个轴堆叠成多通道图像
            img = img.astype(np.uint8)  # 转换为 uint8 类型数组
            return img
        except Exception as e:
            logging.error(f"Error loading image {filename}: {e}")
            raise

    def __getitem__(self, idx):
        """
        索引数据集。

        索引图像名称列表以获取图像名称，根据名称在图像路径中搜索图像，
        打开图像并将其转换为数组。

        预处理数组，对其应用数据增强和噪声添加（可选），然后将数组转换为张量。

        参数:
            idx(int): 数据集的索引。

        返回:
            tensor(tensor): 图像的张量。
            label_tensor(tensor): 标签的张量。
            name(str): 图像和标签相同的名称。
        """
        name = self.ids[idx]  # Name 是数据集中第 idx 个样本的文件名（不包括扩展名）

        # img_file 和 label_file 是与该文件名匹配的图像和标签文件的完整路径。glob 是 pathlib.Path 类的方法，用于基于通配符模式匹配文件路径。它返回一个生成器，生成匹配模式的文件路径
        # 通配符 .* 的含义是匹配任意扩展名的文件
        img_file = list(self.images_dir.glob(name + '.*'))
        label_file = list(self.labels_dir.glob(name + '.*'))

        # 确保每个图像和标签文件唯一
        assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {name}: {label_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # 分别加载唯一的图像和标签文件，并对加载后的标签进行二值化处理，将非零值转换为1
        img = self.load(img_file[0])
        label = self.load(label_file[0])
        label = self.label_preprocess(label)

        # 在训练模式下对图像和标签应用数据增强操作
        #if self.train:  # self.train 是一个布尔值属性
        #   sample = self.train_transforms_all(image=img, mask=label)
        #    img, label = sample['image'], sample['mask']

        # 对图像进行标准化和转换为张量格式
        img = self.normalize(image=img)['image']
        sample = self.to_tensor(image=img, mask=label)
        tensor, label_tensor = sample['image'].contiguous(), sample['mask'].contiguous()

        # 修改部分：去除 labels 的最后一个维度
        if label_tensor.shape[-1] == 1:
            label_tensor = label_tensor.squeeze(-1)

        return tensor, label_tensor, name  # 分别返回对原始图像经过张量处理后的图像和标签，以及图像和标签的文件名
