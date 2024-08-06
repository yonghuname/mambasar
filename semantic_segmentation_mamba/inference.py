import sys
from torch.utils.data import DataLoader
from Utils.data_loading import BasicDataset
import logging
from Utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score,JaccardIndex
from rs_mamba_ss import RSM_SS
from BaseUnet import UNet
from tqdm import tqdm


def train_net(dataset_name, load_checkpoint=True):
    # 1. Create dataset

    test_dataset = BasicDataset(images_dir=f'./{dataset_name}/test/image/',
                                labels_dir=f'./{dataset_name}/test/label/',
                                train=False)
    # 2. Create data loaders
    # 2. Create data loaders
    loader_args = dict(num_workers=8,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=1, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Using device {device}')

    # net = RSM_SS(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank, \
    #            ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version, patchembed_version=ph.patchembed_version)

    net =UNet(n_channels=4,n_classes=1,bilinear=True)
    net.to(device=device)

    assert ph.load, 'Loading model error, checkpoint ph.load'
    load_model = torch.load(ph.load, map_location=device)
    if load_checkpoint:
        net.load_state_dict(load_model)
    logging.info(f'Model loaded from {ph.load}')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    metric_collection = MetricCollection({
        'miou': JaccardIndex(num_classes=2).to(device=device),
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device)
    })  # metrics calculator

    net.eval()
    logging.info('SET model mode to test!')

    with torch.no_grad():
        for batch_img1, labels, name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            labels = labels.float().to(device)

            ss_preds = net(batch_img1)
            ss_preds = torch.sigmoid(ss_preds)

            # Calculate and log other batch metrics
            ss_preds = ss_preds.float()
            labels = labels.int().unsqueeze(1)
            metric_collection.update(ss_preds, labels)

            # clear batch variables from memory
            del batch_img1, labels

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

    print('over')


if __name__ == '__main__':

    try:
        train_net(dataset_name='SAR_Dataset_CD')
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)
