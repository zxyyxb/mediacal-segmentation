# Copyright (c) MONAI Consortium

import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

import monai
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    AddChanneld,
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
)

import math


def main():
    # 这两句没啥用
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    # create a temporary directory and 40 random image, mask pairs
    print(f"loading data")
    # 数据格式为.npy，img和seg分别存储在不同的文件夹中，若在相同的文件夹，则根具命名规则进行导入
    imgdir = "/home/tione/notebook/model/data/monai/img/"
    segdir = "/home/tione/notebook/model/data/monai/seg/"
    images = sorted([(imgdir + s) for s in os.listdir(imgdir)])
    segs = sorted([(segdir + s) for s in os.listdir(segdir)])
    # 数据的80%用于训练，20%用于验证，可以进行修改
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[: math.ceil(0.8 * len(images))], segs[: math.ceil(0.8 * len(segs))])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[math.ceil(0.8 * len(images)):], segs[math.ceil(0.8 * len(segs)):])]


    # define transforms for image and segmentation
    # 这一部分可以根据实际情况增加图像增广的方法
    train_transforms = Compose(
        [
            # img和seg为数据地址，LoadImaged用于读取数据，可以直接读取dicom格式的数据
            LoadImaged(keys=["img", "seg"], image_only=True),
            # ScaleIntensityd用于对图像进行归一化，这里使用了默认参数
            ScaleIntensityd(keys="img"),
            # RandCropByPosNegLabeld的spatial_size参数可以根据实际数据大小进行修改，num_samples也可以进行修改
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=(48, 96, 96), pos=2, neg=1, num_samples=4
            ),
        ]
    )
    # val_transforms和train_transforms类似，但是不需要进行数据增广
    val_transforms = Compose(
        [
            # img和seg为数据地址，LoadImaged用于读取数据，可以直接读取dicom格式的数据
            LoadImaged(keys=["img", "seg"], image_only=True),
            # ScaleIntensityd用于对图像进行归一化，这里使用了默认参数
            ScaleIntensityd(keys="img"),
        ]
    )


    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        # batch_size根据机器显存大小进行修改
        # 如果batch_size是8，由于RandCropByPosNegLabeld的num_samples参数为8，所以每次训练会有8张图像输入（8 * 8）
        batch_size=8,
        shuffle=True,
        # 这一参数影响数据读取速度，更具机器性能进行修改
        num_workers=4,
        # collate_fn用于将数据转换为batch
        collate_fn=list_data_collate,
        # pin_memory用于将数据放入内存（非虚拟内存）中，读取速度更快
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    # val_loader与train_loader类似，但是不需要shuffle，batch_size为1
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

    # dice_metric用于计算dice系数
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # post_trans用于将输出的概率图转换为二值图（黑白，0/1）
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型可以修改
    model = monai.networks.nets.AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
    # 损失函数和优化器可以在之后进行修改
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                           cooldown=10, patience=10, verbose=True, min_lr=1e-8)


    # start a typical PyTorch training
    # 每val_interval个epoch进行一次验证
    val_interval = 4
    # 初始化最高的metric值和最高的metric值对应的epoch
    best_metric = -1
    best_metric_epoch = -1
    # 用于保存每一个epoch的loss值和每一个val的metric值，仿佛没用，可以试验一下删去后的效果
    epoch_loss_values = list()
    metric_values = list()
    # 使用tensorboard进行可视化
    # writer = SummaryWriter()
    # epoch个数和每个epoch的长度（向上取整）
    num_epochs = 100
    epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)
    # 开始训练
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

        # 验证
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    # sliding_window_inference用于将大图分割为小图进行预测，然后将小图拼接起来
                    # 这一做法的原因是当图像过大时，无法一次性放入显存中进行预测
                    # roi_size表示分割图像的大小，sw_batch_size表示预测的窗口数（即每次预测多少张小图，类似batch_size）
                    # 但是可能在拼接处出现明显的痕迹，可以将roi_size改为(-1, -1, -1)，即将整张图片放入预测，不进行拼接
                    roi_size = (48, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    # 将输出的概率图转换为二值图
                    # decollate_batch用于将batch中的数据拆分为单个数据，即将原来[B, C, H, W, D]的数据拆分为B个[C, H, W, D]的list
                    # 由于val_loader的batch_size为1，所以在这里没有作用
                    # 感觉这一步是多余的，可以试验一下去掉这一步的效果
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    # 之前定义过dice_metric，用于计算dice系数
                    # dice_metric可以计算多个batch的dice系数，而这里只有一个batch，所以直接计算
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                # 选择metric最好的模型进行保存
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation3d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    print("\n")
    print("*" * 10)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


if __name__ == "__main__":
        main()