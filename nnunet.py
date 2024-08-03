import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

# 导入NNUNet相关模块和类
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss

# 导入MONAI相关模块和类
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd
from monai.losses import DiceLoss

class YourDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        # 实现数据加载逻辑
        # 返回图像和标签的文件列表
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 加载单个数据样本并应用转换
        image = np.load(self.images[index])
        label = np.load(self.labels[index])

        if self.transform:
            transformed = self.transform(image=image, label=label)
            image = transformed["image"]
            label = transformed["label"]

        return {
            "image": image,
            "label": label
        }

# 创建NNUNet模型
class SimpleNNUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleNNUNet, self).__init__()
        # 使用NNUNet的网络结构
        self.model = Generic_UNet(
            input_channels=in_channels,
            base_num_features=64,
            num_classes=num_classes,
            num_pool=4,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
        )

    def forward(self, x):
        return self.model(x)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False),
    AddChanneld(keys=["image", "label"]),
    ScaleIntensityd(keys="image"),
    # 可添加更多的数据增强或预处理方法
])


train_dataset = YourDataset(data_dir="your_data_dir", transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 创建NNUNet模型
model = SimpleNNUNet(in_channels=1, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, {})
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建NNUNet训练器
trainer = nnUNetTrainer(
    network=model,
    optimizer=optimizer,
    loss_function=criterion,
    device=device,
    input_data_is_spatial_2D=False,  # 根据您的数据维度设置
)

# 开始训练
num_epochs = 10  # 设置训练轮数
for epoch in range(num_epochs):
    # 在每个epoch中迭代训练数据
    for batch_data in train_loader:
        inputs, labels = batch_data['image'], batch_data['label']
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播和计算损失
        predictions = trainer.network(inputs)
        loss = trainer.loss_function(predictions, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        print(f'Epoch [{epoch + 1}/{num_epochs}], Batch Loss: {loss.item():.4f}')


