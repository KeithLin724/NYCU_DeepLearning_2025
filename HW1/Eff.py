import math
import torch
import torch.nn as nn
import lightning as L


# -----------------------------------------------
# 輔助函數：對通道數與重複次數進行縮放（參考 EfficientNet 論文）
def round_filters(filters, width_mult: float, divisor=8):
    filters *= width_mult
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_mult):
    return int(math.ceil(repeats * depth_mult))


# -----------------------------------------------
# 根據版本選擇各項參數（EfficientNet 論文中的參數）
def get_efficientnet_params(version: str):
    params = {
        "B0": {"width_mult": 1.0, "depth_mult": 1.0, "resolution": 224, "dropout": 0.2},
        "B1": {"width_mult": 1.0, "depth_mult": 1.1, "resolution": 240, "dropout": 0.2},
        "B2": {"width_mult": 1.1, "depth_mult": 1.2, "resolution": 260, "dropout": 0.3},
        "B3": {"width_mult": 1.2, "depth_mult": 1.4, "resolution": 300, "dropout": 0.3},
        "B4": {"width_mult": 1.4, "depth_mult": 1.8, "resolution": 380, "dropout": 0.4},
        "B5": {"width_mult": 1.6, "depth_mult": 2.2, "resolution": 456, "dropout": 0.4},
        "B6": {"width_mult": 1.8, "depth_mult": 2.6, "resolution": 528, "dropout": 0.5},
        "B7": {"width_mult": 2.0, "depth_mult": 3.1, "resolution": 600, "dropout": 0.5},
    }
    return params[version]


# -----------------------------------------------
# Squeeze-and-Excitation 模組
class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.fc1(se), inplace=True)
        se = torch.sigmoid(self.fc2(se))
        return x * se


# -----------------------------------------------
# MBConv 模組（帶 SE）
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion
        layers = []
        # 擴展層（1x1 conv）
        if expansion != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU(inplace=True))
        # Depthwise 卷積
        layers.append(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU(inplace=True))
        # SE 模組
        layers.append(SEModule(hidden_dim))
        # Projection 層（1x1 conv）
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)

        return self.block(x)


# -----------------------------------------------
# EfficientNet 主網路：可設定 version 從 B0 到 B7
class EfficientNetVariant(L.LightningModule):
    def __init__(
        self,
        version: str = "B0",
        num_classes: int = 1000,
        ce_weights: list[float] = 0,
        t_max: int = 150,
    ) -> None:
        super().__init__()
        self.ce_weights = torch.tensor(ce_weights) if ce_weights is not None else None
        self.t_max = t_max

        params = get_efficientnet_params(version)
        width_mult = params["width_mult"]
        depth_mult = params["depth_mult"]
        dropout_rate = params["dropout"]

        # Stem 卷積
        out_channels = round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Baseline MBConv 配置（參照 EfficientNet-B0）
        self.cfgs = [
            [1, 16, 1, 3, 1],
            [6, 24, 2, 3, 2],
            [6, 40, 2, 5, 2],
            [6, 80, 3, 3, 2],
            [6, 112, 3, 5, 1],
            [6, 192, 4, 5, 2],
            [6, 320, 1, 3, 1],
        ]
        layers = []
        in_channels = out_channels
        for exp, c, n, k, s in self.cfgs:
            out_channels = round_filters(c, width_mult)
            repeats = round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                layers.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        stride=stride,
                        expansion=exp,
                    )
                )
                in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

        # Head 卷積（最後一層）
        head_channels = round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(head_channels, num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    # @torch.autocast(device_type="cuda")
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.ce_weights = self.ce_weights.to(y.device)

        loss = F.cross_entropy(y_hat, y, weight=self.ce_weights)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()

        self.log_dict({"train_loss": loss, "train_acc": acc})
        return loss

    # @torch.autocast(device_type="cuda")
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.ce_weights = self.ce_weights.to(y.device)

        loss = F.cross_entropy(y_hat, y, weight=self.ce_weights)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean()

        self.log_dict({"val_loss": loss, "val_acc": acc})
        return

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        # optimizer = optim.RMSprop(self.parameters(), lr=0.01, weight_decay=0.9, momentum=0.9)
        # optimizer = optim.AdamW(self.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=1e-6)

        # # 原始 SGD 優化器設定
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

        # # StepLR 每 30 epoch 衰減 0.1 倍學習率
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
