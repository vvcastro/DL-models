from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential([
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]) 
    
    def forward(self, x):
        return self.conv_block(x)

class DownConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        feature_map = self.conv_block(x)
        x = self.pooling(x)
        return x, feature_map

class UpConvolution(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=(2, 2), align_corners=True)
        self.conv = nn.Conv2d(channels, channels // 2, kernel_size=2, padding=1, dilation=2)
    
    def forward(self, x):
        upsampled_x = self.upsampling(x)
        x = self.conv(upsampled_x)
        return x

class ContractingPath(nn.Module):
    def __init__(self, hidden_channels, input_channels=3):
        super().__init__()

        encoder_blocks = []
        for ch in hidden_channels:
            encoder_blocks.append(DownConvolution(input_channels, ch))
            input_channels = ch
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

    def forward(self, x):
        features_map = []
        for conv_block in self.encoder_blocks:
            x, block_feature_map = conv_block(x)
            features_map.append(block_feature_map)
        return block_feature_map[:-1][::-1], x

class ExpandingPath(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        expading_blocks, upsampling_convs = [], []
        for i in range(len(channels) - 1):
            upsampling_convs.append(UpConvolution(channels[i]))
            expading_blocks.append(ConvBlock(channels[i], channels[i + 1]))
        self.upsamplings_convs = nn.ModuleList(upsampling_convs)
        self.expading_blocks = nn.ModuleList(expading_blocks)

    def forward(self, x, enc_features):
        blocks = zip(self.upsamplings_convs, self.expading_blocks, enc_features)
        for upsample_conv, block_conv, enc_feature in blocks:
            x = upsample_conv(x)
            x = self.crop_and_concat(x, enc_feature)
            x = block_conv(x)
        return x

    def crop_and_concat(self, x, enc_feature):
        diff_y = enc_feature.size()[2] - x.size()[2]
        diff_x = enc_feature.size()[3] - x.size()[3]

        div_diff_x = torch.div(diff_x, 2, rounding_mode="floor")
        div_diff_y = torch.div(diff_y, 2, rounding_mode="floor")

        x = F.pad(x, [ div_diff_x, diff_x - div_diff_x, div_diff_y, diff_y - div_diff_y])
        concat = torch.cat([x, enc_feature], dim=1)
        return concat


class UNet(pl.LightningModule):
    def __init__(self, hidden_channels, output_size, num_clases=1, example=None):
        super().__init__()
        self.output_size = output_size

        self.encoder = ContractingPath(hidden_channels)
        self.decoder = ExpandingPath(hidden_channels[::-1])
        self.head = nn.Conv2d(hidden_channels[0], num_clases, kernel_size=1)

        self.loss_funct = nn.BCEWithLogitsLoss()
        self.example_input_array = example
        
    def forward(self, x):
        enc_features, dec_input = self.encoder(x)
        dec_output = self.decoder(dec_input, enc_features)
        raw_map = self.head(dec_output)
        map = F.interpolate(raw_map, self.output_size)
        return map

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_map = self(x)
        loss = self.loss_funct(pred_map, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred_map = self(x)
        test_loss = self.loss_funct(pred_map, y)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_map = self(x)
        valid_loss = self.loss_funct(pred_map, y)
        self.log("val_loss", valid_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
