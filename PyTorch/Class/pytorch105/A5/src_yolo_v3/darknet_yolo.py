# Copyright (c) 2023 Aladdin Persson
# The following code is derived from the YOLOv3 implementation by Aladdin Persson available at
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLOv3/
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

# Scratch From: https://youtu.be/Grir6TZbc1M?si=hq_ksfv5FrESPseF
class ConvBNLeakReLu(nn.Module):
    """
    Convolutional block with Batch Normalization and LeakReLu Activation.

    This block applies a convolution followed by batch normalization with specified
    epsilon and momentum values, and then uses the LeakReLu activation function.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
    """

    def __init__(self, in_channels: int, out_channels: int, bn_act: bool = True, **kwargs):
        """
        Initializes the ConvBNLeakReLu block.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride of the convolution. Default: 1.
            padding (int, optional): Zero-padding added to both sides of the input. Default: 0.
        """
        super(ConvBNLeakReLu, self).__init__()

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the ConvBNLeakReLu block.

        Args:
            x (torch.Tensor): Input to the block.

        Returns:
            torch.Tensor: Output of the block.
        """
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Residual Block for a neural network.

    This block repeats a set of convolutional layers a specified number of times
    and optionally applies residual connections.

    Args:
        channels (int): Number of input and output channels.
        use_residual (bool, optional): Whether to apply residual connections. Default is True.
        num_repeats (int, optional): Number of times to repeat the convolutional layers. Default is 1.
    """

    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    ConvBNLeakReLu(channels, channels // 2, kernel_size=1),
                    ConvBNLeakReLu(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying convolutional layers.
        """
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x  # Return the final output


class ScalePrediction(nn.Module):
    """
    Scale Prediction Module for YOLO-like Object Detection.

    This module predicts bounding box scales and objectness scores for each grid cell.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of object classes to be detected.

    Attributes:
        pred (nn.Sequential): Sequential module for predictions.
        num_classes (int): Number of object classes.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Define the prediction layers using a Sequential module
        self.pred = nn.Sequential(
            ConvBNLeakReLu(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            ConvBNLeakReLu(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1)
        )

        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass of the ScalePrediction module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Reshaped and permuted output tensor representing predictions.
        """

        # Make predictions using the defined layers
        predictions = self.pred(x)

        # Reshape predictions to have the shape (batch_size, 3, num_classes + 5, height, width)
        # and permute dimensions for compatibility with loss calculations
        predictions = predictions.reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
        predictions = predictions.permute(0, 1, 3, 4, 2)

        return predictions



class YOLOv3(nn.Module):
    """
    YOLOv3 Object Detection Model.

    This module defines the architecture of the YOLOv3 model for object detection.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        num_classes (int): Number of object classes to be detected.

    Attributes:
        num_classes (int): Number of object classes.
        in_channels (int): Number of input channels.
        layers (nn.ModuleList): List of convolutional layers and blocks that define the model.
    """

    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        """
        Forward pass of the YOLOv3 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List of tensors representing predictions at different scales.
        """

        outputs = []  # for each scale
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        """
        Create the convolutional layers and blocks for the YOLOv3 model.

        Returns:
            nn.ModuleList: List of convolutional layers and blocks.
        """

        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module

                # Add ConvBNLeakReLu layer with specified parameters
                layers.append(
                    ConvBNLeakReLu(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]

                # Add ResidualBlock with specified number of repeats
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == "S":
                    # Add ScalePrediction layer
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        ConvBNLeakReLu(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    # Add Upsample layer
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")