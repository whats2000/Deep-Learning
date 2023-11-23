import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode, Resize


# Copyright (c) 2023 Ultralytics LLC
# Adapted from the YOLOv5 architecture documentation by Ultralytics available at
# https://docs.ultralytics.com/yolov5/tutorials/architecture_description/
#
# This adapted code is based on the descriptions and concepts provided in the YOLOv5 documentation,
# which are part of the broader YOLOv5 project. The use of this adapted code should comply with the
# terms and conditions set forth by Ultralytics for the use of their materials and intellectual property.
#
# Ultralytics provides detailed documentation and tutorials on the YOLOv5 architecture, which serves as
# a guideline for understanding and implementing the model. This adapted code is a result of interpreting
# and coding the concepts described in their official documentation.
class ConvBNSiLu(nn.Module):
    """
    Convolutional block with Batch Normalization and SiLU Activation.

    This block applies a convolution followed by batch normalization with specified
    epsilon and momentum values, and then uses the SiLU activation function.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the ConvBNAct block.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolution kernel.
            stride (int, optional): Stride of the convolution. Default: 1.
            padding (int, optional): Zero-padding added to both sides of the input. Default: 0.
        """
        super(ConvBNSiLu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBNAct block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolution, batch normalization, and activation function.
        """
        # Apply the convolution, batch normalization, and activation function
        out = self.conv(x)

        return out


class BottleNeck(nn.Module):
    """
    Generalized bottleneck block used in YOLOv5 that can operate with or without a skip connection.

    The block uses two convolutional layers, each followed by batch normalization and SiLU activation.
    If skip_connection is set to True, the output of the second convolutional layer is added to the block's input.
    """

    def __init__(self, channels, width_multiplier=1.0, skip_connection=True):
        """
        Initializes the Bottleneck block.

        Args:
            channels (int): Number of channels in the input and output. This remains unchanged throughout the block.
            width_multiplier (float): Multiplier for the internal channels. Default: 1.0.
            skip_connection (bool): Whether to use a skip connection. Default: True.
        """
        super(BottleNeck, self).__init__()
        hidden_channels = int(channels * width_multiplier)

        # The first convolutional layer
        self.conv1 = ConvBNSiLu(channels, hidden_channels, kernel_size=1, stride=1, padding=0)

        # The second convolutional layer
        self.conv2 = ConvBNSiLu(hidden_channels, channels, kernel_size=3, stride=1, padding=1)

        # Whether to use a skip connection
        self.skip_connection = skip_connection

    def forward(self, x):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the bottleneck block operations.
        """
        # First convolutional layer
        out = self.conv1(x)

        # Second convolutional layer
        out = self.conv2(out)

        # Add the skip connection if specified
        if self.skip_connection:
            out += x

        return out


class C3(nn.Module):
    """
    CSP block (C3) used in YOLOv5 architecture, allowing for different behaviors in the backbone.
    """

    def __init__(self, in_channels, out_channels, number_of_bottlenecks, width_multiplier=0.5, is_backbone=True):
        """
        Initializes the C3 block.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output tensor after concatenation and the final ConvBNSiLu layer.
            number_of_bottlenecks (int): The depth of bottleneck blocks to apply to part 'a'.
            width_multiplier (float): Determines the scaling factor for the channel width in the bottleneck blocks.
            is_backbone (bool): Indicates whether the block is part of the backbone and should use skip connections in the bottlenecks.
        """
        super(C3, self).__init__()
        hidden_channels = int(in_channels * width_multiplier)

        # The first convolutional layer for a path 'a'
        self.conv1 = ConvBNSiLu(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)

        # The second convolutional layer for path 'b'
        self.conv2 = ConvBNSiLu(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)

        # Final convolutional layer after concatenation
        self.final_conv = ConvBNSiLu(2 * hidden_channels, out_channels, kernel_size=1, stride=1,
                                     padding=0)

        # Bottleneck layers for path 'a'
        self.bottlenecks = nn.Sequential(
            *[BottleNeck(hidden_channels, width_multiplier=1,
                         skip_connection=is_backbone) for _ in range(number_of_bottlenecks)]
        )

    def forward(self, x):
        """
        Forward pass of the C3 block.
        """
        # Apply the first convolutional layers to get 'a' and 'b'
        a = self.conv1(x)
        b = self.conv2(x)

        # Apply the bottleneck blocks to 'a'
        a = self.bottlenecks(a)

        # Concatenate 'a' and 'b'
        concatenated = torch.cat((a, b), dim=1)

        # Apply the final convolutional layer
        out = self.final_conv(concatenated)

        return out


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) block used to aggregate context features at different scales.
    """

    def __init__(self, in_channels):
        """
        Initializes the SPPF block.

        Args:
            in_channels (int): Number of channels in the input tensor.
        """
        super(SPPF, self).__init__()

        # Define the downsample convolution
        self.downsample = ConvBNSiLu(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)

        # Define max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        # Define the final convolution layer that restores the channel dimension
        self.final_conv = ConvBNSiLu(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass of the SPPF block.
        """
        # Downsample the input and store as 'a'
        a = self.downsample(x)

        # Apply max pooling to 'a' to get 'b'
        b = self.max_pool(a)

        # Apply max pooling to 'b' to get 'c'
        c = self.max_pool(b)

        # Apply max pooling to 'c' to get 'd'
        d = self.max_pool(c)

        # Concatenate a, b, c, d
        x_cat = torch.cat([a, b, c, d], dim=1)

        # Apply the final convolution
        x = self.final_conv(x_cat)

        return x


# Copyright (c) 2023 Alessandro Mondin
# The following code is derived from the YOLOv5 implementation by Alessandro Mondin available at
# https://github.com/AlessandroMondin/YOLOV5m
class Heads(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Heads, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])  # number of anchors per scale
        self.stride = [8, 16, 32]

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6,
                                                                                                         1).T.reshape(3,
                                                                                                                      3,
                                                                                                                      2)
        self.register_buffer('anchors', anchors_)

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(5 + self.nc) * self.naxs, kernel_size=1)
            ]

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])
            bs, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(bs, self.naxs, (5 + self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

        return x

class YOLOv5m(nn.Module):
    def __init__(self, first_out, nc=10, anchors=(), ch=(), inference=False):
        super(YOLOv5m, self).__init__()
        self.inference = inference

        self.backbone = nn.ModuleList()
        self.backbone += [
            ConvBNSiLu(in_channels=3, out_channels=first_out, kernel_size=6, stride=2, padding=2),
            ConvBNSiLu(in_channels=first_out, out_channels=first_out * 2, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 2, out_channels=first_out * 2, width_multiplier=0.5, number_of_bottlenecks=2),
            ConvBNSiLu(in_channels=first_out * 2, out_channels=first_out * 4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 4, out_channels=first_out * 4, width_multiplier=0.5, number_of_bottlenecks=4),
            ConvBNSiLu(in_channels=first_out * 4, out_channels=first_out * 8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 8, out_channels=first_out * 8, width_multiplier=0.5, number_of_bottlenecks=6),
            ConvBNSiLu(in_channels=first_out * 8, out_channels=first_out * 16, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 16, out_channels=first_out * 16, width_multiplier=0.5, number_of_bottlenecks=2),
            SPPF(in_channels=first_out * 16)
        ]

        self.neck = nn.ModuleList()
        self.neck += [
            ConvBNSiLu(in_channels=first_out * 16, out_channels=first_out * 8, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out * 16, out_channels=first_out * 8, width_multiplier=0.25, number_of_bottlenecks=2, is_backbone=False),
            ConvBNSiLu(in_channels=first_out * 8, out_channels=first_out * 4, kernel_size=1, stride=1, padding=0),
            C3(in_channels=first_out * 8, out_channels=first_out * 4, width_multiplier=0.25, number_of_bottlenecks=2, is_backbone=False),
            ConvBNSiLu(in_channels=first_out * 4, out_channels=first_out * 4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 8, out_channels=first_out * 8, width_multiplier=0.5, number_of_bottlenecks=2, is_backbone=False),
            ConvBNSiLu(in_channels=first_out * 8, out_channels=first_out * 8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=first_out * 16, out_channels=first_out * 16, width_multiplier=0.5, number_of_bottlenecks=2, is_backbone=False)
        ]

        self.head = Heads(nc=nc, anchors=anchors, ch=ch)

    def forward(self, x):
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!"
        backbone_connection = []
        neck_connection = []
        outputs = []
        for idx, layer in enumerate(self.backbone):
            # takes the out of the 2nd and 3rd C3 block and stores it
            x = layer(x)
            if idx in [4, 6]:
                backbone_connection.append(x)

        for idx, layer in enumerate(self.neck):
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = Resize([x.shape[2] * 2, x.shape[3] * 2], interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif idx in [4, 6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)

            elif isinstance(layer, C3) and idx > 2:
                x = layer(x)
                outputs.append(x)

            else:
                x = layer(x)

        return self.head(outputs)
# End of code derived from the YOLOv5m implementation by Alessandro Mondin


def load_pretrained_weights(model: nn.Module) -> nn.Module:
    """
    Loads the pretrained weights into the model.

    Args:
        model (nn.Module): The model to load the weights into.
    """
    # Load the pretrained YOLOv5 model
    pretrained_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    # Counter check all layer loads correctly
    loaded_layer = 0

    # Iterate over your model's state dictionary and transfer weights based on shape
    for ((name_m, param_m), (name_p, param_p)) in zip(model.state_dict().items(), pretrained_model.state_dict().items()):
        if param_m.shape == param_p.shape:
            param_m.data.copy_(param_p.data)
            loaded_layer += 1
        else:
            print(f"Skipped parameter: {name_m} | MisMatched Shapes: {param_m.shape} vs {param_p.shape}")

    print(f'Loaded {loaded_layer} layers successfully, total {len(pretrained_model.state_dict().items())} layers in pretrained model.')
    return model