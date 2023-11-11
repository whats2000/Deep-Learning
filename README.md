# Deep-Learning

## Customized Convolution Blocks
```python
class IBConvWithSE(nn.Module):
            def __init__(self,
                         in_channels: int,  # Number of channels in the input feature map
                         out_channels: int,  # Number of channels produced by the block
                         kernel_size: int,  # Size of the convolving kernel
                         padding: int,  # Padding added to both sides of the input
                         stride: int,  # Stride of the convolution
                         num_layers: int,  # Number of depthwise layers to apply
                         expansion: int = 6,  # Expansion factor for the middle layers
                         use_se: bool = True,  # Flag to include SE block within the architecture
                         use_residual: bool = True,  # Flag to use residual connections
                         use_shortcut: bool = True,  # Flag to use shortcut connections when changing dimensions
                         activation_fn: nn.Module = nn.ReLU6(inplace=True)):  # Activation function to be used throughout the block
                """
                Initializes the IBConvWithSE block, which is an inverted bottleneck convolution block with optional squeeze-and-excitation (SE),
                residual connections, and shortcut connections.
        
                Args:
                    - in_channels (int): The number of input channels to the block.
                    - out_channels (int): The number of output channels from the block.
                    - kernel_size (int): The size of the kernel to be used in the depthwise convolution.
                    - padding (int): The amount of padding to be applied to the input of the depthwise convolution.
                    - stride (int): The stride to be used in the depthwise convolution.
                    - num_layers (int): The number of depthwise separable convolution layers to be stacked.
                    - expansion (int, optional): The multiplier for the channel expansion in the block. Default is 6.
                    - use_se (bool, optional): Whether to incorporate a squeeze-and-excitation block. Default is True.
                    - use_residual (bool, optional): Whether to include a residual connection if input and output channels are the same. Default is True.
                    - use_shortcut (bool, optional): Whether to include a shortcut connection if input and output channels differ. Default is True.
                    - activation_fn (nn.Module, optional): The activation function to use after each convolution. Default is ReLU6 with in-place operation for efficiency.
                """
                super(IBConvWithSE, self).__init__()

                class Swish(nn.Module):
                    def forward(self, x):
                        return x * torch.sigmoid(x)

                self.use_se = use_se
                self.use_residual = use_residual and in_channels == out_channels
                self.use_shortcut = use_shortcut and in_channels != out_channels
                self.num_layers = num_layers
                self.expanded_channels = in_channels * expansion

                # Initial Convolution
                self.initial_conv = nn.Sequential(
                    nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.expanded_channels),
                    activation_fn
                )

                # Short-cut point-wise convolution
                if self.use_shortcut:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )

                # Depth-wise separable convolutions with multiple layers
                self.depthwise_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.expanded_channels, self.expanded_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.expanded_channels, bias=False),
                        nn.BatchNorm2d(self.expanded_channels),
                        activation_fn
                    ) for _ in range(num_layers)
                ])

                # SE block
                if self.use_se:
                    self.se_block = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(self.expanded_channels, self.expanded_channels // 16, kernel_size=1),
                        Swish(),
                        nn.Conv2d(self.expanded_channels // 16, self.expanded_channels, kernel_size=1),
                        nn.Sigmoid()
                    )

                # Point-wise convolution
                self.pointwise_conv = nn.Sequential(
                    nn.Conv2d(self.expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    activation_fn
                )

                # Dropout layer
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                identity = x

                x = self.initial_conv(x)
                for depthwise_layer in self.depthwise_layers:
                    x = depthwise_layer(x)

                if self.use_se:
                    se = self.se_block(x)
                    x = x * se

                x = self.pointwise_conv(x)
                x = self.dropout(x)

                if self.use_residual:
                    x += identity

                if self.use_shortcut:
                    x += self.shortcut(identity)

                return x
```
