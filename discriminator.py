import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, dim_args={}):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nc = dim_args.get("num_channels", 3)    # Input image channels
        ndf = dim_args.get("disc_dim", 64)      # Size of feature maps in discriminator
        
        # Add options
        use_spectral_norm = dim_args.get("use_spectral_norm", True)  # Whether to use spectral normalization
        use_dropout = dim_args.get("use_dropout", True)  # Whether to use Dropout
        
        # Auxiliary function to choose whether to use spectral normalization
        def get_conv(in_channels, out_channels, kernel_size, stride, padding, bias=False):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            if use_spectral_norm:
                return nn.utils.spectral_norm(conv)
            return conv
        
        layers = []
        
        # Input is (nc) x 64 x 64
        # First layer has no batchnorm
        layers.append(get_conv(nc, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))  # Add dropout
        
        # State size: (ndf) x 32 x 32
        layers.append(get_conv(ndf, ndf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        
        # State size: (ndf*2) x 16 x 16
        layers.append(get_conv(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        
        # State size: (ndf*4) x 8 x 8
        layers.append(get_conv(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        
        # State size: (ndf*8) x 4 x 4
        layers.append(get_conv(ndf * 8, 1, 4, 1, 0, bias=False))
        
        # Use a smooth activation function instead of a hard Sigmoid
        # This helps prevent the discriminator from being too confident
        layers.append(nn.Sigmoid())
        
        self.main = nn.Sequential(*layers)
        
        # Use improved weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)