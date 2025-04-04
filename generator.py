import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, dim_args={}):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        nz = dim_args.get("random_dim", 100)     # Dimension of latent vector z
        ngf = dim_args.get("gen_dim", 64)        # Size of feature maps in generator
        nc = dim_args.get("num_channels", 3)     # Number of channels in the output image
        
        # Apply spectral normalization option
        use_spectral_norm = dim_args.get("use_spectral_norm", True)  # Whether to use spectral normalization
        norm_layer = nn.BatchNorm2d  # Default to use BatchNorm
        
        # Add helper function to select whether to use spectral normalization
        def get_conv(in_channels, out_channels, kernel_size, stride, padding, bias=False):
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            if use_spectral_norm:
                return nn.utils.spectral_norm(conv)
            return conv
        
        # Main generator network
        self.main = nn.Sequential(
            # Input is Z, enters the convolution
            get_conv(nz, ngf * 8, 4, 1, 0, bias=False),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # Changed to LeakyReLU to enhance stability
            # State size: (ngf*8) x 4 x 4
            get_conv(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            norm_layer(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ngf*4) x 8 x 8
            get_conv(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ngf*2) x 16 x 16
            get_conv(ngf * 2, ngf, 4, 2, 1, bias=False),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ngf) x 32 x 32
            get_conv(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: (nc) x 64 x 64
        )
        
        # Initialize weights using Kaiming initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # Ensure input has the correct shape
        if input.dim() == 2:
            input = input.unsqueeze(-1).unsqueeze(-1)
        return self.main(input)