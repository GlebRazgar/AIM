import torch
import math
from torch import nn
import gdn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_channels=64, num_filters=512, **kwargs):
        super(SparseAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=1, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, 2*num_filters, 5, stride=1, padding=2)
        self.gdn3 = gdn.GDN(2*num_filters)
        '''self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)'''

        self.deconv1 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=1, padding=2, output_padding=0)
        '''self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)'''


    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        '''x = self.conv4(x)'''
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        '''x = self.igdn3(x)
        x = self.deconv4(x)'''

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat

class SparseAutoencoder2(nn.Module):
    def __init__(self, input_channels=64, num_filters=512, **kwargs):
        super(SparseAutoencoder2, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        #self.gdn3 = gdn.GDN(num_filters)
        '''self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)'''

        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=2, padding=2, output_padding=1)
        '''self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)'''


    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        #x = self.gdn3(x)
        '''x = self.conv4(x)'''
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        '''x = self.igdn3(x)
        x = self.deconv4(x)'''

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat

class SparseAutoencoder3(nn.Module):
    def __init__(self, input_channels=32, num_filters=512, **kwargs):
        super(SparseAutoencoder3, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, 2*num_filters, 5, stride=1, padding=2)
        self.gdn2 = gdn.GDN(2*num_filters)
        self.conv3 = nn.Conv2d(2*num_filters, 2*num_filters, 5, stride=1, padding=2)
        #self.gdn3 = gdn.GDN(num_filters)
        '''self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)'''

        self.deconv1 = nn.ConvTranspose2d(2*num_filters, 2*num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn1 = gdn.GDN(2*num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=2, padding=2, output_padding=1)
        '''self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)'''

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        #x = self.gdn3(x)
        '''x = self.conv4(x)'''
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        '''x = self.igdn3(x)
        x = self.deconv4(x)'''

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat

class SparseAutoencoder4(nn.Module):
    def __init__(self, input_channels=64, num_filters=512, **kwargs):
        super(SparseAutoencoder4, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=1, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, 2*num_filters, 5, stride=1, padding=2)
        self.gdn2 = gdn.GDN(2*num_filters)
        self.conv3 = nn.Conv2d(2*num_filters, 2*num_filters, 5, stride=1, padding=2)
        #self.gdn3 = gdn.GDN(num_filters)
        '''self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)'''

        self.deconv1 = nn.ConvTranspose2d(2*num_filters, 2*num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn1 = gdn.GDN(2*num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=1, padding=2, output_padding=0)
        '''self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)'''


    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        #x = self.gdn3(x)
        '''x = self.conv4(x)'''
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        '''x = self.igdn3(x)
        x = self.deconv4(x)'''

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat

class SparseAutoencoder5(nn.Module):
    def __init__(self, input_channels=64, num_filters=512, **kwargs):
        super(SparseAutoencoder5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=1, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, 2*num_filters, 5, stride=1, padding=2)
        self.gdn2 = gdn.GDN(2*num_filters)
        self.conv3 = nn.Conv2d(2*num_filters, 2*num_filters, 5, stride=2, padding=2)
        #self.gdn3 = gdn.GDN(num_filters)
        '''self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)'''

        self.deconv1 = nn.ConvTranspose2d(2*num_filters, 2*num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(2*num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=1, padding=2, output_padding=0)
        '''self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)'''


    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        #x = self.gdn3(x)
        '''x = self.conv4(x)'''
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        '''x = self.igdn3(x)
        x = self.deconv4(x)'''

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat


class SparseAutoencoder7(nn.Module):
    def __init__(self, input_channels=64, num_filters=512, **kwargs):
        super(SparseAutoencoder7, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=1, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, 2 * num_filters, 5, stride=1, padding=2)
        self.gdn2 = gdn.GDN(2 * num_filters)
        self.conv3 = nn.Conv2d(2 * num_filters, 2 * num_filters, 5, stride=2, padding=2)
        # self.gdn3 = gdn.GDN(num_filters)
        '''self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)'''

        self.deconv1 = nn.ConvTranspose2d(2 * num_filters, 2 * num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn1 = gdn.GDN(2 * num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=2, padding=2, output_padding=1)
        '''self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)'''

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        # x = self.gdn3(x)
        '''x = self.conv4(x)'''
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        '''x = self.igdn3(x)
        x = self.deconv4(x)'''

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat


class SparseAutoencoder8(nn.Module):
    def __init__(self, input_channels=64, num_filters=512, **kwargs):
        super(SparseAutoencoder8, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=1, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, 2 * num_filters, 5, stride=1, padding=2)
        self.gdn2 = gdn.GDN(2 * num_filters)
        self.conv3 = nn.Conv2d(2 * num_filters, 2 * num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(2 * num_filters)
        self.conv4 = nn.Conv2d(2 * num_filters, 4 * num_filters, 5, stride=2, padding=2)

        self.deconv1 = nn.ConvTranspose2d(4 * num_filters, 2 * num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn1 = gdn.GDN(2 * num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2 * num_filters, 2 * num_filters, 5, stride=1, padding=2, output_padding=0)
        self.igdn2 = gdn.GDN(2 * num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=2, padding=2, output_padding=1)

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        x = self.igdn3(x)
        x = self.deconv4(x)

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat


class SparseAutoencoder9(nn.Module):
    def __init__(self, input_channels=32, num_filters=512, **kwargs):
        super(SparseAutoencoder9, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, 2 * num_filters, 5, stride=2, padding=2)

        self.deconv1 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, input_channels, 5, stride=2, padding=2, output_padding=1)

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)

        return x

    def decode(self, x):
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)

        return x

    def forward(self, x):
        z = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        xhat = self.decode(z)  # p(w|y), i.e. the "common variable"

        return z, xhat