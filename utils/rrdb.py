import torch.nn as nn
import torch,math,sys
import torch.nn.functional as F
from fastai import *
from fastai.vision import *
from fastai.layers import Lambda, PixelShuffle_ICNR, conv_layer, NormType

__all__ = ['RRDB_Net', 'rrdb_learner']

class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc, gc=32):
        super().__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_layer(nc, gc, norm_type=NormType.Weight, leaky=0.02)
        self.conv2 = conv_layer(nc+gc, gc, norm_type=NormType.Weight, leaky=0.02)
        self.conv3 = conv_layer(nc+2*gc, gc, norm_type=NormType.Weight, leaky=0.02)
        self.conv4 = conv_layer(nc+3*gc, gc, norm_type=NormType.Weight, leaky=0.02)
        # turn off activation?
        self.conv5 = conv_layer(nc+4*gc, nc, norm_type=NormType.Weight, leaky=0.02, use_activ=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

class RRDB(nn.Module):
    def __init__(self, nc, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, gc)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x

class RRDB_Net(nn.Module):
    def __init__(self, in_nc, out_nc, nf=32, nb=8, gcval=32, upscale=4):
        super(RRDB_Net, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        fea_conv = conv_layer(in_nc, nf, norm_type=NormType.Weight, use_activ=False)
        rb_blocks = [RRDB(nf, gc=gcval) for _ in range(nb)]
        LR_conv = conv_layer(nf, nf, leaky=0.2)

        if upscale == 3:
            upsampler = PixelShuffle_ICNR(nf, blur=True, leaky=0.02, scale=3)
        else:
            upsampler = [PixelShuffle_ICNR(nf, blur=True, leaky=0.02) for _ in range(n_upscale)]

        HR_conv0 = conv_layer(nf, nf, leaky=0.02, norm_type=NormType.Weight)
        HR_conv1 = conv_layer(nf, out_nc, leaky=0.02, norm_type=NormType.Weight, use_activ=False)

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1
        )

    def forward(self, x):
        x = self.model(x)
        return x

def rrdb_learner(data, in_c=1, out_c=1, rrdb_args=None, **kwargs):
    if rrdb_args is None: rrdb_args = {}

    model = RRDB_Net(in_nc=in_c, out_nc=out_c, **rrdb_args)
    learn = Learner(data, model, **kwargs)
    return learn
