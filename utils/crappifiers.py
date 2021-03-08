import numpy as np
from skimage import filters
from skimage.util import random_noise, img_as_ubyte, img_as_float
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.transform import rescale
import PIL

def no_crap(img, scale=4, upsample=False):
    from skimage.transform import rescale
    x = np.array(img)
    multichannel = len(x.shape) > 2
    x = rescale(x, scale=1/scale, order=1, multichannel=multichannel)
    x *= np.iinfo(np.uint8).max
    return PIL.Image.fromarray(x.astype(np.uint8))

def fluo_G_D(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)

    x = np.array(x)
    mu, sigma = 0, 5
    noise = np.random.normal(mu, sigma*0.05, x.shape)
    x = np.clip(x + noise, 0, 1)
    x_down = npzoom(x, 1/scale, order=1)
    #x_up = npzoom(x_down, scale, order=1)
    return PIL.Image.fromarray(x_down.astype(np.uint8))

def fluo_AG_D(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)

    lvar = filters.gaussian(xn, sigma=5) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    x_down = npzoom(x, 1/scale, order=1)
    #x_up = npzoom(x_down, scale, order=1)
    return PIL.Image.fromarray(x_down.astype(np.uint8))

def fluo_downsampleonly(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)
    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    x_down = npzoom(x, 1/scale, order=1)
    #x_up = npzoom(x_down, scale, order=1)
    return PIL.Image.fromarray(x_down.astype(np.uint8))

def fluo_SP_D(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)
    xn = random_noise(xn, mode='salt', amount=0.005)
    xn = random_noise(xn, mode='pepper', amount=0.005)
    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    x_down = npzoom(x, 1/scale, order=1)
    #x_up = npzoom(x_down, scale, order=1)
    return PIL.Image.fromarray(x_down.astype(np.uint8))

def fluo_SP_AG_D_sameas_preprint(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)
    xn = random_noise(xn, mode='salt', amount=0.005)
    xn = random_noise(xn, mode='pepper', amount=0.005)
    lvar = filters.gaussian(xn, sigma=5) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    x_down = npzoom(x, 1/scale, order=1)
    return PIL.Image.fromarray(x_down.astype(np.uint8))

def fluo_SP_AG_D_sameas_preprint_rescale(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)
    xn = random_noise(xn, mode='salt', amount=0.005)
    xn = random_noise(xn, mode='pepper', amount=0.005)
    lvar = filters.gaussian(xn, sigma=5) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    multichannel = len(x.shape) > 2
    x_down = rescale(x, scale=1/scale, order=1, multichannel=multichannel)
    return PIL.Image.fromarray(x_down.astype(np.uint8))

def em_AG_D_sameas_preprint(x, scale=4, upsample=False):
    lvar = filters.gaussian(x, sigma=3)
    x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def em_downsampleonly(x, scale=4, upsample=False):
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def em_G_D_001(x, scale=4, upsample=False):
    noise = np.random.normal(0, 3, x.shape)
    x = x + noise
    x = x - x.min()
    x = x/x.max()
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def em_G_D_002(x, scale=4, upsample=False):
    x = img_as_float(x)
    mu, sigma = 0, 3
    noise = np.random.normal(mu, sigma*0.05, x.shape)
    x = np.clip(x + noise, 0, 1)
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def em_P_D_001(x, scale=4, upsample=False):
    x = random_noise(x, mode='poisson', seed=1)
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def new_crap_AG_SP(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)

    lvar = filters.gaussian(xn, sigma=5) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)

    xn = random_noise(xn, mode='salt', amount=0.005)
    xn = random_noise(xn, mode='pepper', amount=0.005)

    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    multichannel = len(x.shape) > 2

    xn = rescale(xn, scale=1/scale, order=1, multichannel=multichannel)
    return PIL.Image.fromarray(xn.astype(np.uint8))

def new_crap(x, scale=4, upsample=False):
    xn = np.array(x)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)

    xn = random_noise(xn, mode='salt', amount=0.005)
    xn = random_noise(xn, mode='pepper', amount=0.005)
    lvar = filters.gaussian(xn, sigma=5) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    multichannel = len(x.shape) > 2
    x = rescale(x, scale=1/scale, order=1, multichannel=multichannel)
    return PIL.Image.fromarray(x.astype(np.uint8))

###not sure about this one
def em_AG_P_D_001(x, scale=4, upsample=False):
    poisson_noisemap = np.random.poisson(x, size=None)
    set_trace()
    lvar = filters.gaussian(x, sigma=3)
    x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
    x = x + poisson_noisemap
    #x = x - x.min()
    #x = x/x.max()
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up
