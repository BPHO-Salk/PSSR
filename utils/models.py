
import numpy as np
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from scipy.ndimage.interpolation import zoom as npzoom
from .utils import unet_image_from_tiles_blend

__all__ = ['get_named_processor', 'add_model_processor']

def bilinear(img, **kwargs):
    pred_img = npzoom(img, 4, order=1)
    return pred_img / pred_img.max()

def bicubic(img, **kwargs):
    pred_img=npzoom(img, 4, order=3)
    return pred_img

def original(img, **kwargs):
    pred_img = npzoom(img, 4, order=0)
    return pred_img / pred_img.max()

processors = {
    'original': (original, 1),
    'bilinear': (bilinear, 1),
    'bicubic':  (bicubic, 1)
}

def num_channels(learn):
    ps = [p for p in learn.model.parameters()]
    if len(ps[1].shape) == 1:
        return ps[0].shape[1]
    else:
        return ps[1].shape[1]

def build_processor(name, model_dir, use_tiles):
    learn = load_learner(model_dir, f'{name}.pkl').to_fp16()
    tile_sz = int(name.split('_')[-1])
    def learn_processor(img, img_info=None, mode='L'):
        if len(img.shape) == 2:
            img = img[None]
            if mode == 'RGB': img = img.repeat(3, axis=0)
        pred_img = unet_image_from_tiles_blend(learn, img, use_tiles, tile_sz=tile_sz, img_info=img_info)
        return pred_img

    return learn_processor, num_channels(learn)


def get_named_processor(name, model_dir, use_tiles):
    if not name in processors:
        proc, num_chan = build_processor(name, model_dir, use_tiles)
        processors[name] = proc, num_chan
    proc, num_chan = processors.get(name, None)
    return proc, num_chan

def make_learner(model_name, model_dir, path):
    print('make_learner here')
    return None

def add_model_processor(model_name, model_dir, path='.'):
    if model_name not in processors:
        def learner_proc(lrn, img):
            return lrn.predict(img)
        learner = make_learner(model_name, model_dir, path)
        processors[model_name] = partial(learner_proc, learner)
