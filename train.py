"""
train.py
-------
PSSR PIPELINE - STEP 3:
Train your data -  Train a PSSR model using a dataset generated in step 2.

Parameters:
-------
- gpu: str, GPU to run on", all GPUs will be used if not flagged. (optional)
- arch: str, encode architecture, 'wnresnet34' by default. (optional)
- bs: int, batch size per gpu, 8 by default. (optional)
- lr: float, learning rate, 1e-4 by default. (optional)
- lr_start: float, learning rate starts at, None by default. (optional)
- noise: boolean, add dynamic crappifier, False by default. (optional)
- freeze: boolean, freeze up to last layer group, False by default. (optional)
- pretrain: boolean, if to use a pre-trained arch, False by default. (optional)
- size: int, img size, 256 by default. (optional)
- cycles: int, num of cycles, 5 by default. (optional)
- load_name: str, load model name, None by default. (optional)
- save_name: str, model save name, 'combo' by default. (optional)
- datasetname: str, dataset name, 'tiles_002' by default. (optional)
- tile_sz: int, tile size, 256 by default. (optional)
- attn: boolean, self attention, True by default. (optional)
- blur: boolean, upsample blur, True by default. (optional)
- final_blur: boolean, final upsample blur, True by default. (optional)
- last_cross: boolean, last_cross, True by default. (optional)
- bottle: boolean, bottleneck, True by default. (optional)
- cutout: boolean, cutout, False by default. (optional)
- rrdb: boolean, use RRDB_Net, False by default. (optional)
- nf: int, rrdb nf, 32 by default. (optional)
- nb: int, rrdb nb, 32 by default. (optional)
- gcval: int, rrdb gc, 32 by default. (optional)
- clip_grad: float, gradient clipping, None by default. (optional)
- loss_scale: float, loss scale, None by default. (optional)
- feat_loss: boolean, feat_loss, False by default. (optional)
- n_frames: int, number of frames
- lr_type: str, training input, (s)ingle, (t) multi or (z) multi, 's', 't', or 'z'
- plateau: boolean, cut LR on plateaus, False by default. (optional)
- old_unet: boolean, use old unet_learner, False by default. (optional)
- skip_train: boolean, skip training, e.g. to adjust size, False by default. (optional)
- mode: str, image mode like L or RGB, 'L' by default. (optional)
- norm: boolean, normalize data, True by default. (optional)
- l1_loss: boolean, use L1 loss, False by default. (optional)
- debug: boolean, debug mode, False by default. (optional)

Returns:
-------
- trained PSSR model:
    - .pth models
    - .pkl models.

Examples:
-------
Example 1: Train a singleframe model
Train 50 epochs first, and continue training with another 50 epochs.
python -m fastai.launch train.py --bs 8 --lr 4e-4 --size 512 --tile_sz 512 --datasetname single_mito_AG_SP
          --cycles 50 --save_name single_mito_AG_SP_e50 --lr_type s --n_frames 1
python -m fastai.launch train.py --bs 8 --lr 4e-4 --size 512 --tile_sz 512 --datasetname single_mito_AG_SP
          --cycles 50 --save_name single_mito_AG_SP_e100 --load_name single_mito_AG_SP_e50_best_512 --lr_type s --n_frames 1

Example 2: Train a multiframe model
Train 50 epochs first, and continue training with another 50 epochs.
python -m fastai.launch  train.py --bs 8 --lr 4e-4 --size 512 --tile_sz 512 --datasetname multi_mito_AG_SP
          --cycles 50 --save_name multit_5_mito_AG_SP_e50 --lr_type t --n_frames 5
python -m fastai.launch  train.py --bs 8 --lr 4e-4 --size 512 --tile_sz 512 --datasetname multi_mito_AG_SP
          --cycles 50 --save_name multit_5_mito_AG_SP_e100 --load_name multit_5_mito_AG_SP_e50_best_512 --lr_type t --n_frames 5
"""

import yaml
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.models import resnet18, resnet34, resnet50
from skimage.util import random_noise
from skimage import filters
from utils import *
from utils.resnet import *
from pdb import set_trace
torch.backends.cudnn.benchmark = True

def get_src(x_data, y_data, n_frames=1, mode='L'):
    def map_to_hr(x):
        return y_data/x.relative_to(x_data).with_suffix('.tif')

    if n_frames == 1:
        src = (ImageImageList
                .from_folder(x_data, convert_mode=mode)
                .split_by_folder()
                .label_from_func(map_to_hr, convert_mode=mode))
    else:
        src = (MultiImageImageList
                .from_folder(x_data, extensions=['.npy'])
                .split_by_folder()
                .label_from_func(map_to_hr, convert_mode=mode))
    return src

def get_data(bs, size, x_data, y_data,
             n_frames=1,
             max_rotate=10.,
             min_zoom=1., max_zoom=1.1,
             use_cutout=False,
             use_noise=True,
             scale=4,
             xtra_tfms=None,
             gauss_sigma=(0.4,0.7),
             pscale=(5,30),
             mode='L',
             norm=False,
             **kwargs):
    src = get_src(x_data, y_data, n_frames=n_frames, mode=mode)

    x_tfms, y_tfms = get_xy_transforms(
                          max_rotate=max_rotate,
                          min_zoom=min_zoom, max_zoom=max_zoom,
                          use_cutout=use_cutout,
                          use_noise=use_noise,
                          gauss_sigma=gauss_sigma,
                          pscale=pscale,
                          xtra_tfms = xtra_tfms)
    x_size = size // scale
    data = (src
            .transform(x_tfms, size=x_size)
            .transform_y(y_tfms, size=size)
            .databunch(bs=bs, **kwargs))
    if norm:
        print('normalizing x and y data')
        data = data.normalize(do_y=True)
    data.c = 3
    return data

@call_parse
def main(
        gpu: Param("GPU to run on", str)=None,
        arch: Param("encode architecture", str) = 'wnresnet34',
        bs: Param("batch size per gpu", int) = 8,
        lr: Param("learning rate", float) = 1e-4,
        lr_start: Param("learning rate start", float) = None,
        noise: Param("add dynamic crappifier", action='store_true') = False,
        freeze: Param("learning rate", action='store_true') = False,
        pretrain: Param("arch pre-trained", action='store_true') = False,
        size: Param("img size", int) = 256,
        cycles: Param("num cyles", int) = 5,
        load_name: Param("load model name", str) = None,
        save_name: Param("model save name", str) = 'saved_model',
        datasetname: Param('dataset name', str) = 'tiles_002',
        tile_sz: Param('tile_sz', int) = 256,
        attn: Param('self attention', bool)=True,
        blur: Param('upsample blur', bool)=True,
        final_blur: Param('final upsample blur', bool)=True,
        last_cross: Param('last_cross', bool)=True,
        bottle: Param('bottleneck', action='store_true')=True,
        cutout: Param('cutout', action='store_true')=False,
        rrdb: Param('use RRDB_Net', action='store_true')=False,
        nf: Param('rrdb nf', int) = 32,
        nb: Param('rrdb nb', int) = 32,
        gcval: Param('rrdb gc', int) = 32,
        clip_grad: Param('gradient clipping', float) = None,
        loss_scale: Param('loss scale', float) = None,
        feat_loss: Param('feat_loss', action='store_true')=False,
        n_frames: Param('number of frames', int) = None,
        lr_type: Param('training input, (s)ingle, (t) multi or (z) multi', str)=None,
        plateau: Param('cut LR on plateaus', action='store_true')=False,
        old_unet: Param('use old unet_learner', action='store_true')=False,
        skip_train: Param('skip training, e.g. to adjust size', action='store_true') = False,
        mode: Param('image mode like L or RGB', str)='L',
        norm: Param('normalize data', bool)=True,
        l1_loss: Param('use L1 loss', action='store_true')=False,
        debug: Param('debug mode', action='store_true')=False
):
    if lr_type == 's':
        z_frames, t_frames = 1, 1
        n_frames = 1
    elif lr_type == 't':
        z_frames, t_frames = 1, n_frames
    elif lr_type == 'z':
        z_frames, t_frames = n_frames, 1
    multi_str = f'_{lr_type}_{n_frames}' if lr_type != 's' else ''

    data_path = Path('.')
    datasets = data_path/'datasets'
    dataset = datasets/datasetname
    pickle_models = ensure_folder(data_path/'models/pkl_files')
    model_dir = ensure_folder(data_path/'models/pth_files')

    if tile_sz is None:
        hr_tifs = dataset/f'hr'
        lr_tifs = dataset/f'lr'
    else:
        hr_tifs = dataset/f'hr_{lr_type}_{tile_sz:d}{multi_str}'
        lr_tifs = dataset/f'lr_{lr_type}_{tile_sz:d}{multi_str}'

    if not debug:
        gpu = setup_distrib(gpu)
        print('on gpu: ', gpu)
        n_gpus = num_distrib()
    else:
        print('debug mode')
        gpu = 0
        n_gpus = 0

    if feat_loss: loss = get_feat_loss()
    elif l1_loss: loss = F.l1_loss
    else: loss = F.mse_loss
    print('loss: ', loss)
    metrics = sr_metrics

    bs = max(bs, bs * n_gpus)
    size = size
    arch = eval(arch)

    save_name = f'{lr_type}_{n_frames}_{save_name}_e{cycles}'
    
    print(f'hr dataset path: {hr_tifs}')
    print(f'lr dataset path: {lr_tifs}')
    print('bs:', bs, 'size: ', size, 'ngpu:', n_gpus)
    data = get_data(bs, size, lr_tifs, hr_tifs, n_frames=n_frames,  max_zoom=4.,
                    use_cutout=cutout, use_noise=noise, mode=mode, norm=norm)
    callback_fns = []

    #if gradient_clip:
        #callback_fns.append(partial(GradientClipping, clip=0.1))
    if plateau:
        callback_fns.append(partial(ReduceLROnPlateauCallback, patience=1))
    if gpu == 0 or gpu is None:
        if feat_loss:
            callback_fns = [LossMetrics]
        callback_fns.append(partial(SaveModelCallback, name=f'{save_name}_best_{size}'))

    if rrdb:
        rrdb_args = {
            'nf': nf,
            'nb': nb,
            'gcval': gcval,
            'upscale': 4
        }
        learn = rrdb_learner(data, in_c=n_frames, rrdb_args=rrdb_args, path=Path('.'),
                             loss_func=loss, metrics=metrics, model_dir=model_dir, callback_fns=callback_fns)
    else:
        wnres_args = {
            'blur': blur,
            'blur_final': final_blur,
            'bottle': bottle,
            'self_attention': attn,
            'last_cross': True
        }
        wd = 1e-3
        if old_unet:
            learn = unet_learner(data, arch, wd=wd, loss_func=loss,
                                 metrics=metrics, callback_fns=callback_fns,
                                 norm_type=NormType.Weight,
                                 model_dir=model_dir, path=Path('.'),
                                 **wnres_args)
            learn.model = BilinearWrapper(learn.model)
        else:
            learn = wnres_unet_learner(data, arch, in_c=n_frames, wnres_args=wnres_args,
                                       path=Path('.'), loss_func=loss, metrics=metrics,
                                       model_dir=model_dir, callback_fns=callback_fns, wd=wd)
    gc.collect()

    if load_name:
        learn = learn.load(f'{load_name}')
        print(f'loaded {load_name}')

    if freeze:
        learn.freeze()

    if not debug:
        if gpu is None: learn.model = nn.DataParallel(learn.model)
        else: learn.to_distributed(gpu)

    if not clip_grad is None:
        learn = learn.clip_grad(clip_grad)
    if not loss_scale is None:
        learn = learn.to_fp16(loss_scale=loss_scale)
    else:
        learn = learn.to_fp16()

    if not lr_start is None: lr = slice(lr_start, lr)
    else: lr = slice(None, lr, None)
    if not skip_train:
        learn.fit_one_cycle(cycles, lr)

    if gpu == 0 or gpu is None:
        learn.save(save_name)
        print(f'saved: {save_name}')
        learn.export(pickle_models/f'{save_name}_{size}.pkl')
        print('exported')
