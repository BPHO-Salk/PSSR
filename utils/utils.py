"utility methods for generating movies from learners"
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import shutil
from skimage.filters import gaussian
from skimage.io import imsave
import PIL
import imageio
from scipy.ndimage.interpolation import zoom as npzoom
from .czi import get_czi_shape_info, build_index, is_movie
import czifile
import PIL
import numpy as np
from fastprogress import progress_bar
from pathlib import Path
import torch
import math
from .multi import MultiImage
from time import sleep
import shutil
from skimage.util import random_noise
from skimage import filters
from torchvision.models import vgg16_bn

__all__ = ['generate_movies', 'generate_tifs', 'ensure_folder', 'subfolders',
           'build_tile_info', 'generate_tiles', 'unet_image_from_tiles_blend',
           'get_xy_transforms', 'get_feat_loss', 'unet_image_from_tiles_partialsave',
           'draw_random_tile', 'img_to_float', 'img_to_uint8']


def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts, base_loss=F.l1_loss):
        super().__init__()
        self.base_loss = base_loss
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        feat_input = input.repeat(1,3,1,1)
        feat_target = target.repeat(1,3,1,1)
        base_loss = self.base_loss
        out_feat = self.make_features(feat_target, clone=True)
        in_feat = self.make_features(feat_input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    def __del__(self): self.hooks.remove()

def get_feat_loss():
    vgg_m = vgg16_bn(True).features.cuda().eval()
    requires_grad(vgg_m, False)
    blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
    feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
    return feat_loss

def _down_up(x, scale=4, upsample=False, mode='bilinear'):
    set_trace()
    x = F.interpolate(x[None], scale_factor=1/scale)[0]
    if upsample:
        x = F.interpolate(x[None], scale_factor=scale, mode=mode)[0]
    print('du shpe:', x.shape)
    return x
down_up = TfmPixel(_down_up)

def _my_noise_old(x, gauss_sigma:uniform=0.01, pscale:uniform=10):
    #print('noise')
    #set_trace()
    xn = x.numpy()
    xorig_max = xn.max()
    xn = np.random.poisson(xn*pscale)/pscale
    xn += np.random.normal(0, gauss_sigma*xn.std(), size=x.shape)
    xn = np.maximum(0,xn)
    new_max = xn.max()
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    x = x.new(xn)
    return x


def _my_noise(x, gauss_sigma:uniform=0.01, pscale:uniform=10):
    xn = x.numpy()
    xorig_max = xn.max()

    xn = random_noise(xn, mode='salt', amount=0.005)
    xn = random_noise(xn, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    xn = random_noise(xn, mode='localvar', local_vars=lvar*0.5)
    #xn = np.random.poisson(xn*pscale)/pscale
    #xn += np.random.normal(0, gauss_sigma*xn.std(), size=x.shape)
    x = x.new(xn)
    new_max = xn.max()
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    return x


my_noise = TfmPixel(_my_noise)

def get_xy_transforms(max_rotate=10., min_zoom=1., max_zoom=2., use_cutout=False, use_noise=False, xtra_tfms = None,
                      gauss_sigma=(0.01,0.05), pscale=(5,30)):
    base_tfms = [[
            rand_crop(),
            dihedral_affine(),
            rotate(degrees=(-max_rotate,max_rotate)),
            rand_zoom(min_zoom, max_zoom)
        ],
        [crop_pad()]]

    y_tfms = [[tfm for tfm in base_tfms[0]], [tfm for tfm in base_tfms[1]]]
    x_tfms = [[tfm for tfm in base_tfms[0]], [tfm for tfm in base_tfms[1]]]
    if use_cutout: x_tfms[0].append(cutout(n_holes=(5,10)))
    if use_noise:
        x_tfms[0].append(my_noise(gauss_sigma=gauss_sigma, pscale=pscale))
        #x_tfms[1].append(my_noise(gauss_sigma=(0.01,0.05),pscale=(5,30)))

    if xtra_tfms:
        for tfm in xtra_tfms:
            x_tfms[0].append(tfm)

    return x_tfms, y_tfms



def make_mask(shape, overlap, top=True, left=True, right=True, bottom=True):
    mask = np.full(shape, 1.)
    if overlap > 0:
        h,w = shape
        for i in range(min(shape[0], shape[0])):
            for j in range(shape[1]):
                if top: mask[i,j] = min((i+1)/overlap, mask[i,j])
                if bottom: mask[h-i-1,j] = min((i+1)/overlap, mask[h-i-1,j])
                if left: mask[i,j] = min((j+1)/overlap, mask[i,j])
                if right: mask[i,w-j-1] = min((j+1)/overlap, mask[i,w-j-1])
    return mask.astype(np.uint8)

def unet_image_from_tiles_partialsave(learn, in_img, tile_sz=(256, 256), scale=(4, 4), overlap_pct=(0.50, 0.50), img_info=None):
    """
    This function run inference on a trained model and removes tiling artifacts.  

    Input:
    - learn: learner
    - in_img: input image (2d/3d), floating array
    - tile_sz: XY dimension of the small tile that will be fed into GPU [p q] 
    - scale: upsampling scale
    - overlap_pct: overlap percent while cropping the tiles in xy dimension [alpha beta],
                   floating tuple, ranging from 0 to 1
    - img_info: mi, ma, max

    Output:
    - predicted image (2d), ranging from 0 to 1

    """    
    n_frames = in_img.shape[0]

    if img_info:
        mi, ma, imax = [img_info[fld] for fld in ['mi','ma','img_max']]
        in_img = ((in_img - mi) / (ma - mi + 1e-20)).clip(0.,1.)
    else:
        mi, ma = 0., 1.
    in_img  = np.stack([npzoom(in_img[i], scale, order=1) for i in range(n_frames)])

    Y, X = in_img.shape[1:3]
    p, q = tile_sz[0:2]
    alpha, beta = overlap_pct[0:2]
    print('Y,X=',Y,X)
    assembled = np.zeros((X,Y))

    # X = p + (m - 1) * (1 - alpha) * p + epsilonX
    numX, epsX = divmod(X-p, p-int(p*alpha)) if X-p > 0 else (0, X)
    numY, epsY = divmod(Y-q, q-int(q*beta)) if Y-q > 0 else (0, Y)
    numX = int(numX)+1
    numY = int(numY)+1
    
    for i in range(numX+1):
        for j in range(numY+1):
            crop_x_start = int(i*(1-alpha)*p)
            crop_x_end = min(crop_x_start+p, X)
            crop_y_start = int(j*(1-beta)*q)
            crop_y_end = min(crop_y_start+q, Y)

            src_tile = in_img[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

            in_tile = torch.zeros((p, q, n_frames))
            in_x_size = crop_x_end - crop_x_start
            in_y_size = crop_y_end - crop_y_start
            if (in_y_size, in_x_size) != src_tile.shape[1:3]: set_trace()
            in_tile[0:in_y_size, 0:in_x_size, :] = tensor(src_tile).permute(1,2,0)

            if n_frames > 1:
                img_in = MultiImage([Image(in_tile[:,:,i][None]) for i in range(n_frames)])
            else:
                img_in = Image(in_tile[:,:,0][None])
            y, pred, raw_pred = learn.predict(img_in)

            out_tile = pred.numpy()[0]

            tileROI_x_start = int(0.5*int(alpha*p)) if crop_x_start != 0 else 0
            tileROI_x_end = int(p-0.5*int(alpha*p)) if crop_x_end != X else int(alpha*p+epsX)
            tileROI_y_start = int(0.5*int(beta*q)) if crop_y_start != 0 else 0
            tileROI_y_end = int(q-0.5*int(beta*q)) if crop_y_end != Y else int(beta*q+epsY)

            tileROI_x_end = X if X-q < 0 else tileROI_x_end
            tileROI_y_end = Y if Y-p < 0 else tileROI_y_end

            out_x_start = int(p-0.5*int(alpha*p)+(i-1)*(p-int(alpha*p))) if crop_x_start != 0 else 0
            out_x_end = int(p-0.5*int(alpha*p)+i*(p-int(alpha*p))) if crop_x_end != X else X
            out_y_start = int(q-0.5*int(beta*q)+(j-1)*(q-int(beta*q))) if crop_y_start != 0 else 0
            out_y_end = int(q-0.5*int(beta*q)+j*(q-int(beta*q))) if crop_y_end != Y else Y
            assembled[out_y_start:out_y_end, out_x_start:out_x_end] = out_tile[tileROI_y_start:tileROI_y_end, tileROI_x_start:tileROI_x_end] 

    assembled -= assembled.min()
    assembled /= assembled.max()
    assembled *= (ma - mi)
    assembled += mi

    return assembled.astype(np.float32)

def unet_multi_image_from_tiles(learn, in_img, tile_sz=128, scale=4, wsize=3):
    cur_size = in_img.shape[1:3]
    c = in_img.shape[0]
    new_size = (cur_size[0] * scale, cur_size[1] * scale)
    w, h = cur_size

    in_tile = torch.zeros((c, tile_sz // scale, tile_sz // scale))
    out_img = torch.zeros((1, w * scale, h * scale))
    tile_sz //= scale

    for x_tile in range(math.ceil(w / tile_sz)):
        for y_tile in range(math.ceil(h / tile_sz)):
            x_start = x_tile

            x_start = x_tile * tile_sz
            x_end = min(x_start + tile_sz, w)
            y_start = y_tile * tile_sz
            y_end = min(y_start + tile_sz, h)

            in_tile[:, 0:(x_end - x_start), 0:(y_end - y_start)] = tensor(
                in_img[:, x_start:x_end, y_start:y_end])

            img_list = [
                Image(tensor(npzoom(in_tile[i], scale, order=1))[None])
                for i in range(wsize)
            ]
            #img_list += img_list

            tlist = MultiImage(img_list)
            out_tile, _, _ = learn.predict(tlist)

            out_x_start = x_start * scale
            out_x_end = x_end * scale
            out_y_start = y_start * scale
            out_y_end = y_end * scale

            #print("out: ", out_x_start, out_y_start, ",", out_x_end, out_y_end)
            in_x_start = 0
            in_y_start = 0
            in_x_end = (x_end - x_start) * scale
            in_y_end = (y_end - y_start) * scale
            #print("tile: ",in_x_start, in_y_start, ",", in_x_end, in_y_end)

            out_img[:, out_x_start:out_x_end, out_y_start:
                    out_y_end] = out_tile.data[:, in_x_start:in_x_end,
                                               in_y_start:in_y_end]
    return out_img



# take float in with info about mi,ma,max in and spits out (0-1.0)
def unet_image_from_tiles_blend(learn, in_img, tile_sz=256, scale=4, overlap_pct=5.0, img_info=None):
    n_frames = in_img.shape[0]

    if img_info:
        mi, ma, imax, real_max = [img_info[fld] for fld in ['mi','ma','img_max','real_max']]
        in_img /= real_max
        # in_img = ((in_img - mi) / (ma - mi + 1e-20)).clip(0.,1.)
    else:
        mi, ma, imax, real_max = 0., 1., 1., 1.

    in_img  = np.stack([npzoom(in_img[i], scale, order=1) for i in range(n_frames)])
    overlap = int(tile_sz*(overlap_pct/100.) // 2 * 2)
    step_sz = tile_sz - overlap
    h,w = in_img.shape[1:3]
    assembled = np.zeros((h,w))

    x_seams = set()
    y_seams = set()

    for x_tile in range(0,math.ceil(w/step_sz)):
        for y_tile in range(0,math.ceil(h/step_sz)):
            x_start = x_tile*step_sz
            x_end = min(x_start + tile_sz, w)
            y_start = y_tile*step_sz
            y_end = min(y_start + tile_sz, h)
            src_tile = in_img[:,y_start:y_end,x_start:x_end]


            in_tile = torch.zeros((tile_sz, tile_sz, n_frames))
            in_x_size = x_end - x_start
            in_y_size = y_end - y_start
            if (in_y_size, in_x_size) != src_tile.shape[1:3]: set_trace()
            in_tile[0:in_y_size, 0:in_x_size, :] = tensor(src_tile).permute(1,2,0)

            if n_frames > 1:
                img_in = MultiImage([Image(in_tile[:,:,i][None]) for i in range(n_frames)])
            else:
                img_in = Image(in_tile[:,:,0][None])
            y, pred, raw_pred = learn.predict(img_in)

            out_tile = pred.numpy()[0]

            half_overlap = overlap // 2
            left_adj = half_overlap if x_start != 0 else 0
            right_adj = half_overlap if x_end != w else 0
            top_adj = half_overlap if y_start != 0 else 0
            bot_adj = half_overlap if y_end != h else 0

            trim_y_start = y_start + top_adj
            trim_x_start = x_start + left_adj
            trim_y_end = y_end - bot_adj
            trim_x_end = x_end - right_adj

            out_x_start = left_adj
            out_y_start = top_adj
            out_x_end = in_x_size - right_adj
            out_y_end = in_y_size - bot_adj

            assembled[trim_y_start:trim_y_end, trim_x_start:trim_x_end] = out_tile[out_y_start:out_y_end, out_x_start:out_x_end]

            if trim_x_start != 0: x_seams.add(trim_x_start)
            if trim_y_start != 0: y_seams.add(trim_y_end)

    blur_rects = []
    blur_size = 5
    for x_seam in x_seams:
        left = x_seam - blur_size
        right = x_seam + blur_size
        top, bottom = 0, h
        blur_rects.append((slice(top, bottom), slice(left, right)))

    for y_seam in y_seams:
        top = y_seam - blur_size
        bottom = y_seam + blur_size
        left, right = 0, w
        blur_rects.append((slice(top, bottom), slice(left, right)))

    for xs,ys in blur_rects:
        assembled[xs,ys] = gaussian(assembled[xs,ys], sigma=1.0)

    # if assembled.min() < 0: assembled -= assembled.min()
    # assembled += imax
    # assembled *= imax
    # assembled *= (ma - mi)
    # assembled += mi

    return assembled.astype(np.float32).clip(0.,1.)


def unet_image_from_tiles(learn, in_img, tile_sz=128, scale=4):
    cur_size = in_img.shape[1:3]
    c = in_img.shape[0]
    new_size = (cur_size[0] * scale, cur_size[1] * scale)
    w, h = cur_size

    in_tile = torch.zeros((c, tile_sz // scale, tile_sz // scale))
    out_img = torch.zeros((1, w * scale, h * scale))
    tile_sz //= scale

    for x_tile in range(math.ceil(w / tile_sz)):
        for y_tile in range(math.ceil(h / tile_sz)):
            x_start = x_tile

            x_start = x_tile * tile_sz
            x_end = min(x_start + tile_sz, w)
            y_start = y_tile * tile_sz
            y_end = min(y_start + tile_sz, h)

            in_tile[:, 0:(x_end - x_start), 0:(y_end - y_start)] = tensor(
                in_img[:, x_start:x_end, y_start:y_end])
            img = Image(tensor(npzoom(in_tile[0], scale, order=1)[None]))
            out_tile, _, _ = learn.predict(img)

            out_x_start = x_start * scale
            out_x_end = x_end * scale
            out_y_start = y_start * scale
            out_y_end = y_end * scale

            #print("out: ", out_x_start, out_y_start, ",", out_x_end, out_y_end)
            in_x_start = 0
            in_y_start = 0
            in_x_end = (x_end - x_start) * scale
            in_y_end = (y_end - y_start) * scale
            #print("tile: ",in_x_start, in_y_start, ",", in_x_end, in_y_end)

            out_img[:, out_x_start:out_x_end, out_y_start:
                    out_y_end] = out_tile.data[:, in_x_start:in_x_end,
                                               in_y_start:in_y_end]
    return out_img


def tif_predict_movie(learn,
                      tif_in,
                      orig_out='orig.tif',
                      pred_out='pred.tif',
                      size=128,
                      wsize=3):
    im = PIL.Image.open(tif_in)
    im.load()
    times = im.n_frames
    #times = min(times,100)
    imgs = []

    if times < (wsize + 2):
        print(f'skip {tif_in} only {times} frames')
        return

    for i in range(times):
        im.seek(i)
        imgs.append(np.array(im).astype(np.float32) / 255.)
    img_data = np.stack(imgs)

    def pull_frame(i):
        im.seek(i)
        im.load()
        return np.array(im)

    preds = []
    origs = []
    img_max = img_data.max()

    x, y = im.size
    #print(f'tif: x:{x} y:{y} t:{times}')
    for t in progress_bar(list(range(0, times - wsize + 1))):
        img = img_data[t:(t + wsize)].copy()
        img /= img_max

        out_img = unet_multi_image_from_tiles(learn,
                                              img,
                                              tile_sz=size,
                                              wsize=wsize)
        pred = (out_img * 255).cpu().numpy().astype(np.uint8)
        preds.append(pred)
        orig = (img[1][None] * 255).astype(np.uint8)
        origs.append(orig)
    if len(preds) > 0:
        all_y = img_to_uint8(np.concatenate(preds))
        imageio.mimwrite(
            pred_out, all_y,
            bigtiff=True)  #, fps=30, macro_block_size=None) # for mp4
        all_y = img_to_uint8(np.concatenate(origs))
        imageio.mimwrite(orig_out, all_y,
                         bigtiff=True)  #, fps=30, macro_block_size=None)


def czi_predict_movie(learn,
                      czi_in,
                      orig_out='orig.tif',
                      pred_out='pred.tif',
                      size=128,
                      wsize=3):
    with czifile.CziFile(czi_in) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        #times = min(times, 100)
        x, y = proc_shape['X'], proc_shape['Y']
        #print(f'czi: x:{x} y:{y} t:{times} z:{depths}')
        if times < (wsize + 2):
            print(f'skip {czi_in} only {times} frames')
            return

        #folder_name = Path(pred_out).stem
        #folder = Path(folder_name)
        #if folder.exists(): shutil.rmtree(folder)
        #folder.mkdir()

        data = czi_f.asarray().astype(np.float32) / 255.
        preds = []
        origs = []

        img_max = data.max()
        #print(img_max)
        for t in progress_bar(list(range(0, times - wsize + 1))):
            idx = build_index(
                proc_axes, {
                    'T': slice(t, t + wsize),
                    'C': 0,
                    'Z': 0,
                    'X': slice(0, x),
                    'Y': slice(0, y)
                })
            img = data[idx].copy()
            img /= img_max

            out_img = unet_multi_image_from_tiles(learn,
                                                  img,
                                                  tile_sz=size,
                                                  wsize=wsize)
            pred = (out_img * 255).cpu().numpy().astype(np.uint8)
            preds.append(pred)
            #imsave(folder/f'{t}.tif', pred[0])

            orig = (img[wsize // 2][None] * 255).astype(np.uint8)
            origs.append(orig)
        if len(preds) > 0:
            all_y = img_to_uint8(np.concatenate(preds))
            imageio.mimwrite(
                pred_out, all_y,
                bigtiff=True)  #, fps=30, macro_block_size=None) # for mp4

            all_y = img_to_uint8(np.concatenate(origs))
            imageio.mimwrite(orig_out, all_y,
                             bigtiff=True)  #, fps=30, macro_block_size=None)


def generate_movies(dest_dir, movie_files, learn, size, wsize=5):
    for fn in progress_bar(movie_files):
        ensure_folder(dest_dir)
        pred_name = dest_dir/f'{fn.stem}_pred.tif'
        orig_name = dest_dir/f'{fn.stem}_orig.tif'
        if not Path(pred_name).exists():
            if fn.suffix == '.czi':
                #  print(f'czi {fn.stem}')
                czi_predict_movie(learn,
                                  fn,
                                  size=size,
                                  orig_out=orig_name,
                                  pred_out=pred_name,
                                  wsize=wsize)
            elif fn.suffix == '.tif':
                tif_predict_movie(learn,
                                  fn,
                                  size=size,
                                  orig_out=orig_name,
                                  pred_out=pred_name,
                                  wsize=wsize)
                tif_fn = fn
                #  print(f'tif {fn.stem}')
        else:
            print(f'skip: {fn.stem} - doesn\'t exist')


def max_to_use(img):
    return np.iinfo(np.uint8).max if img.dtype == np.uint8 else img.max()


def img_to_uint8(img, img_info=None):
    img = img.copy()
    if img_info and img_info['dtype'] != np.uint8:
        img -= img.min()
        img /= img.max()
        img *= np.iinfo(np.uint8).max
    return img.astype(np.uint8)

def img_to_float(img):
    dtype = img.dtype
    img_max = max_to_use(img)
    img = img.astype(np.float32).copy()
    mi, ma = np.percentile(img, [2,99.99])
    img_range = ma - mi
    real_max = img.max()
    return img, {'img_max': img_max, 'real_max': real_max, 'mi': mi, 'ma': ma, 'dtype':dtype }

def tif_predict_images(learn,
                       tif_in,
                       dest,
                       category,
                       tag=None,
                       size=128,
                       max_imgs=None):
    under_tag = f'_' if tag is None else f'_{tag}_'
    dest_folder = Path(dest / category)
    dest_folder.mkdir(exist_ok=True, parents=True)
    pred_out = dest_folder / f'{tif_in.stem}{under_tag}pred.tif'
    orig_out = dest_folder / f'{tif_in.stem}{under_tag}orig.tif'
    if pred_out.exists():
        print(f'{pred_out.stem} exists')
        return

    im = PIL.Image.open(tif_in)
    im.load()
    times = im.n_frames
    if not max_imgs is None: times = min(max_imgs, times)

    imgs = []

    for i in range(times):
        im.seek(i)
        im.load()
        imgs.append(np.array(im))

    imgs, img_info = img_to_float(np.stack(imgs))

    preds = []

    x, y = im.size
    print(f'tif: x:{x} y:{y} t:{times}')
    for t in progress_bar(list(range(times))):
        img = imgs[t]
        img = img.copy()

    if len(preds) > 0:
        all_y = img_to_uint8(np.concatenate(preds))
        imageio.mimwrite(pred_out, all_y, bigtiff=True)
        shutil.copy(tif_in, orig_out)


def czi_predict_images(learn,
                       czi_in,
                       dest,
                       category,
                       tag=None,
                       size=128,
                       max_imgs=None):
    with czifile.CziFile(czi_in) as czi_f:

        under_tag = f'_' if tag is None else f'_{tag}_'
        dest_folder = Path(dest / category)
        dest_folder.mkdir(exist_ok=True, parents=True)

        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        if not max_imgs is None: times = min(max_imgs, times)

        x, y = proc_shape['X'], proc_shape['Y']

        data, img_info = img_to_float(czi_f.asarray())
        orig_dtype = data.dtype

        img_max = data.max()
        print(f'czi: x:{x} y:{y} t:{times} c:{channels} z:{depths} {img_max}')

        channels_bar = progress_bar(
            range(channels)) if channels > 1 else range(channels)
        depths_bar = progress_bar(
            range(depths)) if depths > 1 else range(depths)
        times_bar = progress_bar(range(times)) if times > 1 else range(times)

        for c in channels_bar:
            for z in depths_bar:
                preds = []
                origs = []
                if (depths > 1) or (channels > 1):
                    pred_out = dest_folder / f'{czi_in.stem}_c{c:02d}_z{z:02d}_{under_tag}_pred.tif'
                    orig_out = dest_folder / f'{czi_in.stem}_c{c:02d}_z{z:02d}_{under_tag}_orig.tif'
                else:
                    pred_out = dest_folder / f'{czi_in.stem}_{under_tag}_pred.tif'
                    orig_out = dest_folder / f'{czi_in.stem}_{under_tag}_orig.tif'
                if not pred_out.exists():
                    for t in times_bar:
                        idx = build_index(
                            proc_axes, {
                                'T': t,
                                'C': c,
                                'Z': z,
                                'X': slice(0, x),
                                'Y': slice(0, y)
                            })
                        img = data[idx].copy()
                        pred = unet_image_from_tiles_blend(learn,
                                                           img[None],
                                                           tile_sz=size,
                                                           img_info=img_info)
                        preds.append(pred[None])
                        origs.append(img[None])

                    if len(preds) > 0:
                        all_y = img_to_uint8(np.concatenate(preds))
                        imageio.mimwrite(pred_out, all_y, bigtiff=True)
                        all_y = img_to_uint8(np.concatenate(origs))
                        imageio.mimwrite(orig_out, all_y, bigtiff=True)


def generate_tifs(src, dest, learn, size, tag=None, max_imgs=None):
    for fn in progress_bar(src):
        category = fn.parts[-3]
        try:
            if fn.suffix == '.czi':
                czi_predict_images(learn,
                                fn,
                                dest,
                                category,
                                size=size,
                                tag=tag,
                                max_imgs=max_imgs)
            elif fn.suffix == '.tif':
                tif_predict_images(learn,
                                fn,
                                dest,
                                category,
                               size=size,
                                tag=tag,
                                max_imgs=max_imgs)
        except Exception as e:
             print(f'exception with {fn.stem}')
             print(e)


def ensure_folder(fldr, clean=False):
    fldr = Path(fldr)
    if fldr.exists() and clean:
        print(f'wiping {fldr.stem} in 5 seconds')
        sleep(5.)
        shutil.rmtree(fldr)
    if not fldr.exists(): fldr.mkdir(parents=True, mode=0o775, exist_ok=True)
    return fldr


def subfolders(p):
    return [sub for sub in p.iterdir() if sub.is_dir()]


def build_tile_info(data, tile_sz, train_samples, valid_samples, only_categories=None, skip_categories=None):
    if skip_categories == None: skip_categories = []
    if only_categories == None: only_categories = []
    if only_categories: skip_categories = [c for c in skip_categories if c not in only_categories]

    def get_category(p):
        return p.parts[-2]

    def get_mode(p):
        return p.parts[-3]

    def is_only(fn):
        return (not only_categories) or (get_category(fn) in only_categories)

    def is_skip(fn):
        return get_category(fn) in skip_categories

    def get_img_size(p):
        with PIL.Image.open(p) as img:
            h,w = img.size
        return h,w

    all_files = [fn for fn in list(data.glob('**/*.tif')) if is_only(fn) and not is_skip(fn)]
    img_sizes = {str(p):get_img_size(p) for p in progress_bar(all_files)}

    files_by_mode = {}

    for p in progress_bar(all_files):
        category = get_category(p)
        mode = get_mode(p)
        mode_list = files_by_mode.get(mode, {})
        cat_list = mode_list.get(category, [])
        cat_list.append(p)
        mode_list[category] = cat_list
        files_by_mode[mode] = mode_list

    def pull_random_tile_info(mode, tile_sz):
        files_by_cat = files_by_mode[mode]
        category=random.choice(list(files_by_cat.keys()))
        img_file=random.choice(files_by_cat[category])
        h,w = img_sizes[str(img_file)]
        return {'mode': mode,'category': category,'fn': img_file, 'tile_sz': tile_sz, 'h': h, 'w':w}


    tile_infos = []
    for i in range(train_samples):
        tile_infos.append(pull_random_tile_info('train', tile_sz))
    for i in range(valid_samples):
        tile_infos.append(pull_random_tile_info('valid', tile_sz))

    tile_df = pd.DataFrame(tile_infos)[['mode','category','tile_sz','h','w','fn']]
    return tile_df


def draw_tile(img, tile_sz):
    max_x,max_y = img.shape
    x = random.choice(range(max_x-tile_sz)) if max_x > tile_sz else 0
    y = random.choice(range(max_y-tile_sz)) if max_y > tile_sz else 0
    xs = slice(x,min(x+tile_sz, max_x))
    ys = slice(y,min(y+tile_sz, max_y))
    tile = img[xs,ys].copy()
    return tile, (xs,ys)

def check_tile(img, thresh, thresh_pct):
    return (img > thresh).mean() > thresh_pct

def draw_random_tile(img_data, tile_sz, thresh, thresh_pct):
    max_tries = 200

    found_tile = False
    tries = 0
    while not found_tile:
        tile, (xs,ys) = draw_tile(img_data, tile_sz)
        found_tile = check_tile(tile, thresh, thresh_pct)
        # found_tile = True
        tries += 1
        if tries > (max_tries/2): thresh_pct /= 2
        if tries > max_tries: found_tile = True
    box = [xs.start, ys.start, xs.stop, ys.stop]
    return PIL.Image.fromarray(tile), box

def generate_tiles(dest_dir, tile_info, scale=4, crap_dirs=None, crap_func=None):
    tile_data = []
    dest_dir = ensure_folder(dest_dir)
    shutil.rmtree(dest_dir)
    if crap_dirs:
        for crap_dir in crap_dirs.values():
            if crap_dir:
                shutil.rmtree(crap_dir)

    last_fn = None
    tile_info = tile_info.sort_values('fn')
    for row_id, tile_stats in progress_bar(list(tile_info.iterrows())):
        mode = tile_stats['mode']
        fn = tile_stats['fn']
        tile_sz = tile_stats['tile_sz']
        category = tile_stats['category']
        if fn != last_fn:
            img = PIL.Image.open(fn)
            img_data = np.array(img)
            img_max = img_data.max()
            img_data /= img_max

            thresh = 0.01
            thresh_pct = (img_data.mean() > 1) * 0.5
            last_fn = fn
            tile_folder = ensure_folder(dest_dir/mode/category)
        if crap_dirs:
            crap_dir = crap_dirs[tile_sz]
            crap_tile_folder = ensure_folder(crap_dir/mode/category) if crap_dir else None
        else:
            crap_tile_folder = None
            crap_dir = None

        crop_img, box = draw_random_tile(img_data, tile_sz, thresh, thresh_pct)
        crop_img.save(tile_folder/f'{row_id:05d}_{fn.stem}.tif')
        if crap_func and crap_dir:
            crap_img = crap_func(crop_img, scale=scale)
            crap_img.save(crap_tile_folder/f'{row_id:05d}_{fn.stem}.tif')
        tile_data.append({'tile_id': row_id, 'category': category, 'mode': mode, 'tile_sz': tile_sz, 'box': box, 'fn': fn})
    pd.DataFrame(tile_data).to_csv(dest_dir/'tiles.csv', index=False)
