"functions to create synthetic low res images"
import numpy as np
import czifile
import PIL
import random
from skimage.util import random_noise, img_as_ubyte
from skimage import filters
from skimage.io import imsave
from scipy.ndimage.interpolation import zoom as npzoom
from .czi import get_czi_shape_info, build_index, is_movie
from .utils import ensure_folder
from fastai.vision import *

__all__ = ['speckle_crap', 'classic_crap', 'czi_movie_to_synth', 'tif_movie_to_synth']


def speckle_crap(img):
    img = random_noise(img, mode='speckle', var=0.02, clip=True)
    return img


def classic_crap(img):
    img = random_noise(img, mode='salt', amount=0.005)
    img = random_noise(img, mode='pepper', amount=0.005)
    lvar = filters.gaussian(img, sigma=5) + 1e-6
    img = random_noise(img, mode='localvar', local_vars=lvar * 0.5)
    return img

def micro_crappify(img, gauss_sigma = 1, poisson_loop=10): #data: 1 frame; play with sigma?
    x = img * 255.
    for n in range(poisson_loop):
        x = np.random.poisson(np.maximum(0,x).astype(np.int))
    x = x.astype(np.float32)
    noise = np.random.normal(0,gauss_sigma,size=x.shape).astype(np.float32)
    x = np.maximum(0,x+noise)
    x -= x.min()
    x /= x.max()
    return x

def new_crappify(img, add_noise=True, scale=4):
    "a crappifier for our microscope images"
    if add_noise:
        img = random_noise(img, mode='salt', amount=0.005)
        img = random_noise(img, mode='pepper', amount=0.005)
        lvar = filters.gaussian(img, sigma=5)
        img = random_noise(img, mode='localvar', local_vars=lvar * 0.5)
    img_down = npzoom(img, 1 / scale, order=1)
    img_up = npzoom(img_down, scale, order=1)
    return img_down, img_up


def czi_data_to_tifs(data, axes, shape, crappify, max_scale=1.05):
    np.warnings.filterwarnings('ignore')
    lr_imgs = {}
    lr_up_imgs = {}
    hr_imgs = {}
    channels = shape['C']
    depths = shape['Z']
    times = shape['T']
    x, y = shape['X'], shape['Y']

    for channel in range(channels):
        for depth in range(depths):
            for time_col in range(times):
                try:
                    idx = build_index(
                        axes, {
                            'T': time_col,
                            'C': channel,
                            'Z': depth,
                            'X': slice(0, x),
                            'Y': slice(0, y)
                        })
                    img = data[idx].astype(np.float).copy()
                    img_max = img.max() * max_scale
                    if img_max == 0:
                        continue  #do not save images with no contents.
                    img /= img_max
                    down_img, down_up_img = crappify(img)
                except:
                    continue

                tag = (channel, depth, time_col)
                img = img_as_ubyte(img)
                pimg = PIL.Image.fromarray(img, mode='L')
                small_img = PIL.Image.fromarray(img_as_ubyte(down_img))
                big_img = PIL.Image.fromarray(img_as_ubyte(down_up_img))
                hr_imgs[tag] = pimg
                lr_imgs[tag] = small_img
                lr_up_imgs[tag] = big_img

    np.warnings.filterwarnings('default')
    return hr_imgs, lr_imgs, lr_up_imgs

def img_data_to_tifs(data, times, crappify, max_scale=1.05):
    np.warnings.filterwarnings('ignore')
    lr_imgs = {}
    lr_up_imgs = {}
    hr_imgs = {}
    for time_col in range(times):
        try:
            img = data[time_col].astype(np.float).copy()
            img_max = img.max() * max_scale
            if img_max == 0: continue  #do not save images with no contents.
            img /= img_max
            down_img, down_up_img = crappify(img)
        except:
            continue

        tag = (0, 0, time_col)
        img = img_as_ubyte(img)
        pimg = PIL.Image.fromarray(img, mode='L')
        small_img = PIL.Image.fromarray(img_as_ubyte(down_img))
        big_img = PIL.Image.fromarray(img_as_ubyte(down_up_img))
        hr_imgs[tag] = pimg
        lr_imgs[tag] = small_img
        lr_up_imgs[tag] = big_img

    np.warnings.filterwarnings('default')
    return hr_imgs, lr_imgs, lr_up_imgs


def tif_to_synth(tif_fn,
                 dest,
                 category,
                 mode,
                 single=True,
                 multi=False,
                 num_frames=5,
                 max_scale=1.05,
                 crappify_func=None):
    img = PIL.Image.open(tif_fn)
    n_frames = img.n_frames

    if crappify_func is None: crappify_func = new_crappify
    for i in range(n_frames):
        img.seek(i)
        img.load()
        data = np.array(img).copy()

        hr_imgs, lr_imgs, lr_up_imgs = img_data_to_tifs(data,
                                                        n_frames,
                                                        crappify_func,
                                                        max_scale=max_scale)
        if single:
            save_tiffs(tif_fn, dest, category, mode, hr_imgs, lr_imgs,
                       lr_up_imgs)
        if multi:
            save_movies(tif_fn, dest, category, mode, hr_imgs, lr_imgs,
                        lr_up_imgs, num_frames)


def save_tiffs(czi_fn, dest, category, mode, hr_imgs, lr_imgs, lr_up_imgs):
    hr_dir = dest / 'hr' / mode / category
    lr_dir = dest / 'lr' / mode / category
    lr_up_dir = dest / 'lr_up' / mode / categroy
    base_name = czi_fn.stem
    for tag, hr in hr_imgs.items():
        lr = lr_imgs[tag]
        lr_up = lr_up_imgs[tag]

        channel, depth, time_col = tag
        save_name = f'{channel:02d}_{depth:02d}_{time_col:06d}_{base_name}.tif'
        hr_name, lr_name, lr_up_name = [
            d / save_name for d in [hr_dir, lr_dir, lr_up_dir]
        ]
        if not hr_name.exists(): hr.save(hr_name)
        if not lr_name.exists(): lr.save(lr_name)
        if not lr_up_name.exists(): lr_up.save(lr_up_name)


def save_movies(czi_fn, dest, category, mode, hr_imgs, lr_imgs, lr_up_imgs,
                num_frames):
    print('WTF save_movies is empty dude')
    print('*****', czi_fn)



def draw_tile(img, tile_sz):
    max_x,max_y = img.shape
    x = random.choice(range(max_x-tile_sz)) if max_x > tile_sz else 0
    y = random.choice(range(max_y-tile_sz)) if max_y > tile_sz else 0
    xs = slice(x,min(x+tile_sz, max_x))
    ys = slice(y,min(y+tile_sz, max_y))
    tile = img[xs,ys].copy()
    return tile, (xs,ys)

def draw_tile_bounds(img, bounds):
    xs,ys = bounds
    tile = img[xs,ys].copy()
    return tile


def save_img(fn, img):
    if len(img.shape) == 2:
        np.warnings.filterwarnings('ignore')
        PIL.Image.fromarray(img_as_ubyte(img), mode='L').save(f'{fn}.tif')
        np.warnings.filterwarnings('default')
    else:
        img8 = (img * 255.).astype(np.uint8)
        np.save(fn.with_suffix('.npy'), img8, allow_pickle=False)


def find_interesting_region(img, tile_sz):
    max_tries = 200
    thresh = 0.01
    thresh_pct = (img > thresh).mean() * 1.5
    tile, bounds = draw_tile(img, tile_sz)
    for tries in range(max_tries):
        if check_tile(tile, thresh, thresh_pct): break
        elif tries > (max_tries//2): thresh_pct /= 2
    return np.array([bounds[0].start, bounds[1].start, bounds[0].stop, bounds[1].stop])

def make_multi_tiles(tiles, category, n_tiles, scale, hr_img, lr_imgs, lrup_imgs,
                     save_name, dest, n_frames, mode, axis):
    if not tiles: return
    for tile_sz in tiles:
        for i in range(n_tiles):
            tile_name = f'{i:03d}_{save_name}'
            hr_dir = ensure_folder(dest/ f'hr_m{axis}_{n_frames:02d}_t_{tile_sz:04d}' / mode / category)
            lr_dir = ensure_folder(dest/ f'lr_m{axis}_{n_frames:02d}_t_{tile_sz:04d}' / mode / category)
            lrup_dir = ensure_folder(dest/ f'lrup_m{axis}_{n_frames:02d}_t_{tile_sz:04d}' / mode / category)

            box = find_interesting_region(hr_img, tile_sz)
            box //= scale
            lr_box = box.copy()
            box *= scale
            xs, ys = slice(box[0], box[2]), slice(box[1], box[3])
            lr_xs, lr_ys = slice(lr_box[0], lr_box[2]), slice(lr_box[1], lr_box[3])

            save_img(hr_dir/tile_name, hr_img[xs,ys])
            save_img(lr_dir/tile_name, lr_imgs[:,lr_xs,lr_ys])
            save_img(lrup_dir/tile_name, lrup_imgs[:,xs,ys])


def czi_movie_to_synth(czi_fn,
                       dest,
                       category,
                       mode,
                       single=True,
                       multi=False,
                       tiles=None,
                       scale=4,
                       n_tiles=5,
                       n_frames=5,
                       crappify_func=None):
    base_name = czi_fn.stem
    if single:
        hr_dir = ensure_folder(dest / 'hr' / mode / category)
        lr_dir = ensure_folder(dest / 'lr' / mode / category)
        lrup_dir = ensure_folder(dest / 'lrup' / mode / category)
        with czifile.CziFile(czi_fn) as czi_f:
            data = czi_f.asarray()
            axes, shape = get_czi_shape_info(czi_f)
            channels = shape['C']
            depths = shape['Z']
            times = shape['T']
            x,y = shape['X'], shape['Y']

            for channel in range(channels):
                for depth in range(depths):
                    for t in range(times):
                        save_name = f'{channel:02d}_{depth:02d}_{t:06d}_{base_name}'
                        idx = build_index( axes, {'T': t, 'C':channel, 'Z': depth, 'X':slice(0,x), 'Y':slice(0,y)})
                        img_data = data[idx].astype(np.float32).copy()
                        img_max = img_data.max()
                        if img_max != 0: img_data /= img_max

                        image_to_synth(img_data, dest, mode, hr_dir, lr_dir, lrup_dir, save_name,
                                    single, multi, tiles, n_tiles, n_frames, scale, crappify_func)

    if multi:
        with czifile.CziFile(czi_fn) as czi_f:
            proc_axes, proc_shape = get_czi_shape_info(czi_f)
            channels = proc_shape['C']
            depths = proc_shape['Z']
            times = proc_shape['T']
            x,y = proc_shape['X'], proc_shape['Y']
            data = czi_f.asarray()
            for channel in range(channels):
                img_max = None
                timerange = list(range(0,times-n_frames+1, n_frames))
                if len(timerange) >= n_frames:
                    hr_mt_dir = ensure_folder(dest / f'hr_mt_{n_frames:02d}' / mode / category)
                    lr_mt_dir = ensure_folder(dest / f'lr_mt_{n_frames:02d}' / mode / category)
                    lrup_mt_dir = ensure_folder(dest / f'lrup_mt_{n_frames:02d}' / mode / category)

                    for time_col in timerange:
                        save_name = f'{channel:02d}_T{time_col:05d}-{(time_col+n_frames-1):05d}_{base_name}'
                        idx = build_index(proc_axes, {'T': slice(time_col,time_col+n_frames), 'C': channel, 'X':slice(0,x),'Y':slice(0,y)})
                        img_data = data[idx].astype(np.float32).copy()
                        img_max = img_data.max()
                        if img_max != 0: img_data /= img_max

                        _,h,w = img_data.shape
                        adjh, adjw = (h//4) * 4, (w//4)*4
                        hr_imgs = img_data[:,0:adjh, 0:adjw]
                        lr_imgs = []
                        lrup_imgs = []

                        for i in range(hr_imgs.shape[0]):
                            hr_img = hr_imgs[i]
                            crap_img = crappify_func(hr_img).astype(np.float32).copy() if crappify_func else hr_img
                            lr_img = npzoom(crap_img, 1/scale, order=0).astype(np.float32).copy()
                            lr_imgs.append(lr_img)
                            lrup_img = npzoom(lr_img, scale, order=0).astype(np.float32).copy()
                            lrup_imgs.append(lrup_img)

                        lr_imgs = np.array(lr_imgs).astype(np.float32).copy()
                        lrup_imgs = np.array(lrup_imgs).astype(np.float32).copy()
                        hr_img = hr_imgs[hr_imgs.shape[0]//2].astype(np.float32).copy()
                        hr_mt_name, lr_mt_name, lrup_mt_name = [d / save_name for d in [hr_mt_dir, lr_mt_dir, lrup_mt_dir]]
                        np.save(hr_mt_name, hr_img)
                        np.save(lr_mt_name, lr_imgs)
                        np.save(lrup_mt_name, lrup_imgs)

                        make_multi_tiles(tiles, category, n_tiles, scale, hr_img, lr_imgs, lrup_imgs,
                                         save_name, dest, n_frames, mode, 't')

                if depths >= n_frames:
                    hr_mz_dir = ensure_folder(dest / f'hr_mz_{n_frames:02d}' / mode / category)
                    lr_mz_dir = ensure_folder(dest / f'lr_mz_{n_frames:02d}' / mode / category)
                    lrup_mz_dir = ensure_folder(dest / f'lrup_mz_{n_frames:02d}' / mode / category)

                    mid_depth = depths // 2
                    start_depth = mid_depth - n_frames//2
                    end_depth = mid_depth + n_frames//2
                    depthrange = slice(start_depth,end_depth+1)
                    save_name = f'{channel:02d}_Z{start_depth:05d}-{end_depth:05d}_{base_name}'
                    idx = build_index(proc_axes, {'Z': depthrange, 'C': channel, 'X':slice(0,x),'Y':slice(0,y)})
                    img_data = data[idx].astype(np.float32).copy()
                    img_max = img_data.max()
                    if img_max != 0: img_data /= img_max

                    _,h,w = img_data.shape
                    adjh, adjw = (h//4) * 4, (w//4)*4
                    hr_imgs = img_data[:,0:adjh, 0:adjw]
                    lr_imgs = []
                    lrup_imgs = []

                    for i in range(hr_imgs.shape[0]):
                        hr_img = hr_imgs[i]
                        crap_img = crappify_func(hr_img).astype(np.float32).copy() if crappify_func else hr_img
                        lr_img = npzoom(crap_img, 1/scale, order=0).astype(np.float32).copy()
                        lr_imgs.append(lr_img)
                        lrup_img = npzoom(lr_img, scale, order=0).astype(np.float32).copy()
                        lrup_imgs.append(lrup_img)

                    lr_imgs = np.array(lr_imgs).astype(np.float32).copy()
                    lrup_imgs = np.array(lrup_imgs).astype(np.float32).copy()
                    hr_img = hr_imgs[hr_imgs.shape[0]//2].astype(np.float32).copy()
                    hr_mz_name, lr_mz_name, lrup_mz_name = [d / save_name for d in [hr_mz_dir, lr_mz_dir, lrup_mz_dir]]
                    np.save(hr_mz_name, hr_img)
                    np.save(lr_mz_name, lr_imgs)
                    np.save(lrup_mz_name, lrup_imgs)

                    make_multi_tiles(tiles, category, n_tiles, scale, hr_img, lr_imgs, lrup_imgs,
                                        save_name, dest, n_frames, mode, 'z')


def tif_movie_to_synth(tif_fn,
                       dest,
                       category,
                       mode,
                       single=True,
                       multi=False,
                       tiles=None,
                       scale=4,
                       n_tiles=5,
                       n_frames=5,
                       crappify_func=None):
    hr_dir = ensure_folder(dest / 'hr' / mode / category)
    lr_dir = ensure_folder(dest / 'lr' / mode / category)
    lrup_dir = ensure_folder(dest / 'lrup' / mode / category)
    base_name = tif_fn.stem

    img = PIL.Image.open(tif_fn)
    n_frames = img.n_frames

    with PIL.Image.open(tif_fn) as img:
        channels = 1
        depths = img.n_frames
        times = 1

        for channel in range(channels):
            for depth in range(depths):
                for t in range(times):
                    save_name = f'{channel:02d}_{depth:02d}_{t:06d}_{base_name}'
                    img.seek(depth)
                    img.load()
                    img_data = np.array(img).astype(np.float32).copy()
                    img_max = img_data.max()
                    if img_max != 0: img_data /= img_max

                    image_to_synth(img_data, dest, mode, hr_dir, lr_dir, lrup_dir, save_name,
                                   single, multi, tiles, n_tiles, n_frames, scale, crappify_func)



def check_tile(img, thresh, thresh_pct):
    return (img > thresh).mean() > thresh_pct

def image_to_synth(img_data, dest, mode, hr_dir, lr_dir, lrup_dir, save_name, single, multi, tiles, n_tiles, n_frames, scale, crappify_func):
    if len(img_data.shape) > 2:
        if len(img_data.shape) == 3:
            img_data = img_data[:,:,0]
        else:
            print(f'skip {save_name} multichannel')
            return

    h,w = img_data.shape
    adjh, adjw = (h//4) * 4, (w//4)*4
    hr_img = img_data[0:adjh, 0:adjw]

    crap_img = crappify_func(hr_img).astype(np.float32).copy() if crappify_func else hr_img
    lr_img = npzoom(crap_img, 1/scale, order=0).astype(np.float32).copy()
    lrup_img = npzoom(lr_img, scale, order=0).astype(np.float32).copy()

    if single:
        hr_name, lr_name, lrup_name = [d / save_name for d in [hr_dir, lr_dir, lrup_dir]]
        save_img(hr_name, hr_img)
        save_img(lr_name, lr_img)
        save_img(lrup_name, lrup_img)

    if tiles:
        for tile_sz in tiles:
            hr_tile_dir = ensure_folder(dest/f'hr_t_{tile_sz}'/mode)
            lr_tile_dir = ensure_folder(dest/f'lr_t_{tile_sz}'/mode)
            lrup_tile_dir = ensure_folder(dest/f'lrup_t_{tile_sz}'/mode)

            tile_id = 0
            tries = 0
            max_tries = 200
            thresh = 0.01
            thresh_pct = (hr_img > thresh).mean() * 1.5
            while tile_id < n_tiles:
                hr_tile, bounds = draw_tile(hr_img, tile_sz)
                if check_tile(hr_tile, thresh, thresh_pct):
                    tile_name = f'{save_name}_{tile_id:03d}'
                    hr_tile_name, lr_tile_name, lrup_tile_name = [d / tile_name for d
                                                                in [hr_tile_dir, lr_tile_dir, lrup_tile_dir]]
                    crap_tile = draw_tile_bounds(crap_img, bounds=bounds)
                    lr_tile = npzoom(crap_tile, 1/scale, order=0).astype(np.float32).copy()
                    lrup_tile = npzoom(lr_tile, scale, order=0).astype(np.float32).copy()
                    save_img(hr_tile_name, hr_tile)
                    save_img(lr_tile_name, lr_tile)
                    save_img(lrup_tile_name, lrup_tile)
                    tile_id += 1
                    tries = 0
                else:
                    tries += 1
                    if tries > (max_tries//2):
                        thresh_pct /= 2
                    if tries > max_tries:
                        print(f'timed out on {save_name}')
                        tries = 0
                        tile_id += 1
