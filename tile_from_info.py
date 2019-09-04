import yaml

from fastai.script import *
from fastai.vision import *
from bpho import *
from pathlib import Path
from fastprogress import master_bar, progress_bar

from time import sleep
from pdb import set_trace
import shutil
import PIL
import czifile
PIL.Image.MAX_IMAGE_PIXELS = 99999999999999

def new_crap(x, gauss_sigma:uniform=0.01, pscale:uniform=10, scale=4, upsample=False):
    from skimage.transform import rescale
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

def no_crap(img, scale=4, upsample=False):
    from skimage.transform import rescale
    x = np.array(img)
    multichannel = len(x.shape) > 2
    x = rescale(x, scale=1/scale, order=1, multichannel=multichannel)
    x *= np.iinfo(np.uint8).max
    return PIL.Image.fromarray(x.astype(np.uint8))

def default_crap(img, scale=4, upsample=True, pscale=12, gscale=0.0001):
    from skimage.transform import rescale


    # set_trace()
    # pull pixel data from PIL image
    x = np.array(img)
    multichannel = len(x.shape) > 2

    # remember the range of the original image
    dmax = np.iinfo(x.dtype).max
    img_max = x.max()

    # normalize 0-1.0
    x = x.astype(np.float32)
    x /= float(img_max)

    # downsample the image
    x = rescale(x, scale=1/scale, order=1, multichannel=multichannel)

    # add poisson and gaussian noise to the image
    x = np.random.poisson(x * pscale)/pscale
    x += np.random.normal(0, gscale, size=x.shape)
    x = np.maximum(0,x)

    # normalize to 0-1.0
    x /= x.max()

    # back to original range of incoming image
    x *= img_max

    if upsample: x = rescale(x, scale=scale, order=0, multichannel=multichannel)

    x /= dmax
    x *= np.iinfo(np.uint8).max
    return PIL.Image.fromarray(x.astype(np.uint8))

def need_cache_flush(tile_stats, last_stats):
    if last_stats is None: return True
    if tile_stats['fn'] != last_stats['fn']: return True
    return False

def get_tile_puller(tile_stat, crap_func, t_frames, z_frames):
    fn = tile_stat['fn']
    ftype = tile_stat['ftype']
    nz = tile_stat['nz']
    nt = tile_stat['nt']

    half_z = z_frames // 2
    half_t = t_frames // 2

    if ftype == 'czi':
        img_f = czifile.CziFile(fn)
        proc_axes, proc_shape = get_czi_shape_info(img_f)
        img_data = img_f.asarray()
        img_data = img_data.astype(np.float32)

        def czi_get(istat):
            c,z,t,x,y,mi,ma,is_uint8,rmax,all_rmax,all_ma = [istat[fld] for fld in ['c','z','t','x','y','mi','ma','uint8','rmax','all_rmax','all_ma']]
            if is_uint8:
                mi, ma, rmax = 0., 255.0, 255.0
                all_ma, all_rmax = 255.0, 255.0

            t_slice = slice(t-half_t, t+half_t+1) if half_t > 0 else t
            z_slice = slice(z-half_z, z+half_z+1) if half_z > 0 else z
            idx = build_index(
                proc_axes, {
                    'C': c,
                    'T': t_slice,
                    'Z': z_slice,
                    'X': slice(0, x),
                    'Y': slice(0, y)
                })
            img = img_data[idx].copy()
            img /= all_rmax
            if len(img.shape) <= 2: img = img[None]
            return img

        img_get = czi_get
        img_get._to_close = img_f
    else:
        pil_img = PIL.Image.open(fn)
        def pil_get(istat):
            c,z,t,x,y,mi,ma,is_uint8,rmax,all_rmax,all_ma = [istat[fld] for fld in ['c','z','t','x','y','mi','ma','uint8','rmax','all_rmax','all_ma']]
            if half_t > 0: n_start, n_end = t-half_t, t+half_t+1
            elif half_z > 0: n_start, n_end = z-half_z, z+half_z+1
            else: n_start, n_end = 0,1

            if is_uint8:
                mi, ma, rmax = 0., 255.0, 255.0
                all_ma, all_rmax = 255.0, 255.0

            img_array = []
            for ix in range(n_start, n_end):
                pil_img.seek(ix)
                pil_img.load()
                img = np.array(pil_img)
                if len(img.shape) > 2: img = img[:,:,0]
                img_array.append(img.copy())

            img = np.stack(img_array)
            img = img.astype(np.float32)
            img /= all_rmax
            return img

        img_get = pil_get
        img_get._to_close = pil_img


    def puller(istat, tile_folder, crap_folder, close_me=False):
        if close_me:
            img_get._to_close.close()
            return None

        id = istat['index']
        fn = Path(istat['fn'])
        tile_sz = istat['tile_sz']
        c,z,t,x,y,mi,ma,is_uint8,rmax = [istat[fld] for fld in ['c','z','t','x','y','mi','ma','uint8','rmax']]

        raw_data = img_get(istat)
        img_data = (np.iinfo(np.uint8).max * raw_data).astype(np.uint8)

        thresh = np.percentile(img_data, 2)
        thresh_pct = (img_data > thresh).mean() * 0.30

        frame_count = img_data.shape[0]
        mid_frame = frame_count // 2
        crop_img, box = draw_random_tile(img_data[mid_frame], istat['tile_sz'], thresh, thresh_pct)
        crop_img.save(tile_folder/f'{id:06d}_{fn.stem}.tif')
        if crap_func and crap_folder:
            if frame_count > 1:
                crap_data = []
                for i in range(frame_count):
                    frame_img = PIL.Image.fromarray(img_data[i, box[1]:box[3], box[0]:box[2]])
                    crap_frame = crap_func(crop_img)
                    crap_data.append(np.array(crap_frame))
                multi_array = np.stack(crap_data)
                np.save(crap_folder/f'{id:06d}_{fn.stem}.npy', multi_array)
            else:
                crap_img = crap_func(crop_img)
                crap_img.save(crap_folder/f'{id:06d}_{fn.stem}.tif')

        info = dict(istat)
        info['id'] = id
        info['box'] = box
        info['tile_sz'] = tile_sz
        crop_data = np.array(crop_img)
        info['after_mean'] = crop_data.mean()
        info['after_sd'] = crop_data.std()
        info['after_max'] = crop_data.max()
        info['after_min'] = crop_data.min()
        return info

    return puller

def check_info(info, t_frames, z_frames):
    t_space = t_frames // 2
    z_space = z_frames // 2

    z_ok = (info['nz'] >= z_frames) and (info['z'] >= z_space) and (info['z'] < (info['nz']-z_space))
    t_ok = (info['nt'] >= t_frames) and (info['t'] >= t_space) and (info['t'] < (info['nt']-t_space))

    return t_ok and z_ok



@call_parse
def main(out: Param("dataset folder", Path, required=True),
         info: Param('info file', Path, required=True),
         tile: Param('generated tile size', int, nargs='+', required=True),
         n_train: Param('number of train tiles', int, required=True),
         n_valid: Param('number of validation tiles', int, required=True),
         crap_func: Param('crappifier name', str) = 'no_crap',
         n_frames: Param('number of frames', int) = 1,
         lr_type: Param('training input, (s)ingle, (t) multi or (z) multi', str) = 's',
         scale: Param('amount to scale', int) = 4,
         ftypes: Param('ftypes allowed e.g. - czi, tif', str, nargs='+') = None,
         upsample: Param('use upsample', action='store_true') = False,
         only: Param('limit to these categories', nargs='+') = None,
         skip: Param("categories to skip", str, nargs='+') = ['random', 'ArgoSIMDL'],
         clean: Param("wipe existing data first", action='store_true') = False):
    "generate tiles from source tiffs"
    up = 'up' if upsample else ''

    if lr_type not in ['s','t','z']:
        print('lr_type should be s, t or z')
        return 1

    if lr_type == 's':
        z_frames, t_frames = 1, 1
    elif lr_type == 't':
        z_frames, t_frames = 1, n_frames
    elif lr_type == 'z':
        z_frames, t_frames = n_frames, 1

    crap_func = eval(crap_func)
    if not crap_func is None:
        if not callable(crap_func):
            print('crap_func is not callable')
            crap_func = None
        else:
            crap_func = partial(crap_func, scale=scale, upsample=upsample)

    out = ensure_folder(out)
    if clean:
        shutil.rmtree(out)

    info = pd.read_csv(info)

    if ftypes: info = info.loc[info.ftype.isin(ftypes)]
    if only: info = info.loc[info.category.isin(only)]
    elif skip: info = info.loc[~info.category.isin(skip)]

    info = info.loc[info.nz >= z_frames]
    info = info.loc[info.nt >= t_frames]

    tile_infos = []
    for mode, n_samples in [('train', n_train),('valid', n_valid)]:
        mode_info = info.loc[info.dsplit == mode]
        categories = list(mode_info.groupby('category'))
        files_by_category  = {c:list(info.groupby('fn')) for c,info in categories}

        for i in range(n_samples):
            category, cat_df = random.choice(categories)
            fn, item_df = random.choice(files_by_category[category])
            legal_choices = [item_info for ix, item_info in item_df.iterrows() if check_info(item_info, t_frames, z_frames)]

            assert(legal_choices)
            item_info = random.choice(legal_choices)
            for tile_sz in tile:
                item_d = dict(item_info)
                item_d['tile_sz'] = tile_sz
                tile_infos.append(item_d)

    tile_info_df = pd.DataFrame(tile_infos).reset_index()
    print('num tile pulls:', len(tile_infos))
    print(tile_info_df.groupby('category').fn.count())

    last_stat = None
    tile_pull_info = []
    tile_puller = None

    multi_str = f'_{lr_type}_{n_frames}' if lr_type != 's' else ''
    mbar = master_bar(tile_info_df.groupby('fn'))
    for fn, tile_stats in mbar:
        if Path(fn).stem == 'high res microtubules for testing before stitching - great quality':
            continue
        for i, tile_stat in progress_bar(list(tile_stats.iterrows()), parent=mbar):
            try:
                mode = tile_stat['dsplit']
                category = tile_stat['category']
                tile_sz = tile_stat['tile_sz']
                tile_folder = ensure_folder(out / f'hr_t_{tile_sz}{multi_str}' / mode / category)
                if crap_func:
                    crap_folder = ensure_folder(out / f'lr{up}_t_{tile_sz}{multi_str}' / mode / category)
                else: crap_folder = None

                if need_cache_flush(tile_stat, last_stat):
                    if tile_puller:
                        tile_puller(None, None, None, close_me=True)
                    last_stat = tile_stat.copy()
                    tile_sz = tile_stat['tile_sz']
                    tile_puller = get_tile_puller(tile_stat, crap_func, t_frames, z_frames)
                tile_pull_info.append(tile_puller(tile_stat, tile_folder, crap_folder))
            except MemoryError as error:
                # some files are too big to read
                fn = Path(tile_stat['fn'])
                print(f'too big: {fn.stem}')

    pd.DataFrame(tile_pull_info).to_csv(out/f'tiles{multi_str}.csv', index = False)
