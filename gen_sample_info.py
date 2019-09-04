"build combo dataset"
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

def process_czi(item, category, mode):
    tif_srcs = []
    base_name = item.stem
    with czifile.CziFile(item) as czi_f:
        data = czi_f.asarray()
        axes, shape = get_czi_shape_info(czi_f)
        channels = shape['C']
        depths = shape['Z']
        times = shape['T']

        x,y = shape['X'], shape['Y']

        mid_depth = depths // 2
        depth_range = range(max(0,mid_depth-2), min(depths, mid_depth+2))
        is_multi = (times > 1) or (depths > 1)

        data = czi_f.asarray()
        all_rmax = data.max()
        all_mi, all_ma = np.percentile(data, [2,99.99])

        dtype = data.dtype
        for channel in range(channels):
            for z in depth_range:
                for t in range(times):
                    idx = build_index(
                        axes, {
                            'T': t,
                            'C': channel,
                            'Z': z,
                            'X': slice(0, x),
                            'Y': slice(0, y)
                        })
                    img = data[idx]
                    mi, ma = np.percentile(img, [2,99.99])
                    if dtype == np.uint8: rmax = 255.
                    else: rmax = img.max()
                    tif_srcs.append({'fn': item, 'ftype': 'czi', 'multi':int(is_multi), 'category': category, 'dsplit': mode,
                                     'uint8': dtype == np.uint8, 'mi': mi, 'ma': ma, 'rmax': rmax,
                                     'all_rmax': all_rmax, 'all_mi': all_mi, 'all_ma': all_ma,
                                     'mean': img.mean(), 'sd': img.std(),
                                     'nc': channels, 'nz': depths, 'nt': times,
                                     'z': mid_depth, 't': t, 'c':channel, 'x': x, 'y': y})
    return tif_srcs

def is_live(item):
    return item.parent.parts[-3] == 'live'

def process_tif(item, category, mode):
    tif_srcs = []
    img = PIL.Image.open(item)
    n_frames = img.n_frames
    x,y = img.size
    is_multi = n_frames > 1

    data = []
    for n in range(n_frames):
        img.seek(n)
        img.load()
        img_data = np.array(img)
        data.append(img_data)

    data = np.stack(data)
    all_rmax = data.max()
    all_mi, all_ma = np.percentile(data, [2,99.99])

    for n in range(n_frames):
        img_data = data[n]
        dtype = img_data.dtype
        mi, ma = np.percentile(img_data, [2,99.99])
        if dtype == np.uint8: rmax = 255.
        else: rmax = img_data.max()
        if is_live(item):
            t, z = n, 0
            nt, nz = n_frames, 1
        else:
            t, z = 0, n
            nt, nz = 1, n_frames

        tif_srcs.append({'fn': item, 'ftype': 'tif', 'multi':int(is_multi), 'category': category, 'dsplit': mode,
                         'uint8': dtype==np.uint8, 'mi': mi, 'ma': ma, 'rmax': rmax,
                         'all_rmax': all_rmax, 'all_mi': all_mi, 'all_ma': all_ma,
                         'mean': img_data.mean(), 'sd': img_data.std(),
                         'nc': 1, 'nz': nz, 'nt': nt,
                         'z': z, 't': t, 'c':0, 'x': x, 'y': y})
    return tif_srcs

def process_unk(item, category, mode):
    print(f"**** WTF: {item}")
    return []

def process_item(item, category, mode):
    try:
        if mode == 'test': return []
        else:
            item_map = {
                '.tif': process_tif,
                '.tiff': process_tif,
                '.czi': process_czi,
            }
            map_f = item_map.get(item.suffix, process_unk)
            return map_f(item, category, mode)
    except Exception as ex:
        print(f'err procesing: {item}')
        print(ex)
        return []

def build_tifs(src, mbar=None):
    tif_srcs = []
    for mode in ['train', 'valid', 'test']:
        live = src.parent.parts[-1] == 'live'
        src_dir = src / mode
        category = src.stem
        items = list(src_dir.iterdir()) if src_dir.exists() else []
        if items:
            for p in progress_bar(items, parent=mbar):
                mbar.child.comment = mode
                tif_srcs += process_item(p, category=category, mode=mode)
    return tif_srcs

@call_parse
def main(out: Param("tif output name", Path, required=True),
         sources: Param('src folders', Path, nargs='...', opt=False) = None,
         only: Param('whitelist subfolders to include', str, nargs='+') = None,
         skip: Param("subfolders to skip", str, nargs='+') = None):

    "generate comobo dataset"
    if skip and only:
        print('you can skip subfolder or whitelist them but not both')
        return 1

    src_dirs = []
    for src in sources:
        sub_fldrs = subfolders(src)
        if skip:  src_dirs += [fldr for fldr in sub_fldrs if fldr.stem not in skip]
        elif only: src_dirs += [fldr for fldr in sub_fldrs if fldr.stem in only]
        else: src_dirs += sub_fldrs

    mbar = master_bar(src_dirs)
    tif_srcs = []
    for src in mbar:
        mbar.write(f'process {src.stem}')
        tif_srcs += build_tifs(src, mbar=mbar)

    tif_src_df = pd.DataFrame(tif_srcs)
    tif_src_df[['category','dsplit','multi','ftype','uint8','mean','sd','all_rmax','all_ma','all_ma','mi','ma','rmax','nc','nz','nt','c','z','t','x','y','fn']].to_csv(out, header=True, index=False)
