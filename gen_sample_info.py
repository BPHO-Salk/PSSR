"""
gen_sample_info.py
-------
PSSR PIPELINE - STEP 1:
Understand your datasource - Extract metadata of images in the datasource(s).

This script is the first step of PSSR pipeline. It walks through files in your
datasources and extract the metadata of each frame/slice in all desired images,
which will serve as a guide map for training data generation. It allows users to
include images from multiple datasources, and each datasource folder can have
multiple subfolders, each of which includes images of a subcategory. Images of
each subcategory need to be split into folder 'train' and 'valid' beforehand.

The datasource folder has to follow the hierarchy as below:
- datasources
  |- live
  |  |- subcategory1
  |  |  |- train
  |  |  |   |- img1
  |  |  |   |- img2
  |  |  |   |- ...
  |  |  |- valid
  |  |      |- img1
  |  |      |- img2
  |  |      |- ...
  |  |
  |  |- subcategory2
  |  |  |- train
  |  |  |   |- img1
  |  |  |   |- img2
  |  |  |   |- ...
  |  |  |- valid
  |  |      |- img1
  |  |      |- img2
  |  |      |- ...
  |  |- ...
  |
  |- fixed
     |- subcategory1
     |  |- train
     |  |   |- img1
     |  |   |- img2
     |  |   |- ...
     |  |- valid
     |      |- img1
     |      |- img2
     |      |- ...
     |
     |- subcategory2
     |  |- train
     |  |   |- img1
     |  |   |- img2
     |  |   |- ...
     |  |- valid
     |      |- img1
     |      |- img2
     |      |- ...
     |- ...

Notes:
-------
1. Except folders named 'fixed' or 'live' (which refers to live cell or fixed samples),
'train' or 'valid', all other folders and files can be changed to different names accrodingly.
2. tif/tiff and czi, the two most widely used scientific image formats are
supported. Here are some additional important information:
- czi images: 3D (XYZ) and 4D (XYZT) stacks are acceptable. For multi-channel
              images, only the first channel will be included (editable in the script).
- tif/tiff images: 2D (XY) or 3D (XYZ/XYT) stacks are acceptable. Hyperstack images are
                   recommended to be preprocessed in ImageJ/FIJI.

Parameters:
-------
- out: path, output csv file name
- sources: path, whitelist all datasource folders to include, if more than one
           need to be included. (optional)
- only: str, whitelist subfolders to include. This is useful if only a few
        subfolders among a large number of subfolders need to be included. (optional)
- skip: str, subfolders to skip. This is useful when most of the subfolders
        need to be included except a few. (optional)

Returns:
-------
- csv file: A csv file that saves useful metadata of images in the datasource.
  Each frame in each image is saved as one row separately, and each row has 24
  columns of information detailed as follows:
  - 'category': str, data category. This is useful when we want to generate a
                'combo' dataset mixed with different biological structures, for
                example, mitochondria and microtubules.
  - 'dsplit': str, 'train' if the source image is in the folder 'train', and
              'valid' if the source image is in the folder 'valid'.
  - 'multi': boolean, equals to 1 if any dimension of the source file other than
             X and Y larger than 1, which can be Z or T, in other words, if the
             source file of this slice is a z-stack or a time-lapse.
  - 'ftype': str, source file type, 'tif' or 'czi'.
  - 'uint8': boolean, 'TRUE' or 'FALSE'. It is 'TRUE' if the source file of this
             frame is 8-bit unsigned interger, otherwise 'FALSE'.
  - 'mean': float, mean value of the frame.
  - 'sd': float, standard deviation of the frame.
  - 'all_rmax': int, maximum value of the whole source image stack.
  - 'all_mi': int, 2 percentile value of the whole source image stack.
  - 'all_ma': int, 99.99 percentile value of the whole source image stack.
  - 'mi': int, 2 percentile value of the frame.
  - 'ma': int, 99.99 percentile value of the frame.
  - 'rmax': int, 255 if the source stack is 8-bit unsigned interger, otherwise
            it is the maximum value of the whole source image stack.
  - 'nc': int, number of channels of the source image stack.
  - 'nz': int, number of slices in Z dimension of the source image stack.
  - 'c': int, channel number of this frame. The first channel starts counting at 0.
  - 'nt': int, number of frames in T dimension of the source image stack.
  - 'z': int, depth number of this frame. The first slice starts counting at 0.
  - 't': int, time frame of this frame. The first frame starts counting at 0.
  - 'x': int, dimension in X.
  - 'y': int, dimension in Y.
  - 'fn': str, relative path of the source image.

Examples:
-------
Following are a couple of examples showing how to generate the metadata
from the datasource in different ways. Given the datasource folder is strctured
as below:
- datasources
  |- live
  |  |- mitotracker
  |  |  |- train
  |  |  |   |- mito_train1.tif
  |  |  |   |- mito_train2.tif
  |  |  |   |- ...
  |  |  |- valid
  |  |      |- mito_valid1.tif
  |  |      |- mito_valid2.tif
  |  |      |- ...
  |  |
  |  |- microtubules
  |     |- train
  |     |   |- microtubules_train1.tif
  |     |   |- microtubules_train2.tif
  |     |   |- ...
  |     |- valid
  |         |- microtubules_valid1.tif
  |         |- microtubules_valid2.tif
  |         |- ...
  |
  |- fixed
     |- neurons
     |  |- train
     |  |   |- neurons_train1.tif
     |  |   |- neurons_train2.tif
     |  |   |- ...
     |  |- valid
     |      |- neurons_valid1.tif
     |      |- neurons_valid2.tif
     |      |- ...
     |
     |- microtubules
        |- train
        |   |- microtubules_fixed_train1.tif
        |   |- microtubules_fixed_train2.tif
        |   |- ...
        |- valid
            |- microtubules_fixed_valid1.tif
            |- microtubules_fixed_valid2.tif
            |- ...

Example 1:
Only 'mitotracker' folder in datasource 'live' is needed. Name output
.csv file as 'live_mitotracker.csv'
python gen_sample_info.py --only mitotracker --out live_mitotracker.csv datasources/live

Example 2:
All subcategories in datasource 'live' are needed for training. Name
output file as 'live.csv'.
python gen_sample_info.py --out live.csv datasources/live

Example 3:
All subcategroies in datasource 'fixed' are needed for training except
files in 'microtubules'. Name output file as 'live.csv'
python gen_sample_info.py --skip microtubules --out live.csv datasources/fixed

Example 4:
Everything in folder 'datasources' are needed. Name output .csv file as 'all.csv'
python gen_sample_info.py --out all.csv datasources
"""
import yaml
from fastai.script import *
from fastai.vision import *
from utils import *
from pathlib import Path
from fastprogress import master_bar, progress_bar
from time import sleep
import shutil
import PIL
import czifile

PIL.Image.MAX_IMAGE_PIXELS = 99999999999999

def process_czi(item, category, mode):
#This function only takes the first channel of the czi files
#since those are the only mitotracker channels
    tif_srcs = []
    base_name = item.stem
    with czifile.CziFile(item) as czi_f:
        data = czi_f.asarray()
        axes, shape = get_czi_shape_info(czi_f)
        channels = shape['C']
        depths = shape['Z']
        times = shape['T']
        #times = min(times, 30) #ONLY USE FIRST 30 frames
        x,y = shape['X'], shape['Y']

        mid_depth = depths // 2
        depth_range = range(max(0,mid_depth-2), min(depths, mid_depth+2))
        is_multi = (times > 1) or (depths > 1)

        data = czi_f.asarray()
        all_rmax = data.max()
        all_mi, all_ma = np.percentile(data, [2,99.99])

        dtype = data.dtype
        #for channel in range(channels): #if other channels are needed, use this line
        for channel in range(0,1):
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
                                     'z': z, 't': t, 'c':channel, 'x': x, 'y': y})
    return tif_srcs

def is_live(item):
    return item.parent.parts[-3] == 'live'

def process_tif(item, category, mode):
    tif_srcs = []
    img = PIL.Image.open(item)
    n_frames = img.n_frames
    x,y = img.size
    is_multi = n_frames > 1
    #n_frames = min(n_frames, 30) #ONLY USE FIRST 30 frames

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
    print(f"**** Unknown: {item}")
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

    "generate combo dataset"
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
    tif_src_df[['category','dsplit','multi','ftype','uint8','mean','sd','all_rmax','all_mi','all_ma','mi','ma','rmax','nc','nz','nt','c','z','t','x','y','fn']].to_csv(out, header=True, index=False)
