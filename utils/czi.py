"utility functions for working with czi files"
from fastai import *
from fastai.vision import *
import czifile

__all__ = ['get_czi_shape_info', 'build_index','CziImageList']


def get_czi_shape_info(czi_file):
    """get_czi_shape_info

    :param czi_file:
    """
    shape = czi_file.shape
    axes_dict = {axis: idx for idx, axis in enumerate(czi_file.axes)}
    shape_dict = {axis: shape[axes_dict[axis]] for axis in czi_file.axes}
    return axes_dict, shape_dict


def build_index(axes, ix_select):
    """build_index

    :param axes:
    :param ix_select:
    """
    idx = [ix_select.get(ax, 0) for ax in axes]
    return tuple(idx)


def is_movie(czi_file):
    axes, shape = get_czi_shape_info(czi_file)
    times = axes.get('T', 1)
    return times > 1


def has_depth(czi_file):
    axes, shape = get_czi_shape_info(czi_file)
    has_depth = axes.get('Z', 1)
    return depth > 1


def is_czi(fn):
    return fn.suffix == '.czi'

def is_tif(fn):
    return fn.suffix == '.tif'

class CziImageList(ImageList):
    def __init__(self, *args, **kwargs):
        if args:
            czi_files = args[0]
            print('type:', type(czi_files[0]))
            items = []
            for fn in czi_files:
                if is_czi(fn):
                    items += self.build_czi_items(fn)
                elif is_tif(fn):
                    items += self.build_tif_items(fn)
                else:
                    print(f'skipping {fn}')
            super().__init__(items, *args[1:], **kwargs)
        else:
            super().__init__(*args, **kwargs)
    
    def build_tif_items(self, tif_fn):
       with PIL.Image.open(tif_fn) as img:
           n_frames = img.n_frames
       items = []
       for t in range(n_frames):
           items.append((tif_fn, t))
       return items

    def build_czi_items(self, czi_fn):
        items = []
        with czifile.CziFile(czi_fn) as czi_f:
            axes, shape = get_czi_shape_info(czi_f)
            channels = shape['C']
            depths = shape['Z']
            times = shape['T']
            x,y = shape['X'], shape['Y']
            for channel in range(channels):
                for depth in range(depths):
                    for t in range(times):
                        items.append((czi_fn, axes, shape, channel, depth, t, x, y))
        return items
    
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection=None, **kwargs):
        if extensions is None: extensions = ['.czi', '*.tif']
        return super().from_folder(path=path, extensions=extensions, **kwargs)
    
    def open(self, item):
        fn = item[0] 
        if is_czi(fn):
            in_fn, axes, shape, channel, depth, t, x, y = item
            idx = build_index( axes, 
                              {'T': t, 'C':channel, 'Z': depth, 
                               'X':slice(0,x), 'Y':slice(0,y)})
            with czifile.CziFile(fn) as czi_f:
                data = czi_f.asarray()
                img_data = data[idx].astype(np.float32).copy()
                img_max = img_data.max()
                if img_max != 0: img_data /= img_max * 0.9
        else:
            in_fn, n_frame = item
            img = PIL.Image.open(fn)
            img.seek(n_frame)
            img.load()

            img_data = np.array(img).copy().astype(np.float32)
            img_max = img_data.max()
            if img_max != 0: img_data /= img_max * 0.9

        return Image(tensor(img_data[None]))
