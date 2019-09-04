"utility classes for creating tiles form imagelist images"
from fastai import *
from fastai.vision import *
from fastprogress import progress_bar

__all__ = ['TileImageTileImageList', 'TileImageList']

def get_image_list_shapes(imlist):
    shapes = [(img_i, fn, imlist.open(fn).shape) for img_i, fn in progress_bar(list(enumerate(imlist.items)))]
    return shapes

def make_tile_info(shapes, tile_sz, num_tiles):
    return [make_tile_xy(i_fn_shape, tile_sz, num_tiles) for i_fn_shape in shapes]
   
def make_tile_xy(i_fn_shape, size, num_tiles):
    img_i, fn, shape = i_fn_shape 
    xy_data = []
    for i in range(num_tiles):
        xs,ys = make_rand_tile_xy(shape, size)
        xy_data.append((img_i,fn,xs,ys))
    return xy_data

def make_rand_tile_xy(shape, size):
    x = random.choice(range(shape[1]-size)) if shape[1] > size else 0
    y = random.choice(range(shape[2]-size)) if shape[2] > size else 0
    xs = slice(x,min(x+size, shape[1]))
    ys = slice(y,min(y+size, shape[2]))
    return xs,ys



class TileImageList(ImageList):
    _img_list_cls = ImageList
    def __init__(self, items, *args, tile_infos=None, tile_sz=128, num_tiles=5, tile_scale=1, 
                 crap_func=None,
                 img_list_cls=None, **kwargs):
        self.crap_func = crap_func 
        if img_list_cls is None: img_list_cls = self._img_list_cls
        self.img_list_cls = img_list_cls
        self.img_list = img_list_cls(items, *args, **kwargs)
        #  if tile_scale is None: tile_scale = 1
        if tile_infos is None:
            shapes = get_image_list_shapes(self.img_list)
            tile_infos = make_tile_info(shapes, tile_sz, num_tiles)
            
        self.tile_infos = tile_infos
        self.tile_scale = tile_scale 

        tile_items = []
        for tile_infos in self.tile_infos:
            for tile_info in tile_infos:
                tile_items.append(tile_info)
        super().__init__(tile_items)

    def open(self, item):
        def scale_xy(xs, ys, scale):
            return (slice(xs.start*scale, xs.stop*scale), 
                    slice(ys.start*scale, ys.stop*scale))
        img_i, fn, xs, ys = item
        img = self.img_list.get(img_i)
        xs,ys = scale_xy(xs, ys, scale=self.tile_scale)
        img_data = img.data[:,xs, ys]
        if self.crap_func: img_data = self.crap_func(img_data.numpy())
        return Image(tensor(img_data))

    def _get_by_folder(self, name):
        return [i for i in range_of(self) if self.items[i][1][0].parts[-2] == name]

    def _label_from_list(self, labels, label_cls=None, from_item_lists=False, **kwargs)->'LabelList':
        "Label `self.items` with `labels`."
        if not from_item_lists: 
            raise Exception("Your data isn't split, if you don't want a validation set, please use `split_none`.")
        fns = []
        last_fn = None
        for j, item in enumerate(labels):
            img_i = item[0]
            fn = item[1]
            if fn != last_fn:
                fns.append(fn)
                last_fn = fn

        labels = array(fns, dtype=object)
        label_cls = self.get_label_cls(labels, label_cls=label_cls, **kwargs)
        y = label_cls(labels, 
                      img_list_cls=self.img_list_cls, 
                      tile_infos=self.tile_infos, 
                      path=self.path, **kwargs)
        res = self._label_list(x=self, y=y)
        return res


class TileImageTileImageList(TileImageList):
    "`ItemList` suitable for `Image` to `Image` tasks."
    _label_cls,_square_show,_square_show_res = TileImageList,False,False

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys`(targets)  on a figure of `figsize`."
        axs = subplots(len(xs), 2, imgsize=imgsize, figsize=figsize)
        for i, (x,y) in enumerate(zip(xs,ys)):
            x.show(ax=axs[i,0], **kwargs)
            y.show(ax=axs[i,1], **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        title = 'Input / Prediction / Target'
        axs = subplots(len(xs), 3, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.show(ax=axs[i,0], **kwargs)
            y.show(ax=axs[i,2], **kwargs)
            z.show(ax=axs[i,1], **kwargs)
