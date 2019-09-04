from numbers import Integral
from fastai import *
from fastai.vision import *

__all__ = ['MultiImage', 'MultiImageImageList', 'MultiImageDataBunch', 'NpyRawImageList']


class MultiImage(ItemBase):
    def __init__(self, img_list):
        self.img_list = img_list

    def __repr__(self):
        return f'MultiImage: {[str(img) for img in self.img_list]}'

    @property
    def size(self):
        return [img.size for img in self.img_list]

    @property
    def data(self):
        img_data = torch.stack([img.data for img in self.img_list])
        num_img, c, h, w = img_data.shape
        data = tensor(img_data.view(num_img * c, h, w))
        return data

    def apply_tfms(self, tfms, **kwargs):
        first_time = True

        save_img_list = []
        for img in self.img_list:
            new_img = img.apply_tfms(tfms, do_resolve=first_time, **kwargs)
            first_time = False
            save_img_list.append(new_img)
        self.img_list = save_img_list
        return self

    def _repr_png_(self):
        return self._repr_image_format('png')

    def _repr_jpeg_(self):
        return self._repr_image_format('jpeg')

    def _repr_image_format(self, format_str):
        #return self.img_lists[0]._repr_image_format(format_str)
        with BytesIO() as str_buffer:
            img_data = np.concatenate(
                [image2np(img.px) for img in self.img_list], axis=1)
            plt.imsave(str_buffer, img_data, format=format_str)
            return str_buffer.getvalue()

    def show(self, **kwargs):
        self.img_list[0].show(**kwargs)


def multi_normalize(x: TensorImage, mean: FloatTensor,
                    std: FloatTensor) -> TensorImage:
    "Normalize `x` with `mean` and `std`."
    return (x - mean) / std


def multi_denormalize(x: TensorImage,
                      mean: FloatTensor,
                      std: FloatTensor,
                      do_x: bool = True) -> TensorImage:
    "Denormalize `x` with `mean` and `std`."
    return x.cpu().float() * std + mean if do_x else x.cpu()


def _multi_normalize_batch(b: Tuple[Tensor, Tensor],
                           mean: FloatTensor,
                           std: FloatTensor,
                           do_x: bool = True,
                           do_y: bool = False) -> Tuple[Tensor, Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x, y = b
    mean, std = mean.to(x.device), std.to(x.device)
    if do_x: x = multi_normalize(x, mean, std)
    if do_y and len(y.shape) == 4: y = multi_normalize(y, mean, std)
    return x, y

def multi_normalize_funcs(mean: FloatTensor,
                          std: FloatTensor,
                          do_x: bool = True,
                          do_y: bool = False) -> Tuple[Callable, Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    mean, std = tensor(mean), tensor(std)
    return (partial(_multi_normalize_batch,
                    mean=mean,
                    std=std,
                    do_x=do_x,
                    do_y=do_y),
            partial(multi_denormalize, mean=mean, std=std, do_x=do_x))


def multi_image_channel_view(x):
    n_chan = 1
    return x.transpose(0, 1).contiguous().view(n_chan, -1)


class MultiImageDataBunch(ImageDataBunch):
    def batch_stats(self,
                    funcs: Collection[Callable] = None,
                    ds_type: DatasetType = DatasetType.Train) -> Tensor:
        "Grab a batch of data and call reduction function `func` per channel"
        funcs = ifnone(funcs, [torch.mean, torch.std])
        x = self.one_batch(ds_type=ds_type, denorm=False)[0].cpu()

        return [func(multi_image_channel_view(x), 1) for func in funcs]

    def normalize(self,
                  stats: Collection[Tensor] = None,
                  do_x: bool = True,
                  do_y: bool = False) -> None:
        "Add normalize transform using `stats` (defaults to `DataBunch.batch_stats`)"
        if getattr(self, 'norm', False):
            raise Exception('Can not call normalize twice')
        if stats is None: self.stats = self.batch_stats()
        else: self.stats = stats
        self.norm, self.denorm = multi_normalize_funcs(*self.stats,
                                                       do_x=do_x,
                                                       do_y=do_y)
        self.add_tfm(self.norm)

        return self


class MultiImageList(ImageList):
    "`ItemList` suitable for computer vision."
    _bunch, _square_show, _square_show_res = MultiImageDataBunch, True, True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = 1

    def open(self, fn):
        img_data = np.load(fn)
        if img_data.dtype == np.uint8:
            img_data = img_data.astype(np.float32) / 255.0
        img_list = []
        if len(img_data.shape) == 4:
            for j in range(img_data.shape[0]):
                for i in range(img_data.shape[1]):
                    img_list.append(Image(tensor(img_data[j, i][None])))
        else:
            for i in range(img_data.shape[0]):
                img_list.append(Image(tensor(img_data[i][None])))

        self.channels = img_list[0].data.shape[0]
        return MultiImage(img_list)

    def reconstruct(self, t: Tensor):
        n, h, w = t.shape
        n //= self.channels
        one_img = t.float().view(self.channels, n * h, w)
        return Image(one_img.clamp(min=0, max=1))


class NpyRawImageList(ImageList):
    def open(self, fn):
        img_data = np.load(fn)
        return Image(tensor(img_data[None]))

    def analyze_pred(self, pred):
        return pred[0:1]

    def reconstruct(self, t):
        return Image(t.float().clamp(min=0, max=1))

    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None, **kwargs)->ItemList:
        extensions = ifnone(extensions, ['.npy'])
        return super().from_folder(path=path, extensions=extensions, **kwargs)


class MultiImageImageList(MultiImageList):
    _label_cls, _square_show, _square_show_res = ImageList, False, False

    def show_xys(self,
                 xs,
                 ys,
                 imgsize: int = 4,
                 figsize: Optional[Tuple[int, int]] = None,
                 **kwargs):
        "Show the `xs` (inputs) and `ys`(targets)  on a figure of `figsize`."
        axs = subplots(len(xs), 2, imgsize=imgsize, figsize=figsize)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x.show(ax=axs[i, 0], **kwargs)
            y.show(ax=axs[i, 1], **kwargs)
        plt.tight_layout()

    def show_xyzs(self,
                  xs,
                  ys,
                  zs,
                  imgsize: int = 4,
                  figsize: Optional[Tuple[int, int]] = None,
                  **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        title = 'Input / Prediction / Target'
        axs = subplots(len(xs),
                       3,
                       imgsize=imgsize,
                       figsize=figsize,
                       title=title,
                       weight='bold',
                       size=14)
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            x.show(ax=axs[i, 0], **kwargs)
            y.show(ax=axs[i, 2], **kwargs)
            z.show(ax=axs[i, 1], **kwargs)
