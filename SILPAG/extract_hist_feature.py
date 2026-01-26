import scanpy as sc
import cv2 as cv
import torch
from torch import nn
import skimage
from skimage.transform import rescale
import numpy as np
from time import time
from torchvision import transforms
from einops import rearrange, reduce, repeat
from .hipt_4k import HIPT_4K



def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img


def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img


def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(img, extent, mode=mode, constant_values=pad_value)
    return img


def eval_transforms():
    """ """
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    eval_t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return eval_t


def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    x = np.pad(
            x,
            (
                (0, shape_ext[0] - x.shape[0]),
                (0, shape_ext[1] - x.shape[1]),
                (0, 0)),
            mode='edge')
    tiles_shape = np.array(x.shape[:2]) // patch_size
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> h1 w1 h w c',
    #         h=patch_size, w=patch_size)
    # x = rearrange(
    #         x, '(h1 h) (w1 w) c -> (h1 w1) h w c',
    #         h=patch_size, w=patch_size)
    tiles = []
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size  # TODO: change to patch_size[0]
        b0 = a0 + patch_size  # TODO: change to patch_size[0]
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size  # TODO: change to patch_size[1]
            b1 = a1 + patch_size  # TODO: change to patch_size[1]
            tiles.append(x[a0:b0, a1:b1])

    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    return tiles, shapes


def get_embeddings_sub(model, x):
    x = x.astype(np.float32) / 255.0
    x = eval_transforms()(x)
    x_cls, x_sub = model.forward_all256(x[None])
    x_cls = x_cls.cpu().detach().numpy()
    x_sub = x_sub.cpu().detach().numpy()
    x_cls = x_cls[0].transpose(1, 2, 0)
    x_sub = x_sub[0].transpose(1, 2, 3, 4, 0)
    return x_cls, x_sub


def get_embeddings_cls(model, x):
    x = torch.tensor(x.transpose(2, 0, 1))
    with torch.no_grad():
        __, x_sub4k = model.forward_all4k(x[None])
    x_sub4k = x_sub4k.cpu().detach().numpy()
    x_sub4k = x_sub4k[0].transpose(1, 2, 0)
    return x_sub4k


def get_embeddings(img, pretrained=True, device='cuda'):
    '''
    Extract embeddings from histology tiles
    Args:
        tiles: Histology image tiles.
            Shape: (N, H, W, C).
            `H` and `W` are both divisible by 256.
            Channels `C` include R, G, B, foreground mask.
    Returns:
        emb_cls: Embeddings of (256 x 256)-sized patches
            Shape: (H/256, W/256, 384)
        emb_sub: Embeddings of (16 x 16)-sized patches
            Shape: (H/16, W/16, 384)
    '''
    print('Extracting embeddings...')
    t0 = time()

    tile_size = 4096
    tiles, shapes = patchify(img, patch_size=tile_size)

    model256_path, model4k_path = None, None
    if pretrained:
        model256_path = '/data/luy/SILPAG/HIPT_pretrain/vit256_small_dino.pth'
        model4k_path = '/data/luy/SILPAG/HIPT_pretrain/vit4k_xs_dino.pth'
    model = HIPT_4K(
            model256_path=model256_path,
            model4k_path=model4k_path,
            device256=device, device4k=device)
    model.eval()
    patch_size = (256, 256)
    subpatch_size = (16, 16)
    n_subpatches = tuple(
            a // b for a, b in zip(patch_size, subpatch_size))

    emb_sub = []
    emb_mid = []
    for i in range(len(tiles)):
        if i % 10 == 0:
            print('tile', i, '/', len(tiles))
        x_mid, x_sub = get_embeddings_sub(model, tiles[i])
        emb_mid.append(x_mid)
        emb_sub.append(x_sub)
    del tiles
    torch.cuda.empty_cache()
    emb_mid = rearrange(
            emb_mid, '(h1 w1) h2 w2 k -> (h1 h2) (w1 w2) k',
            h1=shapes['tiles'][0], w1=shapes['tiles'][1])

    emb_cls = get_embeddings_cls(model, emb_mid)
    del emb_mid, model
    torch.cuda.empty_cache()

    shape_orig = np.array(shapes['original']) // subpatch_size

    chans_sub = []
    for i in range(emb_sub[0].shape[-1]):
        chan = rearrange(
                np.array([e[..., i] for e in emb_sub]),
                '(h1 w1) h2 w2 h3 w3 -> (h1 h2 h3) (w1 w2 w3)',
                h1=shapes['tiles'][0], w1=shapes['tiles'][1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_sub.append(chan)
    del emb_sub

    chans_cls = []
    for i in range(emb_cls[0].shape[-1]):
        chan = repeat(
                np.array([e[..., i] for e in emb_cls]),
                'h12 w12 -> (h12 h3) (w12 w3)',
                h3=n_subpatches[0], w3=n_subpatches[1])
        chan = chan[:shape_orig[0], :shape_orig[1]]
        chans_cls.append(chan)
    del emb_cls

    print(int(time() - t0), 'sec')

    return chans_cls, chans_sub


def get_embeddings_shift(
        img, margin=256, stride=64,
        pretrained=True, device='cuda'):
    # margin: margin for shifting. Divisble by 256
    # stride: stride for shifting. Divides `margin`.
    factor = 16  # scaling factor between cls and sub. Fixed
    shape_emb = np.array(img.shape[:2]) // factor
    chans_cls = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(192)]
    chans_sub = [
            np.zeros(shape_emb, dtype=np.float32)
            for __ in range(384)]
    start_list = list(range(0, margin, stride))
    n_reps = 0
    for start0 in start_list:
        for start1 in start_list:
            print(f'shift {start0}/{margin}, {start1}/{margin}')
            t0 = time()
            stop0, stop1 = -margin+start0, -margin+start1
            im = img[start0:stop0, start1:stop1]
            cls, sub = get_embeddings(
                    im, pretrained=pretrained, device=device)
            del im
            sta0, sta1 = start0 // factor, start1 // factor
            sto0, sto1 = stop0 // factor, stop1 // factor
            for i in range(len(chans_cls)):
                chans_cls[i][sta0:sto0, sta1:sto1] += cls[i]
            del cls
            for i in range(len(chans_sub)):
                chans_sub[i][sta0:sto0, sta1:sto1] += sub[i]
            del sub
            n_reps += 1
            print(int(time() - t0), 'sec')

    mar = margin // factor
    for chan in chans_cls:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0
    for chan in chans_sub:
        chan /= n_reps
        chan[-mar:] = 0.0
        chan[:, -mar:] = 0.0

    return chans_cls, chans_sub


def impute_missing(x, mask, radius=3, method='ns'):

    method_dict = {
            'telea': cv.INPAINT_TELEA,
            'ns': cv.INPAINT_NS}
    method = method_dict[method]

    x = x.copy()
    if x.dtype == np.float64:
        x = x.astype(np.float32)

    x[mask] = 0
    mask = mask.astype(np.uint8)

    expand_dim = np.ndim(x) == 2
    if expand_dim:
        x = x[..., np.newaxis]
    channels = [x[..., i] for i in range(x.shape[-1])]
    y = [cv.inpaint(c, mask, radius, method) for c in channels]
    y = np.stack(y, -1)
    if expand_dim:
        y = y[..., 0]

    return y


def smoothen(
        x, size, kernel='gaussian', backend='cv', mode='mean',
        impute_missing_values=True, device='cuda'):

    if x.ndim == 3:
        expand_dim = False
    elif x.ndim == 2:
        expand_dim = True
        x = x[..., np.newaxis]
    else:
        raise ValueError('ndim must be 2 or 3')

    mask = np.isfinite(x).all(-1)
    if (~mask).any() and impute_missing_values:
        x = impute_missing(x, ~mask)

    if kernel == 'gaussian':
        sigma = size / 4  # approximate std of uniform filter 1/sqrt(12)
        truncate = 4.0
        winsize = np.ceil(sigma * truncate).astype(int) * 2 + 1
        if backend == 'cv':
            print(f'gaussian filter: winsize={winsize}, sigma={sigma}')
            y = cv.GaussianBlur(
                    x, (winsize, winsize), sigmaX=sigma, sigmaY=sigma,
                    borderType=cv.BORDER_REFLECT)
        elif backend == 'skimage':
            y = skimage.filters.gaussian(
                    x, sigma=sigma, truncate=truncate,
                    preserve_range=True, channel_axis=-1)
        else:
            raise ValueError('backend must be cv or skimage')
    elif kernel == 'uniform':
        if backend == 'cv':
            kernel = np.ones((size, size), np.float32) / size**2
            y = cv.filter2D(
                    x, ddepth=-1, kernel=kernel,
                    borderType=cv.BORDER_REFLECT)
            if y.ndim == 2:
                y = y[..., np.newaxis]
        elif backend == 'torch':
            assert isinstance(size, int)
            padding = size // 2
            size = size + 1

            pool_dict = {
                    'mean': nn.AvgPool2d(
                        kernel_size=size, stride=1, padding=0),
                    'max': nn.MaxPool2d(
                        kernel_size=size, stride=1, padding=0)}
            pool = pool_dict[mode]

            mod = nn.Sequential(
                    nn.ReflectionPad2d(padding),
                    pool)
            y = mod(torch.tensor(x, device=device).permute(2, 0, 1))
            y = y.permute(1, 2, 0)
            y = y.cpu().detach().numpy()
        else:
            raise ValueError('backend must be cv or torch')
    else:
        raise ValueError('kernel must be gaussian or uniform')

    if not mask.all():
        y[~mask] = np.nan

    if expand_dim and y.ndim == 3:
        y = y[..., 0]

    return y


def smoothen_embeddings(
        embs, size, kernel,
        method='cv', groups=None, device='cuda'):
    if groups is None:
        groups = embs.keys()
    out = {}
    for grp, em in embs.items():
        if grp in groups:
            if isinstance(em, list):
                smoothened = [
                        smoothen(
                            c[..., np.newaxis], size=size,
                            kernel=kernel, backend=method,
                            device=device)[..., 0]
                        for c in em]
            else:
                smoothened = smoothen(em, size, method, device=device)
        else:
            smoothened = em
        out[grp] = smoothened
    return out


def get_disk_mask(radius, boundary_width=None):
    radius_ceil = np.ceil(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def extract_hist_feature(img, locs, radius, pixel_size_raw, random_state=0, device='cuda'):
    """
    img: histology image, np.array.
    locs: coordinates, np.array.
    radius: radius of spot, spot_diameter_fullres * 0.5.
    pixel_size_raw: μm / pixel, if visium: 8000 / 2000 * tissue_hires_scalef 
    """
    # pixel_size_raw = 0.4 # 8000 / 2000 * adata.uns['spatial']['OldST']['scalefactors']['tissue_hires_scalef']
    pixel_size = 0.5 # fixed
    scale = pixel_size_raw / pixel_size
    img = img.astype(np.float32)
    img = rescale_image(img, scale)
    img = img.astype(np.uint8)
    img = adjust_margins(img, pad=256, pad_value=255)
    print('Shape of processed image: ', img.shape)

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    emb_cls, emb_sub = get_embeddings(img, pretrained=True, device=device)

    embs = dict(cls=emb_cls, sub=emb_sub)
    embs['rgb'] = np.stack([reduce(img[..., i].astype(np.float16) / 255.0, 
                                '(h1 h) (w1 w) -> h1 w1', 'mean', h=16, w=16).astype(np.float32) for i in range(3)])
    
    embs = smoothen_embeddings(embs, size=16, kernel='uniform', groups=['cls'], method='cv', device=device)
    embs = smoothen_embeddings(embs, size=4, kernel='uniform', groups=['sub'], method='cv', device=device)
    embs = np.concatenate([embs['cls'], embs['sub'], embs['rgb']])
    embs = embs.transpose(1, 2, 0)

    locs = locs * scale
    locs = locs.round().astype(int)
    locs = np.stack([locs[:, 1], locs[:, 0]], -1)
    rescale_factor = np.array(img.shape[:2]) // embs.shape[:2]
    locs = locs.astype(float)
    locs /= rescale_factor
    locs = locs.round().astype(int)

    radius = radius * scale
    radius = np.round(radius).astype(int)
    radius = radius / 16

    mask = get_disk_mask(radius)
    x = get_patches_flat(embs, locs, mask)
    fused_x = np.mean(x, axis=1)

    return fused_x