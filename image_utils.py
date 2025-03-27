import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from skimage.transform import resize

# NOTE: This script was created by Indra Heckenbach, but is used in some of the scripts.

def pearson_overlap(pixels1, pixels2):
    pc = scipy.stats.pearsonr(pixels1, pixels2)
    return pc[0]


def view_img(img, title):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_axis_off()

    ax.imshow(img, cmap="gray")
    ax.set_title(title)

    return ax


def view_img_grid(imgs, titles=None, cols=2, conv8b=None):
    rows = math.ceil(len(imgs) / cols)

    fig, axes = plt.subplots(rows, cols)

    for idx in range(rows * cols):
        row = math.floor(idx / cols)
        col = idx % cols
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        ax.set_axis_off()

        if idx < len(imgs):
            img = imgs[idx]
            if conv8b and img.dtype != "uint8":
                img = convert_8b(img)

            ax.imshow(img, cmap="gray")  # , vmin=0, vmax=255
            if titles is not None:
                ax.set_title(titles[idx])

    return fig, axes


def extract_contour(img, block_size=3, max_value=255, ksize=(5, 5), threshold=None):
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise Exception("contour with 0 size")

    # already gray
    if len(img.shape) == 3:
        gray = img[:, :, 0]
    else:
        gray = img
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(src=gray, ksize=ksize, sigmaX=0)

    if threshold is None:
        binary = cv2.adaptiveThreshold(
            blur,
            max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            0,
        )
    else:
        (t, binary) = cv2.threshold(
            src=blur, thresh=threshold, maxval=max_value, type=cv2.THRESH_BINARY
        )

    (cnts, hierarchy) = cv2.findContours(
        image=binary,
        mode=cv2.RETR_EXTERNAL,
        # mode=cv2.RETR_FLOODFILL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(cnts) == 0:
        return None

    cnt = cnts[0]

    if len(cnts) > 1:
        # take largest
        max_area = 0
        for tst_cnt in cnts:
            area = cv2.contourArea(tst_cnt)
            if area > max_area:
                max_area = area
                cnt = tst_cnt

    return cnt


def extract_shape_metrics(cnt):
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    x, y, w, h = cv2.boundingRect(cnt)

    return x, y, w, h, perimeter, area, hull_perimeter, hull_area


def convert_8b(img):
    if img.dtype == "uint16":
        nimg = np.empty(img.shape, dtype=np.uint8)
        nimg[:, :] = img[:, :] / 256  # 65536
    else:
        nimg = img

    return nimg


def make_color_channel(img):
    if len(img.shape) == 2:
        return np.stack((img,) * 3, axis=-1)

    elif img.shape[2] == 1:
        return np.stack((img[:, :, 0],) * 3, axis=-1)

    else:
        return img


def cutout_region(img, region, margin, max_w, max_h):
    omnx, omny = np.min(region, axis=0)
    omxx, omxy = np.max(region, axis=0)
    w, h = omxx - omnx, omxy - omny

    # size is true size of contained item
    size = w * h

    if w > max_w or h > max_h:
        return None

    margin_x = margin
    if w + margin_x * 2 > max_w:
        margin_x = (max_w - w) // 2
    mnx, mxx = omnx - margin_x, omxx + margin_x

    margin_y = margin
    if h + margin_y * 2 > max_h:
        margin_y = (max_h - h) // 2
    mny, mxy = omny - margin_y, omxy + margin_y
    # if spans edge, skip
    if mnx < 0 or mny < 0 or mxx >= img.shape[1] or mxy >= img.shape[0]:
        return None

    w, h = mxx - mnx, mxy - mny

    return mnx, mny, mxx, mxy, size


def cutout_contents(img):
    mid_y = img.shape[0] // 2
    mid_x = img.shape[1] // 2

    for xpos in range(0, mid_x):
        if img[mid_y, xpos].any() > 0:
            xstart = xpos
            break

    for ypos in range(0, mid_y):
        if img[ypos, mid_x].any() > 0:
            ystart = ypos
            break

    return img[ystart:-ystart, xstart:-xstart, :]


def fit_to_area(rimg, shp):
    nrimg = np.zeros(shape=shp, dtype=rimg.dtype)

    if rimg.shape[0] < nrimg.shape[0]:
        h, w = rimg.shape[0:2]
        y1 = (nrimg.shape[0] - h) // 2
        x1 = (nrimg.shape[1] - w) // 2
        if len(shp) == 3:
            for ch in range(rimg.shape[2]):
                nrimg[y1 : y1 + h, x1 : x1 + w, ch] = rimg[:, :, ch]
        else:
            nrimg[y1 : y1 + h, x1 : x1 + w] = rimg[:, :]

    else:
        h, w = nrimg.shape[0:2]
        y1 = (rimg.shape[0] - h) // 2
        x1 = (rimg.shape[1] - w) // 2
        # print(w, h, x1, y1, rimg.shape)
        if len(shp) == 3:
            if len(rimg.shape) == 3:
                for ch in range(rimg.shape[2]):
                    nrimg[:, :, ch] = rimg[y1 : y1 + h, x1 : x1 + w, ch]
            else:
                nrimg[:, :, 0] = rimg[y1 : y1 + h, x1 : x1 + w]
        else:
            nrimg[:, :] = rimg[y1 : y1 + h, x1 : x1 + w]

    return nrimg


def slidescanner_resize(img):
    WIDTH, HEIGHT = 1024, 1024
    dtype = img.dtype

    if img.shape[0] > HEIGHT or img.shape[1] > WIDTH:
        if img.shape[0] > img.shape[1]:
            h = HEIGHT
            w = int(WIDTH * img.shape[1] / img.shape[0])
        else:
            w = WIDTH
            h = int(HEIGHT * img.shape[0] / img.shape[1])

        print("Converting to: ", w, h)
        new_shape = (h, w, img.shape[2])
        img = resize(
            img, new_shape, anti_aliasing=True, preserve_range=True, cval=0
        ).astype(dtype)

        sqr = np.zeros(shape=(HEIGHT, WIDTH, img.shape[2]), dtype=dtype)
        sqr[0:h, 0:w, :] = img[:, :, :]

    else:  # smaller
        h, w, ch = img.shape

        # in case > 3 channels.  need <=3 for NN.  editing that if really need >3
        if ch > 3:
            ch = 3

        sqr = np.zeros(shape=(HEIGHT, WIDTH, ch), dtype=dtype)
        # align left/top for tiling edges
        sqr[0:h, 0:w, :] = img[:, :, 0:ch]

    return sqr


def extend_shape(img, tgt_img):
    if tgt_img.shape[0] != img.shape[0] or tgt_img.shape[1] != img.shape[1]:
        nimg = np.zeros(shape=tgt_img.shape, dtype=img.dtype)
        h, w, _ = img.shape
        nimg[0:h, 0:w, :] = img

        return nimg
    else:
        return img


def contour_to_path(cnt):
    return [tuple(co[0]) for co in cnt]


def image_resize(img, resize=512):
    ny, nx = img.shape[:2]
    if np.array(img.shape).max() > resize:
        if ny > nx:
            nx = int(nx / ny * resize)
            ny = resize
        else:
            ny = int(ny / nx * resize)
            nx = resize
        shape = (nx, ny)
        img = cv2.resize(img, shape)
    img = img.astype(np.uint8)
    return img
