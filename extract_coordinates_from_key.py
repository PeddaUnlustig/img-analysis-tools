import pandas as pd
import re

def tissue(row):
    fat = row["isfat"]
    epi = row["isepi"]
    tdlu = row["istdlu"]

    if tdlu:
        return "tdlu"
    elif fat and epi:
        return "both"
    elif fat:
        return "fat"
    elif epi:
        return "epi"
    else:
        return "stroma"


def parse_coord_old(key):
    m = re.search("_(\d+)x(\d+)_(\d+)_(\d+)$", key)

    if m is None:
        print("failed:", key)
        return None

    row, col, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))

    oy = row * 1024 + y
    ox = col * 1024 + x

    return (ox, oy)


def parse_coord_new(key):
    m = re.search("_(\d+)_(\d+)$", key)

    if m is None:
        print("failed:", key)
        return None

    x, y = int(m.group(1)), int(m.group(2))

    return (x, y)


def make_xykey(row, DIST=10):
    x = int(round(row["x"] / DIST))
    y = int(round(row["y"] / DIST))
    return f'{row["code"]}_{x}x{y}'

