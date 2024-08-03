import numpy as np
import os

imgdir = "/home/tione/notebook/model/data/monai/img/"
segdir = "/home/tione/notebook/model/data/monai/seg/"
images = sorted([(imgdir + s) for s in os.listdir(imgdir)])
segs = sorted([(segdir + s) for s in os.listdir(segdir)])
files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

for f in files:
    img = np.load(f["img"])
    seg = np.load(f["seg"])
    img = img[:, :, 48: 464, 48: 464]
    seg = seg[:, :, 48: 464, 48: 464]
    np.save(f["img"], img)
    np.save(f["seg"], seg)
    