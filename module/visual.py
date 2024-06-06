import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def colored_depthmap(depth, d_min=None, d_max=None):
    # if len(depth.shape) == 4 and depth.shape[1] == 1:
    #     depth = depth.squeeze()
    # else:
    #     raise Exception("Need depth map to be single-channel, but got {} channels".format(depth.shape[1]))

    # Set color mode.
    cmap = plt.cm.jet
    depth = np.array(depth)
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def save_depthmap(depthmap, filename):
    im = Image.fromarray(depthmap.astype('uint8'))
    im.save(filename)