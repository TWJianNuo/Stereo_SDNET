import torch
import numpy as np
from PIL import Image
import os
def visualize_img(inputs, prefix):
    sv_path = '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/resized_cityscape'
    for k in list(inputs):
        if "color" in k and 0 in k:
            image_tuple = inputs[k]
            image_tuple = image_tuple.permute(1,2,0)
            image_tuple = image_tuple.numpy()
            img = Image.fromarray((image_tuple * 255).astype(np.uint8))
            img.save(os.path.join(sv_path, prefix + '_' + str(k[2]) + ".png"))