import runway
import numpy as np
import os
import os.path as osp
import platform
import argparse
import time
import sys
import subprocess
from iPERCore.services import run_imitator

# the gpu ids
gpu_ids = "0"

# the image size
image_size = 512

# the default number of source images, it will be updated if the actual number of sources <= num_source
num_source = 2

# the assets directory. This is very important, please download it from `one_drive_url` firstly.
assets_dir = "/content/iPERCore/assets"

# the output directory.
output_dir = "./results"

# the model id of this case. This is a random model name.
# model_id = "model_" + str(time.time())

# # This is a specific model name, and it will be used if you do not change it.
# model_id = "axing_1"

# symlink from the actual assets directory to this current directory
work_asserts_dir = os.path.join("./assets")
if not os.path.exists(work_asserts_dir):
    os.symlink(osp.abspath(assets_dir), osp.abspath(work_asserts_dir),
               target_is_directory=(platform.system() == "Windows"))

cfg_path = osp.join(work_asserts_dir, "configs", "deploy.toml")


def tensor2np(img_tensor):
    img = (img_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
    img = (img * 255).astype(np.uint8)
    return img
# This is a specific model name, and it will be used if you do not change it. This is the case of `trump`
model_id = "donald_trump_2"

# the source input information, here \" is escape character of double duote "


@runway.command('imitate', inputs={'source': runway.image, 'target': runway.image}, outputs={'image': runway.image})
def imitate(models, inputs):
    run_imitator()
    _, imitator = models
    imitator.personalize(np.array(inputs['source']))
    tgt_imgs = [np.array(inputs['target'])]
    res = imitator.inference(tgt_imgs, cam_strategy='target')
    img = res[0]
    img = (img + 1) / 2.0 * 255
    img = img.astype(np.uint8)
    return img


if __name__ == '__main__':
    runway.run()