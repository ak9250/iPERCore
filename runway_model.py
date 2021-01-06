import runway
import numpy as np
import shutil
import os
import os.path as osp
import platform
import argparse
import time
import sys
from iPERCore.services.run_imitator import run_imitator
import subprocess
from subprocess import call
from PIL import Image
import shutil



work_asserts_dir = os.path.join("./assets")
if not os.path.exists(work_asserts_dir):
    os.symlink(osp.abspath(assets_dir), osp.abspath(work_asserts_dir),
               target_is_directory=(platform.system() == "Windows"))

cfg_path = osp.join(work_asserts_dir, "configs", "deploy.toml")


def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
    

@runway.command('imitate', inputs={'source': runway.image, 'target': runway.image}, outputs={'image': runway.image})
def imitate(models, inputs):
  os.makedirs('images', exist_ok=True)
  inputs['source'].save('images/temp1.jpg')
  inputs['target'].save('images/temp2.jpg')

  paths1 = os.path.join('images','temp1.jpg')
  paths2 = os.path.join('images','temp2.jpg')

  src_path = "./images/temp1.jpg"

  ref_path = "./images/temp2.jpg"

  counter = 0
  trump = 0
  akun = 0
  stage_1_command = ("python -m iPERCore.services.run_imitator"
            + " --gpu_ids 0"
            + " --num_source 2"
            + " --image_size  256"
            + " --output_dir ./results"
            + " --model_id donald_trump_"
            + str(counter)
            + " --cfg_path "
            + cfg_path
            + " --src_path path?=./images/temp1.jpg,name?=donald_trump_"
            + str(trump)
            + " --ref_path path?=./images/temp2.jpg,name?=akun_"+str(akun)+",pose_fc?=300"
  )      
  run_cmd(stage_1_command)
  path = "./results/primitives/donald_trump_"+str(counter)+"/synthesis/imitations/donald_trump_"+str(trump)+"-akun_"+str(akun)+"/pred_00000000.png"
  img = Image.open(open(path, 'rb'))
  return img

if __name__ == '__main__':
    runway.run()