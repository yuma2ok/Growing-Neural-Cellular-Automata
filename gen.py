import os
import torch
import numpy as np
import cv2

from lib.utils import mat_distance
from lib.CAModel import CAModel
from lib.utils_vis import to_rgb, make_seed

os.environ['KMP_DUPLICATE_LIB_OK']='True'

pix_size = 4
_map_shape = (200,100)
CHANNEL_N = 16
CELL_FIRE_RATE = 0.5
model_path = "models/remaster_1.pth"
device = torch.device("cuda:0")

_rows = np.arange(_map_shape[0]).repeat(_map_shape[1]).reshape([_map_shape[0],_map_shape[1]])
_cols = np.arange(_map_shape[1]).reshape([1,-1]).repeat(_map_shape[0],axis=0)
_map_pos = np.array([_rows,_cols]).transpose([1,2,0])

_map = make_seed(_map_shape, CHANNEL_N)

model = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
output = model(torch.from_numpy(_map.reshape([1,_map_shape[0],_map_shape[1],CHANNEL_N]).astype(np.float32)).to(device), 1)



running = True
while running:
    with torch.no_grad():
        output = model(output, 1)
        _map = to_rgb(output.detach()).cpu().numpy()[0]
        print(_map.dtype)

