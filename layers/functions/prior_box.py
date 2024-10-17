import math
import numpy as np
from itertools import product

import torch
from typing import Tuple


class PriorBox:
    def __init__(self, cfg: dict, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.clip = cfg['clip']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.feature_maps = [[
            math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)] for step in self.steps
        ]

    def generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes based on configuration and image size"""
        anchors = []
        for k, (map_height, map_width) in enumerate(self.feature_maps):
            step = self.steps[k]
            for i, j in product(range(map_height), range(map_width)):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [(j + offset) * step / self.image_size[1] for offset in [0, 0.25, 0.5, 0.75]]
                        dense_cy = [(j + offset) * step / self.image_size[0] for offset in [0, 0.25, 0.5, 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * step / self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y * step / self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * step / self.image_size[1]
                        cy = (i + 0.5) * step / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
