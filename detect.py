import os
import cv2
import argparse
import numpy as np
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

from config import cfg
from models.faceboxes import FaceBoxes
from utils.prior_box import PriorBox
from utils.box_utils import decode, nms


class FaceBoxesInference:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.05,
        pre_nms_top_k: int = 300,
        nms_threshold: float = 0.2,
        post_nms_top_k: int = 750
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(weights_path)
        self.conf_threshold = conf_threshold
        self.pre_nms_top_k = pre_nms_top_k
        self.nms_threshold = nms_threshold
        self.post_nms_top_k = post_nms_top_k

    def load_model(self, weights_path: str):
        model = FaceBoxes(num_classes=2)
        model.eval()
        model.to(self.device)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        print('Finished loading model!')
        return model

    def detect_faces(self, image_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            loc, conf = self.model(image_tensor)  # Forward pass
            conf = F.softmax(conf, dim=-1)
        return loc, conf

    def process_image(self, image_arr: np.ndarray) -> np.ndarray:
        im_height, im_width = image_arr.shape[1:3]
        scale = torch.tensor([im_width, im_height, im_width, im_height], device=self.device).float()

        image_tensor = torch.from_numpy(image_arr).unsqueeze(0).to(self.device)  # Add batch and move to device
        loc, conf = self.detect_faces(image_tensor)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.generate_anchors().to(self.device)

        boxes = decode(loc.squeeze(0), priors, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).cpu().numpy()[:, 1]

        # Ignore low scores
        inds = scores > self.conf_threshold
        boxes = boxes[inds]
        scores = scores[inds]

        # Keep top-K before NMS
        order = scores.argsort()[::-1][:self.pre_nms_top_k]
        boxes = boxes[order]
        scores = scores[order]

        # Perform NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep]

        # Keep top-K after NMS
        dets = dets[:self.post_nms_top_k]

        return dets


def read_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    image_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_arr = np.float32(image_raw)
    image_arr -= cfg['rgb_mean']  # actually bgr mean
    image_arr = image_arr.transpose(2, 0, 1)  # HWC => CHW
    return image_raw, image_arr


def draw_bboxes(dets: np.ndarray, image_raw: np.ndarray, image_name: str, save_dir: str, vis_threshold: float) -> None:
    for det in dets:
        x1, y1, x2, y2, score = det
        if score >= vis_threshold:
            conf = f"{score:.4f}"
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(image_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image_raw, conf, (x1, y1 + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    save_path = f"{save_dir}/images/{image_name}.jpg"
    print(f"Saving annotated image to {save_path}")
    cv2.imwrite(save_path, image_raw)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Arguments for FaceBoxes Model')

    parser.add_argument(
        '--weights',
        default='./weights/faceboxes.pth',
        type=str,
        help='Path to the trained model state dict file.'
    )
    parser.add_argument(
        '--save-dir',
        default='results',
        type=str,
        help='Directory to save the detection results.'
    )
    parser.add_argument('--image-path', type=str, help='Path to the image for inference.')
    parser.add_argument(
        '--conf-threshold',
        default=0.05,
        type=float,
        help='Minimum confidence threshold for considering detections.'
    )
    parser.add_argument(
        '--pre-nms-top-k',
        default=300,
        type=int,
        help='Number of top bounding boxes to consider for NMS.'
    )
    parser.add_argument('--nms-threshold', default=0.2, type=float, help='Non-maximum suppression threshold.')
    parser.add_argument(
        '--post-nms-top-k',
        default=750,
        type=int,
        help='Number of top bounding boxes to keep after NMS.'
    )
    parser.add_argument('--vis-threshold', default=0.15, type=float, help='Visualization threshold for bounding boxes')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size for batch inference.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Create the save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(f"{save_dir}/images", exist_ok=True)

    inference = FaceBoxesInference(
        weights_path=args.weights,
        conf_threshold=args.conf_threshold,
        pre_nms_top_k=args.pre_nms_top_k,
        nms_threshold=args.nms_threshold,
        post_nms_top_k=args.post_nms_top_k
    )

    assert args.image_path, f"Please provide a value for `--image-path` argument"
    assert os.path.isfile(args.image_path), f"Please check the image path, {args.image_path}"

    image_raw, image_arr = read_image(args.image_path)
    dets = inference.process_image(image_arr)
    draw_bboxes(dets, image_raw, Path(args.image_path).stem, args.save_dir, args.vis_threshold)
