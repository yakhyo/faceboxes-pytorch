import os
import cv2
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from config import cfg
from models.faceboxes import FaceBoxes
from utils.prior_box import PriorBox
from utils.box_utils import decode, nms


def parse_args():
    parser = argparse.ArgumentParser(description='Testing Arguments for FaceBoxes Model')

    parser.add_argument(
        '--weights',
        default='./weights/faceboxes.pth',
        type=str,
        help='Path to the trained model state dict file.'
    )
    parser.add_argument('--save-dir', default='eval', type=str, help='Directory to save the detection results.')
    parser.add_argument(
        '--dataset',
        default='PASCAL',
        type=str,
        choices=['AFW', 'PASCAL', 'FDDB'],
        help='Select the dataset to evaluate on.'
    )
    parser.add_argument(
        '--conf-threshold',
        default=0.05,
        type=float,
        help='Minimum confidence threshold for considering detections.'
    )
    parser.add_argument(
        '--pre-nms-top-k',
        default=5000,
        type=int,
        help='Number of top bounding boxes to consider for NMS.'
    )
    parser.add_argument('--nms-threshold', default=0.3, type=float, help='Non-maximum suppression threshold.')
    parser.add_argument(
        '--post-nms-top-k',
        default=750,
        type=int,
        help='Number of top bounding boxes to keep after NMS.'
    )
    parser.add_argument('--show-image', action="store_true", default=False, help='Display detection results on images.')
    parser.add_argument('--vis-threshold', default=0.5, type=float, help='Visualization threshold for bounding boxes')

    args = parser.parse_args()
    return args


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (104, 117, 123)
    # Model initialization
    model = FaceBoxes(num_classes=2)
    model.eval()
    model.to(device)

    # Load pretrained model weights
    model.load_state_dict(torch.load(params.weights, map_location=device))
    print('Finished loading model!')

    # Create folder to save results if not exists
    save_dir = Path(params.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # To save inferenced images
    os.makedirs(f"{params.save_dir}/{params.dataset}", exist_ok=True)

    with open(save_dir / f'{params.dataset}_dets.txt', 'w') as fw:

        # Testing dataset
        testset_folder = Path('data') / params.dataset / 'images'
        testset_list = Path('data') / params.dataset / 'img_list.txt'

        with open(testset_list, 'r') as fr:
            test_dataset = fr.read().split()
        num_images = len(test_dataset)

        # Testing scale
        resize_map = {"FDDB": 3, "PASCAL": 2.5, "AFW": 1}
        resize = resize_map.get(params.dataset, 1)

        # Testing begin
        for idx, image_name in enumerate(test_dataset):
            image_path = testset_folder / f'{image_name}.jpg'
            image_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image_np = np.float32(image_raw)
            if resize != 1:
                image_np = cv2.resize(image_np, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width = image_np.shape[:2]
            scale = torch.tensor([image_np.shape[1], image_np.shape[0],
                                  image_np.shape[1], image_np.shape[0]], device=device)
            image_np -= rgb_mean
            image_np = image_np.transpose(2, 0, 1)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(device)  # Add batch and move to device

            loc, conf = model(image_tensor)  # Forward pass
            conf = F.softmax(conf, dim=-1)

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.generate_anchors().to(device)

            boxes = decode(loc.squeeze(0), priors, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).cpu().numpy()[:, 1]

            # Ignore low scores
            inds = scores > params.conf_threshold
            boxes = boxes[inds]
            scores = scores[inds]

            # Keep top-K before NMS
            order = scores.argsort()[::-1][:params.pre_nms_top_k]
            boxes = boxes[order]
            scores = scores[order]

            # Perform NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(dets, params.nms_threshold)
            dets = dets[keep]

            # Keep top-K after NMS
            dets = dets[:params.post_nms_top_k]

            # Save detections
            if params.dataset == "FDDB":
                fw.write(f'{image_name}\n{dets.shape[0]:.1f}\n')
                for det in dets:
                    xmin, ymin, xmax, ymax, score = det
                    w, h = xmax - xmin + 1, ymax - ymin + 1
                    fw.write(f'{xmin:.3f} {ymin:.3f} {w:.3f} {h:.3f} {score:.10f}\n')
            else:
                for det in dets:
                    xmin, ymin, xmax, ymax, score = det
                    ymin += 0.2 * (ymax - ymin + 1)
                    fw.write(f'{image_name} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}\n')

            print(f'Processing: {idx + 1}/{num_images} | {params.save_dir}/{params.dataset}/{image_name}.jpg')

            # Show image
            if params.show_image:
                for det in dets:
                    xmin, ymin, xmax, ymax, score = det
                    if score < params.vis_threshold:
                        continue
                    text = f"{score:.4f}"
                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    cv2.rectangle(image_raw, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.putText(image_raw, text, (xmin, ymin + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # cv2.imshow('res', image_raw)
                # cv2.waitKey(0)
                cv2.imwrite(f"{params.save_dir}/{params.dataset}/{image_name}.jpg", image_raw)


if __name__ == '__main__':
    args = parse_args()
    main(params=args)
