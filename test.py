import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn.functional as F


from utils.prior_box import PriorBox

from models.faceboxes import FaceBoxes
from utils.box_utils import decode, nms
from config import cfg

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Testing Arguments for FaceBoxes Model')

    parser.add_argument(
        '--weights',
        default='./weights/faceboxes.pth',
        type=str,
        help='Path to the trained model state dict file.'
    )
    parser.add_argument('--save-dir', default='eval/', type=str, help='Directory to save the detection results.')
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
        default=300,
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
    parser.add_argument('--vis-threshold', default=0.5, type=float,  help='Visualization threshold for bounding boxes')

    args = parser.parse_args()
    return args


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    os.makedirs(f"{params.save_dir}/{params.dataset}")

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
        for idx, img_name in enumerate(test_dataset):
            image_path = testset_folder / f'{img_name}.jpg'
            img_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            img = np.float32(img_raw)
            if resize != 1:
                img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width = img.shape[:2]
            scale = torch.tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], device=device)
            img -= cfg['rgb_mean']
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0).to(device)

            loc, conf = model(img)  # Forward pass
            conf = F.softmax(conf, dim=-1)

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.generate_anchors().to(device)

            boxes = decode(loc.data.squeeze(0), priors.data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

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
                fw.write(f'{img_name}\n{dets.shape[0]:.1f}\n')
                for det in dets:
                    xmin, ymin, xmax, ymax, score = det
                    w, h = xmax - xmin + 1, ymax - ymin + 1
                    fw.write(f'{xmin:.3f} {ymin:.3f} {w:.3f} {h:.3f} {score:.10f}\n')
            else:
                for det in dets:
                    xmin, ymin, xmax, ymax, score = det
                    ymin += 0.2 * (ymax - ymin + 1)
                    fw.write(f'{img_name} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}\n')

            print(f'Processing: {idx+1}/{num_images}')

            # Show image
            if params.show_image:
                for b in dets:
                    if b[4] < params.vis_threshold:
                        continue
                    text = f"{b[4]:.4f}"
                    b = list(map(int, b))
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cv2.putText(img_raw, text, (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                
                # cv2.imshow('res', img_raw)
                # cv2.waitKey(0)
                cv2.imwrite(f"{params.save_dir}/{params.dataset}/{img_name}.jpg", img_raw)


if __name__ == '__main__':
    args = parse_args()
    main(params=args)
