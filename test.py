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


def parse_args():
    parser = argparse.ArgumentParser(description='Testing Arguments for FaceBoxes Model')

    parser.add_argument(
        '--weights',
        default='./weights/FaceBoxes_epoch_295_1.pth',
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
        default=5000,
        type=int,
        help='Number of top bounding boxes to consider for NMS.'
    )
    parser.add_argument('--nms-threshold', default=0.3, type=float, help='Non-maximum suppression threshold.')
    parser.add_argument(
        '--post-nms-top-k',
        default=1000,
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
    model = model.to(device)
    
    # Load pretrained model weights
    model.load_state_dict(torch.load(params.weights, map_location="cpu"))
    print('Finished loading model!')

    # Create folder to save results if not exists
    os.makedirs(params.save_dir, exist_ok=True)

    fw = open(os.path.join(params.save_dir, params.dataset + '_dets.txt'), 'w')

    # testing dataset
    testset_folder = os.path.join('data', params.dataset, 'images/')
    testset_list = os.path.join('data', params.dataset, 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing scale
    if params.dataset == "FDDB":
        resize = 3
    elif params.dataset == "PASCAL":
        resize = 2.5
    elif params.dataset == "AFW":
        resize = 1

    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name + '.jpg'
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= cfg['rgb_mean']
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf = model(img)  # forward pass
        conf = F.softmax(conf, dim=-1)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.generate_anchors()
        priors = priors.to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > params.conf_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:params.pre_nms_top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, params.nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:params.post_nms_top_k, :]

        # save dets
        if params.dataset == "FDDB":
            fw.write('{:s}\n'.format(img_name))
            fw.write('{:.1f}\n'.format(dets.shape[0]))
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
        else:
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                ymin += 0.2 * (ymax - ymin + 1)
                score = dets[k, 4]
                fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img_name, score, xmin, ymin, xmax, ymax))
        print('im_detect: {:d}/{:d}'.format(i + 1, num_images))

        # show image
        if params.show_image:
            for b in dets:
                if b[4] < params.vis_threshold:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # cv2.imshow('res', img_raw)
            # cv2.waitKey(0)
            cv2.imwrite(f"results/pascal/{img_name}.png", img_raw)

    fw.close()


if __name__ == '__main__':
    args = parse_args()
    main(params=args)
