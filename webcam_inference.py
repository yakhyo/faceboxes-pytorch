import os
import cv2
import argparse
import numpy as np

import torch

from config import cfg
from layers import PriorBox
from models import FaceBoxes
from utils.transform import draw_detections
from utils.box_utils import decode, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description="Retinaface Webcam Inference")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/Resnet34_Final.pth',
        help='Path to the trained model weights'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.4,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )

    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)

    return loc, conf


def resize_image(frame, target_shape=(640, 640)):
    width, height = target_shape

    # Aspect-ratio preserving resize
    im_ratio = float(frame.shape[0]) / frame.shape[1]
    model_ratio = height / width
    if im_ratio > model_ratio:
        new_height = height
        new_width = int(new_height / im_ratio)
    else:
        new_width = width
        new_height = int(new_width * im_ratio)

    resize_factor = float(new_height) / frame.shape[0]
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Create blank image and place resized image on it
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:new_height, :new_width, :] = resized_frame

    return image, resize_factor


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    # model initialization
    model = FaceBoxes(num_classes=2)
    model.to(device)

    # loading state_dict
    state_dict = torch.load(params.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # Open webcam
    cap = cv2.VideoCapture("assets/in_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        image, resize_factor = resize_image(frame, target_shape=(640, 640))

        # Prepare image for inference
        image = np.float32(image)
        img_height, img_width, _ = image.shape
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).unsqueeze(0).to(device)

        # forward pass
        loc, conf = inference(model, image)

        # generate anchor boxes
        priorbox = PriorBox(cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors().to(device)

        # decode boxes
        boxes = decode(loc, priors, cfg['variance'])

        # scale adjustments
        bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
        boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

        scores = conf.cpu().numpy()[:, 1]

        # filter by confidence threshold
        inds = scores > params.conf_threshold
        boxes = boxes[inds]
        scores = scores[inds]

        # sort by scores
        order = scores.argsort()[::-1][:params.pre_nms_topk]
        boxes,  scores = boxes[order],  scores[order]

        # apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, params.nms_threshold)

        detections = detections[keep]

        # keep top-k detections
        detections = detections[:params.post_nms_topk]

        # draw detections on the frame
        draw_detections(frame, detections, params.vis_threshold)

        out.write(frame)

        # Display the resulting frame
        # cv2.imshow('Webcam Inference', frame)

        # Press 'q' to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the webcam and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
