import cv2
import random
import numpy as np
from utils.box_utils import matrix_iof


def draw_detections(original_image, detections, vis_threshold):
    """
    Draws bounding boxes and landmarks on the image based on multiple detections.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of detected bounding boxes and landmarks.
        vis_threshold (float): The confidence threshold for displaying detections.
    """

    # Colors for visualization
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    # Filter by confidence
    detections = detections[detections[:, 4] >= vis_threshold]

    print(f"#faces: {len(detections)}")

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]

    for box, score in zip(boxes, scores):
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

        # Draw confidence score
        text = f"{score:.2f}"
        cx, cy = box[0], box[1] + 12
        cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)


def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 16.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def distort_image(image):
    """Applies random color distortions to an image, including brightness, contrast, saturation, and hue adjustments.

    Args:
        image (numpy.ndarray): The input image to be distorted.

    Returns:
        numpy.ndarray: The distorted image.
    """

    def apply_adjustments(image, alpha=1.0, beta=0.0):
        """Adjusts the image based on alpha (contrast) and beta (brightness) values.

        Args:
            image (numpy.ndarray): The image to be adjusted.
            alpha (float): Contrast adjustment factor.
            beta (float): Brightness adjustment factor.
        """
        tmp = image.astype(float) * alpha + beta
        np.clip(tmp, 0, 255, out=tmp)  # Ensure values are within [0, 255]
        image[:] = tmp

    image = image.copy()

    # Brightness distortion
    if random.random() >= 0.5:
        apply_adjustments(image, beta=random.uniform(-32, 32))

    # Contrast distortion
    if random.random() >= 0.5:
        apply_adjustments(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Saturation distortion
    if random.random() >= 0.5:
        apply_adjustments(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    # Hue distortion
    if random.random() >= 0.5:
        hue_shift = image[:, :, 0].astype(int) + random.randint(-18, 18)
        image[:, :, 0] = np.mod(hue_shift, 180)

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def horizontal_flip(image, boxes, p=0.5):
    # Get the height and width of the image
    height, width, _ = image.shape

    if random.random() >= p:
        # Flip the image horizontally
        flipped_image = np.fliplr(image)

        # Create a copy of the bounding boxes to adjust
        adjusted_boxes = boxes.copy()

        # Adjust the x-coordinates of the bounding boxes
        # The x-coordinates are mirrored around the center of the image
        adjusted_boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        return flipped_image, adjusted_boxes

    # If the image is not flipped, return the original image and bounding boxes
    return image, boxes


def pad_image_to_square(image, bgr_mean, pad_image):
    if not pad_image:
        return image

    # Get the dimensions of the original image
    height, width, _ = image.shape

    # Determine the length of the longest side
    longest_side = max(width, height)

    # Create an empty square image with the RGB mean value
    padded_image = np.empty((longest_side, longest_side, 3), dtype=image.dtype)
    padded_image[:, :] = bgr_mean

    # Place the original image in the top-left corner of the padded image
    padded_image[0:height, 0:width] = image

    return padded_image


def resize_subtract_mean(image, target_size, bgr_mean):
    # Define interpolation methods
    interpolation_methods = [
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4
    ]

    # Randomly select an interpolation method
    interpolation_method = random.choice(interpolation_methods)

    # Resize the image
    resized_image = cv2.resize(image, (target_size, target_size), interpolation=interpolation_method)

    # Convert image to float32
    resized_image = resized_image.astype(np.float32)

    # Subtract the RGB mean
    resized_image -= bgr_mean

    # Transpose the image to (C, H, W)
    processed_image = resized_image.transpose(2, 0, 1)

    return processed_image


class Augmentation:
    def __init__(self, image_size, bgr_mean):
        self.image_size = image_size
        self.bgr_mean = bgr_mean

    def __call__(self, image, targets):
        # Ensure the image has ground truth
        assert targets.shape[0] > 0, "This image does not have ground truth."

        # Separate bounding boxes and labels
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # Crop the image and adjust boxes and labels
        image, boxes, labels, requires_padding = _crop(image, boxes, labels, self.image_size)

        # Apply image distortions
        image = distort_image(image)

        # Pad the image to a square
        image = pad_image_to_square(image, self.bgr_mean, requires_padding)

        # Apply horizontal flip
        image, boxes = horizontal_flip(image, boxes)

        # Get the dimensions of the image
        height, width, _ = image.shape

        # Resize image and subtract mean BGR values
        image = resize_subtract_mean(image, self.image_size, self.bgr_mean)

        # Normalize the bounding boxes
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        # Expand labels dimension and concatenate with boxes
        labels = np.expand_dims(labels, 1)
        targets = np.hstack((boxes, labels))

        return image, targets
