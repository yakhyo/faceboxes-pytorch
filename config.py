cfg = {
    'name': 'FaceBoxes',
    'image_size': 640,
    'feature_maps': [[32, 32], [16, 16], [8, 8]],
    'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'milestones': [100, 200],
    'epochs': 300,
    'rgb_mean': (104, 117, 123)  # bgr order
}
