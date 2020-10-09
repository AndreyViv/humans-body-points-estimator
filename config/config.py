# Device config. Set 'cpu' or 'cuda' device
DEVICE = 'cpu'

# Main score threshold
SCORE_THRESH = 0.9

# Keypoints score threshold for keypoints logging.
# Default value = 0.05 and got from Detectron2 Visualizer (PROB_THERSH).
# Rise up this param for logging more visible keypoints. It`s means
# that key points will visualized in to result video file, but not logged to
# result text file
POINT_THRESH = 0.05

# Names for body points
BODY_POINTS = {
    0: 'nose',
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}
