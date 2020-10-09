import os
import cv2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from youtube_dl import YoutubeDL
from config.config import DEVICE, SCORE_THRESH


class Manager(object):
    def __init__(self):
        self.downloader = YoutubeDL({'outtmpl': 'data/tmp/%(id)s.%(ext)s'})
        self.cfg = get_cfg()

    """def download_video(self, link):
        with self.downloader:
            info = self.downloader.extract_info(link, download=True)
            self.processed_file = f'{info["id"]}.{info["ext"]}'"""

    def prepare_processing(self, link):
        with self.downloader:
            info = self.downloader.extract_info(link, download=True)
            self.processed_file = f'{info["id"]}.{info["ext"]}'

        self.cfg.merge_from_file(model_zoo.get_config_file(
            'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = DEVICE

        self.video = cv2.VideoCapture(f'data/tmp/{self.processed_file}')

        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = self.video.get(cv2.CAP_PROP_FPS)

        self.frames_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.output_video = cv2.VideoWriter(
            filename=f'data/results/{self.processed_file}',
            fourcc=cv2.VideoWriter_fourcc(*"avc1"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )

    def finish_processing(self):
        self.video.release()
        self.output_video.release()
        os.remove(f'data/tmp/{self.processed_file}')
