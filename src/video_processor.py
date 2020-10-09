import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from src.licenced_scripts.fb_predictor import AsyncPredictor
from collections import deque

from config.config import DEVICE


class VideoProcessor(object):
    def __init__(self, cfg):
        self.cpu_device = torch.device("cpu")
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
            self.parallel = True
            self.predictor = AsyncPredictor(cfg, torch.cuda.device_count())
        else:
            self.parallel = False
            self.predictor = DefaultPredictor(cfg)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()

            if success:
                frame_pos = video.get(cv2.CAP_PROP_POS_MSEC)
                yield frame, frame_pos
            else:
                break

    def run(self, video):
        video_visualizer = VideoVisualizer(self.metadata, ColorMode.IMAGE)

        def process_predictions(frame, predictions):
            predictions = predictions["instances"].to(self.cpu_device)

            vis_frame = video_visualizer.draw_instance_predictions(
                frame, predictions)
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

            return vis_frame, predictions

        frame_gen = self._frame_from_video(video)

        if self.parallel:
            buffer_size = self.predictor.default_buffer_size
            frame_data = deque()

            for cnt, (frame, frame_pos) in enumerate(frame_gen):
                frame_data.append([frame, frame_pos])
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame, frame_pos = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield frame_pos, process_predictions(frame, predictions)

            while len(frame_data):
                frame, frame_pos = frame_data.popleft()
                predictions = self.predictor.get()
                yield frame_pos, process_predictions(frame, predictions)
        else:
            for frame, frame_pos in frame_gen:
                yield frame_pos, process_predictions(frame, self.predictor(frame))
