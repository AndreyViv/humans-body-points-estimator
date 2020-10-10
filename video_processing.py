import argparse
import tqdm

from youtube_dl.utils import DownloadError
from src.manager import Manager
from src.video_processor import VideoProcessor
from src.keypoints_loger import make_log


def youtube_video_process(link: str):
    manager = Manager()
    manager.prepare_processing(link)

    processor = VideoProcessor(manager.cfg)
    process_gen = processor.run(manager.video)

    file_name = manager.processed_file.split(".")[0]
    frames_count = manager.frames_count

    output_txt_path = f'data/results/{file_name}.txt'

    with open(output_txt_path, 'a') as output_txt:
        for frame_pos, (vis_frame, predictions) in tqdm.tqdm(process_gen, total=frames_count):
            manager.output_video.write(vis_frame)
            output_txt.write(make_log(frame_pos, predictions))

    manager.finish_processing()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 human pose estimation")

    parser.add_argument(
        "--link",
        help="Link to youtube video",
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.link:
        try:
            youtube_video_process(args.link)
        except DownloadError:
            print(f'Can not download video from link: {args.link}')
        except Exception as ex:
            print(f'Something went wrong!\n{ex}')
    else:
        print('Link to YouTube video required')
