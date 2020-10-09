import datetime

from config.config import BODY_POINTS, POINT_THRESH


def make_log(frame_pos, instanses) -> str:
    result = '***\n\n'

    persons_count = len(instanses)
    frame_time = datetime.datetime.fromtimestamp(
        frame_pos/1000.0, tz=datetime.timezone(
            datetime.timedelta(0))).strftime('%H:%M:%S.%f')[:-3]

    result += f'Timing: {frame_time}\n'

    if persons_count > 0:
        result += f'Detected {persons_count} person(s)\n\n'

        for i, keypoints in enumerate(instanses.pred_keypoints, start=1):
            result += f'Person #{i} keypoints:\n'

            for i, keypoint in enumerate(keypoints):
                if keypoint[2] > POINT_THRESH:
                    xy_keypoint = f'{keypoint[0]}, {keypoint[1]}'
                else:
                    xy_keypoint = 'NOT DETECTED'

                result += f'{BODY_POINTS[i]} [{xy_keypoint}]\n'

            result += '\n'
    else:
        result += 'Persons not detected\n'

    return result
