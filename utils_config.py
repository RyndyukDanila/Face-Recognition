import cv2


class UtilsConfig():
    # changeable
    recognize = True
    use_webcam = False
    post_fps = 12
    _source_name = 'oscar2023'

    count_fps = True
    show_landmarks = False
    record_video = True

    size_convert = 512
    conf_thres = 0.6
    iou_thres = 0.01
    
    # prefer not to change
    _video_name = 'webcam' if use_webcam else _source_name
    _video_type = 'recognition' if recognize else 'detect'
    _have_landmarks = '-landmarks' if show_landmarks else ''
    source_path = f'./static/videos/{_source_name}.mp4'
    save_video_path = f'./static/results/{_video_name}-{_video_type}-{post_fps}fps-{conf_thres}conf-{iou_thres}iou-{size_convert}{_have_landmarks}.mp4'


    class Fonts():
        capture = cv2.FONT_HERSHEY_PLAIN
        fps = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

    class Colors():
        # BGR
        background = (55,255,55)

        black = [0,0,0]
        white = (255,255,255)

        red = (0,0,255)
        green = (0,255,0)
        blue = (255,0,0)
        yellow = (0,255,255)
        cyan = (255,255,0)

        caption = black
        fps = red

        landmarks = [blue, green, red, cyan, yellow]
        