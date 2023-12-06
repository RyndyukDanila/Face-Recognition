import cv2


class Config():

    def __init__(self, video_source_name='oscar2023', image_source_name='rock'):
        self._video_source_name = video_source_name
        self._image_source_name = image_source_name

        # changeable
        self.recognize = True
        self.use_webcam = False
        self.post_fps = 12

        self.count_fps = True
        self.show_landmarks = False
        self.record_video = True

        self.size_convert = 512
        self.conf_thres = 0.6
        self.iou_thres = 0.01
        
        # prefer not to change
        self._video_name = 'webcam' if self.use_webcam else self._video_source_name
        self._recognition_or_detect = 'recognition' if self.recognize else 'detect'
        self._have_landmarks = '-landmarks' if self.show_landmarks else ''
        self.video_source_path = f'./static/videos/{self._video_source_name}.mp4'
        self.image_source_path = f'./static/images/{self._image_source_name}.jpg'

        self.save_directory = './static/results/'
        self.save_video_path = f'{self.save_directory}{self._video_name}-{self._recognition_or_detect}-{self.post_fps}fps-{self.conf_thres}conf-{self.iou_thres}iou-{self.size_convert}{self._have_landmarks}.mp4'
        self.save_image_path = f'{self.save_directory}{self._image_source_name}-{self._recognition_or_detect}-{self.conf_thres}conf-{self.iou_thres}iou-{self.size_convert}{self._have_landmarks}.jpg'

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
        