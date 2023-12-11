import cv2


class Config():

    def __init__(
            self, 
            colors,
            fonts,
            video_source_name='oscar2023', 
            save_video = True,
            post_fps = 12,
            count_fps = True,
            use_webcam = False,
            image_source_name='rock',
            save_image = True,
            recognize = True,
            show_landmarks = False,
            size_convert = 512,
            conf_thres = 0.6,
            iou_thres = 0.01,
            conf_score = 0.5
            ) -> None:
        
        self.colors = colors
        self.fonts = fonts
        self._video_source_name = video_source_name
        self.save_video = save_video
        self.post_fps = post_fps
        self.count_fps = count_fps
        self.use_webcam = use_webcam
        self._image_source_name = image_source_name
        self.save_image = save_image
        self.recognize = recognize
        self.show_landmarks = show_landmarks
        self.size_convert = size_convert
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.conf_score = conf_score

        
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

    def __init__(
            self, 
            capture = cv2.FONT_HERSHEY_PLAIN, 
            fps = cv2.FONT_HERSHEY_SIMPLEX, 
            thickness = 1
            ) -> None:
        self.capture = capture
        self.fps = fps
        self.thickness = thickness


class Colors():
    
    def __init__(
            self,
            background = (55,255,55)
            ) -> None:
        self.black = [0,0,0]
        self.white = (255,255,255)
        self.red = (0,0,255)
        self.green = (0,255,0)
        self.blue = (255,0,0)
        self.yellow = (0,255,255)
        self.cyan = (255,255,0)

        self.background = background
        self.caption = self.black
        self.fps = self.red
        self.landmarks = [self.blue, self.green, self.red, self.cyan, self.yellow]
        