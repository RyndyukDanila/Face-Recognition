import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from utils_functions import read_features, get_feature, get_face
from utils_config import Config, Colors, Fonts


def recognition(face)-> tuple[int, float]:
    '''recognize face'''

    face_image = face[0]

    query_emb = get_feature(face_image, training=False)

    images_names, images_embs = read_features()

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]

    return (name, score)


def recognize_frame(frame, config: Config):
    '''recognize/detect frame'''

    bboxs, landmarks = get_face(frame, config)

    if config.recognize:
        faces = []

    for i, bbox in enumerate(bboxs):
        x1, y1, x2, y2 = bbox
        face_image = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), config.colors.background, config.fonts.thickness)

        if config.show_landmarks:
            for x in range(5):
                point_x = int(landmarks[i][2 * x])
                point_y = int(landmarks[i][2 * x + 1])
                cv2.circle(frame, (point_x, point_y), 2, config.colors.landmarks[x], -1)

        if config.recognize:
            faces.append((face_image, bbox))

    if config.recognize:
        with ThreadPoolExecutor(16) as executor:
            results = executor.map(recognition, faces)

        for i, face in enumerate(results):
            x1, y1, x2, y2 = bboxs[i]
            name = face[0]
            score = face[1]
            t_size = (0, 0)
            caption = ''

            if name and score >= config.conf_score:
                caption = f"{name.split('_')[0].upper()} {score:.2f}"
                t_size = cv2.getTextSize(
                    caption,
                    config.fonts.capture,
                    config.fonts.thickness,
                    config.fonts.thickness
                    )[0]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x1 + t_size[0], y1 + t_size[1]),
                config.colors.background,
                -1
                )
            if caption:
                cv2.putText(
                    frame,
                    caption,
                    (x1, y1 + t_size[1]),
                    config.fonts.capture,
                    config.fonts.thickness,
                    config.colors.caption,
                    config.fonts.thickness
                    )

    return frame


def recognize_video(config: Config) -> None:
    '''recognize/detect video using recognize_frame method'''

    if config.use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(config.video_source_path)

    if config.save_video:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        video = cv2.VideoWriter(
            config.save_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            config.post_fps,
            size
            )

    start = time.time_ns()
    fps = 0

    while cap.isOpened():
        _, frame = cap.read()

        frame = recognize_frame(frame, config)

        if config.count_fps:
            end = time.time_ns()
            fps = 1e9 * 1 / (end - start)
            start = time.time_ns()

            if fps > 0:
                fps_label = f"FPS: {round(fps)}"
                cv2.putText(frame, fps_label, (5, 30), config.fonts.fps, 1, config.colors.fps, 2)

        if config.save_video:
            video.write(frame)

        cv2.imshow("Face Recognition", frame)

        # press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    if config.save_video:
        video.release()
    cap.release()
    cv2.destroyAllWindows()


def recognize_image(config: Config) -> None:
    '''recognize/detect image using recognize_frame method'''

    image = cv2.imread(config.image_source_path)
    image = recognize_frame(image, config)

    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if config.save_image:
        cv2.imwrite(config.save_image_path, image)


if __name__=="__main__":
    colors = Colors(
        background=(0,255,0), # green
        # background=(0,255,255), # yellow
        )
    fonts = Fonts(
        thickness=1
    )
    config = Config(
        colors=colors,
        fonts=fonts,
        video_source_name='rock',
        save_video=True,
        post_fps=12,
        count_fps=True,
        use_webcam=False,
        image_source_name='rock',
        save_image=True,
        size_convert=512,
        conf_thres=0.6,
        iou_thres=0.01,
        show_landmarks=False,
        recognize=False,
        conf_score=0.5,
    )
    recognize_video(config=config)
    recognize_image(config=config)
