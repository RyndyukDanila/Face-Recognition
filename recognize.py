import os
from utils_functions import *

import time
from concurrent.futures import ThreadPoolExecutor
from utils_config import Config


def recognition(face):
    face_image = face[0]
    
    query_emb = (get_feature(face_image, training=False))
    
    images_names, images_embs = read_features() 

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]

    return (name, score)    


def recognize_video(config: Config):     
    if config.use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(config.video_source_path)
    
    if config.record_video: 
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        video = cv2.VideoWriter(config.save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), config.post_fps, size)

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
                fps_label = "FPS: %.2f" % fps
                cv2.putText(frame, fps_label, (5, 30), config.Fonts.fps, 1, config.Colors.fps, 2)
        
        if config.record_video:
            video.write(frame)

        cv2.imshow("Face Recognition", frame)
        
        # press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
    
    if config.record_video:
        video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


def recognize_image(config: Config):
    print(config.save_image_path)
    image = cv2.imread(config.image_source_path)
    image = recognize_frame(image, config)

    cv2.imshow('Face Recognition', image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    cv2.imwrite(config.save_image_path, image)


def recognize_frame(frame, config: Config):
    bboxs, landmarks = get_face(frame, config)

    if config.recognize:
        faces = []
        
    for i in range(len(bboxs)):
        x1, y1, x2, y2 = bboxs[i]
        face_image = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), config.Colors.background, config.Fonts.thickness)

        if config.show_landmarks:
            for x in range(5):
                point_x = int(landmarks[i][2 * x])
                point_y = int(landmarks[i][2 * x + 1])
                cv2.circle(frame, (point_x, point_y), 2, config.Colors.landmarks[x], -1)

        if config.recognize:
            faces.append((face_image, bboxs[i]))

    if config.recognize:
        with ThreadPoolExecutor(16) as executor:
            results = executor.map(recognition, faces)

        for i, face in enumerate(results):
            x1, y1, x2, y2 = bboxs[i]
            name = face[0]
            score = face[1]
            t_size = (0, 0)
            caption = ''

            if name and score >= 0.5:
                caption = f"{name.split('_')[0].upper()} {score:.2f}"
                t_size = cv2.getTextSize(caption, config.Fonts.capture, config.Fonts.thickness, config.Fonts.thickness)[0]

            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), config.Colors.background, -1)
            if caption:
                cv2.putText(frame, caption, (x1, y1 + t_size[1]), config.Fonts.capture, config.Fonts.thickness, config.Colors.caption, config.Fonts.thickness) 
    
    return frame
    

if __name__=="__main__":
    config = Config(video_source_name='oscar2023', image_source_name='rock')
    # recognize_video(config)
    recognize_image(config=config)