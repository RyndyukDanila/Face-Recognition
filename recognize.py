from utils_functions import *

import time
from concurrent.futures import ThreadPoolExecutor
from utils_config import UtilsConfig as Config


def recognition(face):
    face_image = face[0]
    
    query_emb = (get_feature(face_image, training=False))
    
    images_names, images_embs = read_features() 

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]

    return (name, score)    


def main():     
    if Config.use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(Config.source_path)
    
    if Config.record_video: 
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        video = cv2.VideoWriter(Config.save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), Config.post_fps, size)

    start = time.time_ns()
    fps = 0
    score = 0
    name = null
    
    while cap.isOpened():
        _, frame = cap.read()
        
        bboxs, landmarks = get_face(frame)
        if Config.recognize:
            faces = []
        
        for i in range(len(bboxs)):
            x1, y1, x2, y2 = bboxs[i]
            face_image = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.Colors.background, Config.Fonts.thickness)

            if Config.show_landmarks:
                for x in range(5):
                    point_x = int(landmarks[i][2 * x])
                    point_y = int(landmarks[i][2 * x + 1])
                    cv2.circle(frame, (point_x, point_y), 2, Config.Colors.landmarks[x], -1)

            if Config.recognize:
                faces.append((face_image, bboxs[i]))

        if Config.recognize:
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
                    t_size = cv2.getTextSize(caption, Config.Fonts.capture, Config.Fonts.thickness, Config.Fonts.thickness)[0]

                cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), Config.Colors.background, -1)
                if caption:
                    cv2.putText(frame, caption, (x1, y1 + t_size[1]), Config.Fonts.capture, Config.Fonts.thickness, Config.Colors.caption, Config.Fonts.thickness)   

        if Config.count_fps:
            end = time.time_ns()
            fps = 1e9 * 1 / (end - start)
            start = time.time_ns()
    
            if fps > 0:
                fps_label = "FPS: %.2f" % fps
                cv2.putText(frame, fps_label, (5, 30), Config.Fonts.fps, 1, Config.Colors.fps, 2)
        
        if Config.record_video:
            video.write(frame)

        cv2.imshow("Face Recognition", frame)
        
        # press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
    
    if Config.record_video:
        video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

if __name__=="__main__":
    main()