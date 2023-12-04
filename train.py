from utils_functions import *

import os
import argparse
import shutil


def train(
        full_training_dir, 
        additional_training_dir, 
        faces_save_dir, 
        features_save_dir, 
        is_add_user
        ):
    
    # Init results output
    images_name = []
    images_emb = []
    
    # Check mode full training or additidonal
    if is_add_user == True:
        source = additional_training_dir
    else:
        source = full_training_dir

    print(additional_training_dir)
    
    # Read train folder, get and save face 
    for name_person in os.listdir(source):

        print(source)
        person_image_path = os.path.join(source, name_person)

        print(person_image_path)
        
        # Create path save person face
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)
        
        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", 'jpg', 'jpeg')):
                image_path = person_image_path + f"/{image_name}"
                input_image = cv2.imread(image_path)  # BGR 

                # Get faces
                bboxs, _ = get_face(input_image)

                # Get boxs
                for i in range(len(bboxs)):
                    # Get number files in person path
                    number_files = len(os.listdir(person_face_path))

                    # Get location face
                    x1, y1, x2, y2 = bboxs[i]

                    # Get face from location
                    face_image = input_image[y1:y2, x1:x2]

                    # Path save face
                    path_save_face = person_face_path + f"/{number_files}.jpg"
                    
                    # Save to face database 
                    cv2.imwrite(path_save_face, face_image)
                    
                    # Get feature from face
                    images_emb.append(get_feature(face_image, training=True))
                    images_name.append(name_person)
    
    # Convert to array
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)
    
    features = read_features(features_save_dir) 
    if features == null or is_add_user== False:
        pass
    else:        
        # Read features
        old_images_name, old_images_emb = features  
    
        # Add feature and name of image to feature database
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))
        
        print("feature updated")
    
    # Save features
    np.savez_compressed(features_save_dir, 
                        arr1 = images_name, arr2 = images_emb)
    
    # Move additional data to full train data
    if is_add_user == True:
        for sub_dir in os.listdir(additional_training_dir):
            dir_to_move = os.path.join(additional_training_dir, sub_dir)
            shutil.move(dir_to_move, full_training_dir, copy_function = shutil.copytree)
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-training-dir', type=str, default='./database/full-training-datasets/', help='dir folder full training')
    parser.add_argument('--additional-training-dir', type=str, default='./database/additional-training-datasets/', help='dir folder additional training')
    parser.add_argument('--faces-save-dir', type=str, default='./database/face-datasets/', help='dir folder save face features')
    parser.add_argument('--features-save-dir', type=str, default='./static/feature/face_features', help='dir folder save face features')
    parser.add_argument('--is-add-user', type=bool, default=True, help='Mode add user or full training')
    
    opt = parser.parse_args()
    return opt

def main(opt):
    train(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    