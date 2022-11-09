# Upload images that contain face(s) within 2 meters from the camera.
import files
import cv2
#from google.colab.patches import cv2_imshow
import math
import numpy as np
import mediapipe as mp
import os
from PIL import Image
from util_mp import align_face


set_size = 480
DESIRED_HEIGHT = set_size
DESIRED_WIDTH = set_size

def resize_and_show(image):
  h, w = image.shape[:2]
  img1 = cv2.resize(image, (set_size, set_size))
  cv2.imshow('img1', img1)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return img1


def face_detection(img):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils 
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    p_key_point = mp_face_detection.get_key_point
    p_face_part = mp_face_detection.FaceKeyPoint
    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5, model_selection=0) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        annotated_image = img.copy()
        face_lmk = []
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
            lmk_name = mp_face_detection.FaceKeyPoint._member_names_
            for i in range(len(lmk_name)):
                # bbox length..?
                lmk_x = p_key_point(detection, i).x * img.shape[0]
                lmk_y = p_key_point(detection, i).y * img.shape[1]
                face_lmk.append(np.array([lmk_x, lmk_y])) ## i : index of lmk_name
        resize_and_show(annotated_image)
    
        bbox_lst = []
        bbox = face_detection.process(img)
        bbox = bbox.detections[0].location_data.relative_bounding_box
        bbox_lst.append(bbox.xmin * set_size) # box_x
        bbox_lst.append(bbox.ymin * set_size) # box_y
        bbox_lst.append(bbox.width * set_size) # box_w
        bbox_lst.append(bbox.height * set_size) # box_h
    
    return face_lmk, annotated_image, bbox_lst
    
if __name__ == "__main__":
    os.getcwd()
    file_path = os.getcwd()

    img_name = 'img9.png'
    img_dir = file_path + '\\'+ img_name
    img = cv2.imread(img_dir) #Image.open(img_dir)
    img = resize_and_show(img)
    
    ### face detection
    face_lmk, img_detect, box_list = face_detection(img)
    cv2.imshow('img_detect', img_detect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ### use util_mp
    img_aligned = align_face(img_dir, face_lmk, set_size, box_list)
    img_aligned = cv2.cvtColor(np.array(img_aligned), cv2.COLOR_RGB2BGR)
    output_dir = file_path + '\\' + "output_mp"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_dir + '\\' + 'rst_' + img_name, img_aligned)
    cv2.imshow('img_aligned', img_aligned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

