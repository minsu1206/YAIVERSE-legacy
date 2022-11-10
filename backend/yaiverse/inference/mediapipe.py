# Upload images that contain face(s) within 2 meters from the camera.
import cv2
#from google.colab.patches import cv2_imshow
import math
import numpy as np
import mediapipe as mp
import os
from PIL import Image
from yaiverse.inference.util_mp import align_face_mp


def face_detection(img_dir:str):
    img = cv2.imread(img_dir)
    set_size = 480
    y_ratio = img.shape[0]/set_size
    x_ratio = img.shape[1]/set_size
    img = cv2.resize(img, (set_size, set_size))
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
                lmk_x = p_key_point(detection, i).x * img.shape[0] * x_ratio
                lmk_y = p_key_point(detection, i).y * img.shape[1] * y_ratio
                face_lmk.append(np.array([int(lmk_x), int(lmk_y)])) ## i : index of lmk_name
        # resize_and_show(annotated_image)
    
        bbox_lst = []
        bbox = face_detection.process(img)
        bbox = bbox.detections[0].location_data.relative_bounding_box
        bbox_lst.append(int(bbox.xmin * set_size * x_ratio)) # box_x
        bbox_lst.append(int(bbox.ymin * set_size * y_ratio)) # box_y
        bbox_lst.append(int(bbox.width * set_size* x_ratio)) # box_w
        bbox_lst.append(int(bbox.height * set_size * y_ratio)) # box_h
        print(bbox_lst)
        
    cv2.imwrite(img_dir.replace("image.jpg", "bbox.jpg"), cv2.resize(annotated_image, (int(x_ratio*set_size), int(y_ratio*set_size))))
    
    return align_face_mp(img_dir, face_lmk, set_size, bbox_lst)
