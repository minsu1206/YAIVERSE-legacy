import torch
from facenet_pytorch import InceptionResnetV1
# from deepface import DeepFace

def load_facenet():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model