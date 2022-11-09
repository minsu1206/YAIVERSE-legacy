import torch
from facenet_pytorch import InceptionResnetV1
# from deepface import DeepFace

def load_facenet():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model

def CosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

if __name__ == "__main__":
    # for GPU memory check
    
    import time
    
    model = load_facenet()
    model.cuda()

    with torch.no_grad():
        start = time.time()
        while time.time() - start < 10:
            
            _ = model(torch.zeros((1, 3, 256, 256)).cuda())
            # print(time.time() - start)
