import glob
import os
import cv2
import time
import face_detection
from face_detection.retinaface.tensorrt_wrap import TensorRTRetinaFace

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

if __name__ == "__main__":
    image = cv2.imread(".local/test_images/a00119.jpeg")
    width = 1280
    height = 720
    expected_imsize = (height, width)
    image = cv2.resize(image, (width, height))
    detector = TensorRTRetinaFace(
        (height, width),
        (480, 640))
    print(detector.infer(image))
    boxes, landms, scores = detector.infer(image)
    for i in range(boxes.shape[0]):
        print(boxes[i])
        x0, y0, x1, y1 = boxes[i].astype(int)
        image = cv2.rectangle(image, (x0, y0), (x1, y1),(255, 0, 0), 1 )
        for kp in landms[i]:
            image = cv2.circle(image, tuple(kp), 5, (255, 0, 0))
    cv2.imwrite("test.png", image)
