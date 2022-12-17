import time
import numpy as np

import cv2
import torch
from torchvision.utils import draw_bounding_boxes

# initilize the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# open camera feed
cam = cv2.VideoCapture(0)

while(True):
    # Capture the camera frame
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # run the frame through the model
    result = model(frame)
    
    # print results to terminal
    result.print()

    # extract labels and bounding boxes
    labels = (result.pandas().xyxy[0])["name"]
    bboxes = (result.xyxy[0])[:, 0:4]
    
    # convert opencv frame to pytorch tensor
    frame = np.moveaxis(frame, 2, 0)
    frame = torch.from_numpy(frame)
    
    # draw bounding boxes
    out_frame = draw_bounding_boxes(frame, bboxes, labels.values, width=2)
    
    # convert the frame back into opencv format
    out_frame = out_frame.numpy()
    out_frame = np.moveaxis(out_frame, 0, 2)
    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('frame', out_frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()