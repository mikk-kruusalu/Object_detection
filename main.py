import time
import numpy as np

import cv2
import torch

# initilize the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# open camera feed
cam = cv2.VideoCapture(0)

while(True):
    # Capture the camera frame
    ret, frame = cam.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame[..., ::-1]
    
    # run the frame through the model
    sec = time.time()
    result = model(frame)[0]
    print(f"Propagation time: {time.time() - sec} s")

    result.print()
    result.show()

    # Display the resulting frame
    #cv2.imshow('frame', box)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()