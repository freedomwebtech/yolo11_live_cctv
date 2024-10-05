import cv2
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
import cvzone
import numpy as np
stream = CamGear(source='https://www.youtube.com/watch?v=_TusTf0iZQU', stream_mode = True, logging=True).start() # YouTube Video URL as input

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")

# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('pt2.mp4')
count=0


while True:
    frame = stream.read()

  

    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,imgsz=240,classes=[2])

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
            cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
stream.stop()
cv2.destroyAllWindows()

