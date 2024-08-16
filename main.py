import cv2
import os
import mediapipe as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='webcam')
parser.add_argument("--filePath", default=None)
args = parser.parse_args()

print(args)
            
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detect faces
mp_face_detection = mp.solutions.face_detection

# out.detections gives an output like this.
"""
[label_id: 0
score: 0.950264394
location_data {
  format: RELATIVE_BOUNDING_BOX
  relative_bounding_box {
    xmin: 0.302663326
    ymin: 0.268652439
    width: 0.360203683
    height: 0.334942758
  }
  relative_keypoints {
    x: 0.42136991
    y: 0.360001385
  }
  relative_keypoints {
    x: 0.560749769
    y: 0.36526075
  }
  relative_keypoints {
    x: 0.493497878
    y: 0.447201222
  }
  relative_keypoints {
    x: 0.486922294
    y: 0.510917842
  }
  relative_keypoints {
    x: 0.33247152
    y: 0.383688301
  }
  relative_keypoints {
    x: 0.626719296
    y: 0.396808326
  }
}
]
"""

def processImage(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    # If faces in the image
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            
            # Converting from relative value to image coordinates
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            # Bounding Box around the image
            # img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 10, cv2.LINE_AA)
            
            # Blur Faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (30, 30))
            
    return img


# Model selection 0 -> face within 2 meters of camera; 1 -> face within 5 meters of camera
with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    if args.mode == 'image':
        img = cv2.imread(args.filePath)
        img = processImage(img, face_detection)
        cv2.imwrite(os.path.join(output_dir, 'blurred.jpg'), img)   # save image
    
    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.filePath)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output_video = cv2.VideoWriter(
            os.path.join(output_dir, 'output.mp4'),
            cv2.VideoWriter_fourcc(*"MP4V"),
            cap.get(cv2.CAP_PROP_FPS),
            frame_size
        )
        ret, frame = cap.read() # Reading the first frame
        while ret:
            frame = processImage(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()
            
        cap.release()
        output_video.release()
        
    elif args.mode == 'webcam':
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        ret, frame = cap.read()
        if opened:
            while ret:
                frame = processImage(frame, face_detection)
                cv2.imshow('Live Blur', frame)
                ret, frame = cap.read()
                
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    print("Quitting...")
                    break
            cap.release()
            
            