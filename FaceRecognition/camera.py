from pathlib import Path
import urllib.request
import numpy as np
import face_recognition
import cv2 as cv
import pickle
from collections import Counter
from django.conf import settings

"""For implementation in a Django project"""

# Defining the path for encodings from training image files
DEFAULT_ENCODINGS_PATH = Path("xxxx/output/encodings.pkl")
encodings_location = DEFAULT_ENCODINGS_PATH

# Defining bounding boxes for faces
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog" # cnn

with encodings_location.open(mode="rb") as f:
    loaded_encodings = pickle.load(f)

class Webcam(object):

    """Class for facial recognition webcam"""

    def __init__(self):
        self.webcam = cv.VideoCapture(0)

    def __del__(self):
        self.webcam.release()
       
    def get_frame(self):

        ret, frame = self.webcam.read()
        
        input_face_locations = face_recognition.face_locations(
            frame, model=MODEL
        )
        input_face_encodings = face_recognition.face_encodings(
            frame, input_face_locations
        )

        for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
        ):

            boolean_matches = face_recognition.compare_faces(
                loaded_encodings["encodings"], unknown_encoding
            )

            votes = Counter(
                name
                for match, name in zip(boolean_matches, loaded_encodings["names"])
                if match
            )
            
            # Define and execute bounding box for match   
            top_left = (bounding_box[3], bounding_box[0])
            bottom_right = (bounding_box[1], bounding_box[2])
            color = [255, 0, 0]
            cv.rectangle(frame, top_left, bottom_right, color, FRAME_THICKNESS)
            
            # Insert name of match in box
            if votes:
                name = votes.most_common(1)[0][0]
                top_left = (bounding_box[3], bounding_box[2])
                bottom_right = (bounding_box[1], bounding_box[2]+22)
                cv.rectangle(frame, top_left, bottom_right, color, cv.FILLED)
                cv.putText(frame, name, (bounding_box[3]+10, bounding_box[2]+15),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
            
            else:
                name = "Unknown"
                top_left = (bounding_box[3], bounding_box[2])
                bottom_right = (bounding_box[1], bounding_box[2]+22)
                cv.rectangle(frame, top_left, bottom_right, color, cv.FILLED)
                cv.putText(frame, name, (bounding_box[3]+10, bounding_box[2]+15),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)

            success, jpeg = cv.imencode(".jpg", frame)
            return jpeg.tobytes()
        
class IPWebcam:

    """Class for facial recognition IP webcam"""

    def __init__(self):
        self.url = "http://xxxxx/shot.jpg"

    def __del__(self):
        cv.destroyAllWindows()
       
    def get_frame(self):

        frame_req = urllib.request.urlopen(self.url)
        frame_np = np.array(bytearray(frame_req.read()))
        frame = cv.imdecode(frame_np, -1)
        frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)

        input_face_locations = face_recognition.face_locations(
            frame, model=MODEL
        )
        input_face_encodings = face_recognition.face_encodings(
            frame, input_face_locations
        )
        
        input_face_locations = face_recognition.face_locations(
            frame, model=MODEL
        )
        input_face_encodings = face_recognition.face_encodings(
            frame, input_face_locations
        )

        for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
        ):

            boolean_matches = face_recognition.compare_faces(
                loaded_encodings["encodings"], unknown_encoding
            )

            votes = Counter(
                name
                for match, name in zip(boolean_matches, loaded_encodings["names"])
                if match
            )
            
            # Define and execute bounding box for match   
            top_left = (bounding_box[3], bounding_box[0])
            bottom_right = (bounding_box[1], bounding_box[2])
            color = [255, 0, 0]
            cv.rectangle(frame, top_left, bottom_right, color, FRAME_THICKNESS)
            
            # Insert name of match in box
            if votes:
                name = votes.most_common(1)[0][0]
                top_left = (bounding_box[3], bounding_box[2])
                bottom_right = (bounding_box[1], bounding_box[2]+22)
                cv.rectangle(frame, top_left, bottom_right, color, cv.FILLED)
                cv.putText(frame, name, (bounding_box[3]+10, bounding_box[2]+15),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
            
            else:
                name = "Unknown"
                top_left = (bounding_box[3], bounding_box[2])
                bottom_right = (bounding_box[1], bounding_box[2]+22)
                cv.rectangle(frame, top_left, bottom_right, color, cv.FILLED)
                cv.putText(frame, name, (bounding_box[3]+10, bounding_box[2]+15),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)

            
            success, jpeg = cv.imencode(".jpg", frame)
            return jpeg.tobytes()
    

# No face detection or recognition; just (IP) webcam!

# class Webcam(object):

#     def __init__(self):
#         self.webcam = cv.VideoCapture(0)

#     def __del__(self):
#         self.webcam.release()
       
#     def get_frame(self):
#         ret, frame = self.webcam.read()
#         success, jpeg = cv.imencode(".jpg", frame)
#         return jpeg.tobytes()


# class IPWebcam(object):

#     def __init__(self, url):
#         self.url = url

#     def __del__(self):
#         cv.destroyAllWindows()
       
#     def get_frame(self):

#         frame_req = urllib.request.urlopen(self.url)
#         frame_np = np.array(bytearray(frame_req.read()), dtype=np.uint8)
#         frame = cv.imdecode(frame_np, -1)
#         frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
#         success, jpeg = cv.imencode(".jpg", frame)
#         return jpeg.tobytes()
    