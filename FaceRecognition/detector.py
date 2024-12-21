"""Face Recogntion by Charles Umesi (19 September 2023)"""

from pathlib import Path
import argparse
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw

import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix  # Comment out during training


# Defining the path for encodings from training image files
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

# Defining bounding boxes for faces
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

predicted_names = []

# Defining arguments for options on launch of program
parser = argparse.ArgumentParser(description="Recognise faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "--accuracy", action="store_true", help="Check accuracy of predictions"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
args = parser.parse_args()

# Auto-generation of appropriate folders
Path("training").mkdir(exist_ok=True)  # This folder will already need to exist as you should have placed in it the photos for training!
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

# Bounding faces function
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

# Encoding faces function
def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
        ) -> None:
    """
    Encodes images in the training directory and stores them in a dictionary
    """
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

encode_known_faces()

# Recognise faces function
def recognise_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODINGS_PATH,
        ) -> None:   
    """
    Accepts an unknown image, encodes it, and tells you whether there are matches with the stored known encodings
    """   
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    if args.validate or args.test:

        pillow_image = Image.fromarray(input_image)
        draw = ImageDraw.Draw(pillow_image)
        
        for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
        ):
            name = _recognise_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            #print(name, bounding_box) #Optional
            _display_face(draw, bounding_box, name)

        del draw
        pillow_image.show()

    elif args.accuracy:

        for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
        ):
            predicted_name = _recognise_face(unknown_encoding, loaded_encodings)
            if not predicted_name:
                predicted_name = "Unknown"
            predicted_names.append(predicted_name)

# Recognise test face function
def _recognise_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    """
    Compares an unknown encoding with stored known encodings, 
    returning encodings that match the most with the unknown encoding
    """
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

# Validation function
def validate(model: str = "hog"):
    """
    Runs recognise_faces on a set of images with known faces to validate known encodings
    """
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognise_faces(
                image_location=str(filepath.absolute()), model=model
            )

# Accuracy of prediction function  # Comment this function out during training as it appears to prevent training
# def accuracy(name: str,
#              model: str = "hog"
#              ) -> None:
#     """
#     Quantifies accuracy of face recognition predictions
#     """
#     true_names = []

#     for filepath in Path("validation/" + name).rglob("*"):
#         true_name = str(filepath.parent.name)
#         true_names.append(true_name)
#         if filepath.is_file():
#             recognise_faces(
#                 image_location=str(filepath.absolute()), model=model
#             )
#     print(confusion_matrix(np.array(true_names),np.array(predicted_names)))
#     print('\n',classification_report(np.array(true_names),np.array(predicted_names)))



if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognise_faces(input('Enter image file name and its directory : '))
    # if args.accuracy:
    #     accuracy(input('Enter folder for accuracy of predictions : '))