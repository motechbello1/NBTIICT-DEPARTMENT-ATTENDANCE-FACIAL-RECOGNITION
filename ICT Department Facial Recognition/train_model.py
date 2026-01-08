import cv2
import os
import numpy as np
from PIL import Image
import pickle

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"

def train_recognizer():
    # Create the Local Binary Patterns Histograms (LBPH) Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # We still need the detector to make sure we are looking at faces
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("[INFO] Training started. This might take a few seconds...")

    current_id = 0
    label_ids = {}      # Dictionary to map Names -> IDs (e.g. "Bello": 0)
    y_labels = []       # List of IDs for every image
    x_train = []        # List of image arrays (pixel data)

    # Walk through the dataset folder
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                
                # Get the folder name (which is the person's name_ID)
                # Structure: dataset/Bello_NBTI-PER-1190/User...
                person_label = os.path.basename(root) 
                
                # If we haven't seen this person before, give them an ID
                if person_label not in label_ids:
                    label_ids[person_label] = current_id
                    current_id += 1
                
                id_ = label_ids[person_label]

                # Convert image to Grayscale and Numpy Array
                pil_image = Image.open(path).convert("L") # L = Grayscale
                image_array = np.array(pil_image, "uint8")

                # (Optional) Detect face again to be 100% sure we are training on a face
                # Since we already cropped in the previous script, this is a safety check.
                faces = detector.detectMultiScale(image_array)

                for (x, y, w, h) in faces:
                    # Add the face pixels to training data
                    # We create a region of interest (ROI)
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    # --- THE TRAINING HAPPENS HERE ---
    if len(x_train) == 0:
        print("[ERROR] No data found! Did you run manual_capture.py first?")
        return

    print(f"[INFO] Training on {len(x_train)} images for {len(label_ids)} people...")
    
    recognizer.train(x_train, np.array(y_labels))
    
    # Save the model
    recognizer.save(TRAINER_FILE)
    
    # Save the labels so we know who is who later
    with open(LABELS_FILE, 'wb') as f:
        pickle.dump(label_ids, f)

    print("--- TRAINING COMPLETE ---")
    print(f"[INFO] Model saved to '{TRAINER_FILE}'")
    print(f"[INFO] Labels saved to '{LABELS_FILE}'")

if __name__ == "__main__":
    train_recognizer()