import cv2
import os
import time
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector

# --- CONFIGURATION ---
# Uses the current folder so it works on ANY machine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESTINATION_DIR = os.path.join(BASE_DIR, "ImagesAttendance")

IMAGES_TO_TAKE = 10             # How many photos to take
OFFSET = 30                     # Padding around face (Zoom level)

# Initialize Detector (High confidence for better quality)
detector = FaceDetector(minDetectionCon=0.75)

# --- HELPER FUNCTIONS ---
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_text(text):
    """Removes spaces and special chars."""
    return "".join([c for c in text if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- MAIN SYSTEM ---
def main_system():
    # Setup Main Folder
    create_dir(DESTINATION_DIR)
    
    while True:
        clear_screen()
        # --- 1. MAIN MENU ---
        print("\n" + "="*50)
        print("   📸 NBTI STAFF REGISTRATION SYSTEM")
        print("   (Auto-Crop & Format for Attendance App)")
        print("="*50)
        print("   Type 'exit' to quit at any time.\n")

        try:
            name = input("1. Enter Full Name      : ").strip()
            if name.lower() == 'exit': break
            if not name: continue 

            dept = input("2. Enter Department     : ").strip()
            if dept.lower() == 'exit': break

            role = input("3. Enter Job Role       : ").strip()
            if role.lower() == 'exit': break

            staff_id = input("4. Enter Staff ID       : ").strip()
            if staff_id.lower() == 'exit': break
            
            # Sanitize Inputs
            safe_name = clean_text(name)
            safe_dept = clean_text(dept)
            safe_role = clean_text(role)
            safe_id = clean_text(staff_id)

            # Folder Name: Name_Department_Role_ID
            folder_name = f"{safe_name}_{safe_dept}_{safe_role}_{safe_id}"
            user_folder_path = os.path.join(DESTINATION_DIR, folder_name)

            print(f"\n[INFO] Saving to: .../ImagesAttendance/{folder_name}")
            print("[INSTRUCTIONS]")
            print(" - Look at camera")
            print(" - Keep face inside the Green Box")
            print(" - Press 's' to START capturing")
            print(" - Press 'q' to CANCEL/QUIT")
            input("\nPress Enter to open camera...")

            # --- 2. CAMERA LOOP ---
            cap = cv2.VideoCapture(0)
            
            count = 0
            capturing = False
            
            while True:
                success, img = cap.read()
                if not success: break
                
                # 1. Mirror Flip
                img = cv2.flip(img, 1)
                img_clean = img.copy() # Keep a clean copy for saving
                
                # 2. Detect Face
                img, bboxs = detector.findFaces(img, draw=False)
                
                face_valid = False
                target_crop = None

                if bboxs:
                    # Get the largest face
                    x, y, w, h = bboxs[0]['bbox']
                    
                    # 3. Check Boundaries (Prevent crashing at edges)
                    # We ensure the crop area (plus OFFSET) is actually inside the image
                    if (x - OFFSET > 0) and (y - OFFSET > 0) and \
                       (x + w + OFFSET < img.shape[1]) and (y + h + OFFSET < img.shape[0]):
                        
                        face_valid = True
                        
                        # Crop the face with padding
                        target_crop = img_clean[y-OFFSET : y+h+OFFSET, x-OFFSET : x+w+OFFSET]
                        
                        # Visual: Draw Green Box
                        cv2.rectangle(img, (x-OFFSET, y-OFFSET), (x+w+OFFSET, y+h+OFFSET), (0, 255, 0), 2)
                    else:
                        # Visual: Draw Red Box (Too close to edge)
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(img, "Move to Center", (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255), 2)

                # --- UI OVERLAY ---
                # Top Bar Background
                cv2.rectangle(img, (0,0), (640, 60), (0,0,0), -1)
                
                if capturing:
                    status_color = (0, 255, 0) # Green
                    msg = f"CAPTURING: {count}/{IMAGES_TO_TAKE}"
                else:
                    status_color = (0, 200, 255) # Orange
                    msg = "Press 's' to Start | 'q' to Quit"
                
                cv2.putText(img, msg, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 1)

                cv2.imshow("NBTI Registrar", img)
                key = cv2.waitKey(1)

                # --- CONTROLS ---
                if key == ord('q'):
                    print("\n[INFO] Cancelled.")
                    # Clean up empty folder if no images were taken
                    if os.path.exists(user_folder_path) and count == 0:
                        try: os.rmdir(user_folder_path)
                        except: pass
                    break 

                if key == ord('s') and not capturing:
                    capturing = True

                # --- SAVE LOGIC ---
                if capturing and face_valid and target_crop is not None:
                    # Create folder only when we actually start saving
                    if count == 0:
                        create_dir(user_folder_path)

                    # Save File
                    file_name = f"{safe_name}_{count}.jpg"
                    save_path = os.path.join(user_folder_path, file_name)
                    
                    try:
                        cv2.imwrite(save_path, target_crop)
                        print(f"📸 Saved: {file_name}")
                        count += 1
                        
                        # Visual Flash Effect
                        white_flash = np.full_like(img, 255)
                        cv2.imshow("NBTI Registrar", white_flash)
                        cv2.waitKey(50) 
                        
                    except Exception as e:
                        print(f"Error saving: {e}")

                    # Stop when done
                    if count >= IMAGES_TO_TAKE:
                        print(f"\n✅ SUCCESS! {count} images saved.")
                        print(f"📂 Location: {user_folder_path}")
                        time.sleep(1) # Let user see the success message
                        break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Ask to continue
            cont = input("\nRegister another staff member? (y/n): ")
            if cont.lower() != 'y':
                break

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main_system()