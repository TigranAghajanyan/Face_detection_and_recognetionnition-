import cv2
import mediapipe as mp
import os
import time
import numpy as np 
# Initialize the MediaPipe face mesh model
mp_face_detection = mp.solutions.face_detection.FaceDetection()

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
start = time.time()
# Initialize the drawing utility to visualize the landmarks
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # Replace 0 with the appropriate video source index if multiple cameras are available

# for idneri_path in os.listdir("/home/art/Downloads/yolov8-face-main/all_ids"):
#     for hertakan_id in os.listdir("all_ids/"+idneri_path):  
while True:

        # mer_nkarner = "all_ids/"+idneri_path+"/"+hertakan_id
        # image = cv2.imread(mer_nkarner)
        # Capture a frame from a video stream
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = frame
        # Convert the image to RGB format
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_face_detection.process(image_rgb)

        # Check if faces were detected
        if results.detections:
            for detection in results.detections:
                # Extract bounding box coordinates
                bbox = detection.location_data.relative_bounding_box
                x1 = abs(int(bbox.xmin * image_rgb.shape[1]))
                y1 = abs(int(bbox.ymin * image_rgb.shape[0]))
                w1 = abs(int(bbox.width * image_rgb.shape[1]))
                h1 = abs(int(bbox.height * image_rgb.shape[0]))

                # Draw bounding box around the face
                crop_image = image_rgb[y1: y1+h1,x1: x1+w1]
                # img_zeros = np.zeros((500,500,3))
                # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # cv2.imshow("Face Landmarks", crop_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                # Process the image with the MediaPipe face mesh model
                results = mp_face_mesh.process(crop_image)
                # results = mp_face_mesh.process(image_rgb)

                keteri_qanak = 0
                achqeri_cord = []
                # Check if landmarks were detected
                if results.multi_face_landmarks:
                    # Iterate over the detected faces
                    for face_landmarks in results.multi_face_landmarks:
                        # Get the specific unique landmarks by index
                        landmarks = [
                                    face_landmarks.landmark[71],  # qnerak
                                    face_landmarks.landmark[301], # qnerak
                                    # face_landmarks.landmark[168],  # Nose tip
                                    face_landmarks.landmark[4], #qit
                                    # face_landmarks.landmark[359],  # Right eye outer corner
                                    # face_landmarks.landmark[130],# Right eye inner corner
                                    # face_landmarks.landmark[57], #beran
                                    # face_landmarks.landmark[287], #beran
                                    face_landmarks.landmark[199], #kzak
                                    # face_landmarks.landmark[133],  # Left eye outer corner
                                    # face_landmarks.landmark[362]
                                    ]  # Left eye inner corner
                        
                        count = 0
                        # Draw the landmarks on the image
                        hert = 0
                        for landmark in landmarks:
                            hert += 1
                            if landmark.x > 1 or landmark.x < -1 or landmark.y > 1 or landmark.y < -1:
                                count += 1
                            x = x1 + int(landmark.x * crop_image.shape[1])
                            y = y1 + int(landmark.y * crop_image.shape[0])
                            color = (0, 255, 0)
                            if hert == 3:
                                qti_cord = (landmark.x,landmark.y)
                                color = (0,0,255)
                            elif hert == 4:
                                color = (255,0,0)
                            else:
                                achqeri_cord.append((landmark.x,landmark.y))
                            cv2.circle(image_rgb, (x, y), 2, color, -1)
                            cv2.circle(image_rgb, (x, y), 2, color, -1)
                    try:
                        dzax_d = qti_cord[0] - achqeri_cord[0][0]
                        aj_d   = achqeri_cord[1][0] - qti_cord[0]
                        if dzax_d < 0 or aj_d < 0 or dzax_d/aj_d > 4 or aj_d/dzax_d > 4:
                            demqi_tequtyun = True
                        else:
                            demqi_tequtyun = False
                    except:
                        demqi_tequtyun = "True2"
                        pass
                    # print(landmarks)
                    cv2.putText(        
                                image_rgb,
                                str(demqi_tequtyun), 
                                (x1,y1),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0,255,0),
                                thickness=2
                                        )

        # Display the image with landmarks
        cv2.imshow("Face Landmarks", image_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
        # cv2.imshow("Face Landmarks", img_zeros)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

