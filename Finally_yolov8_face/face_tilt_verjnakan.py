# import cv2
import mediapipe as mp
# from numba import jit,cuda
# import time
# import numpy as np 
# Initialize the MediaPipe face mesh model
# mp_face_detection = mp.solutions.face_detection.FaceDetection()

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# for idneri_path in os.listdir("/home/art/Downloads/yolov8-face-main/all_ids"):
#     for hertakan_id in os.listdir("all_ids/"+idneri_path):  
        # mer_nkarner = "all_ids/"+idneri_path+"/"+hertakan_id
        # image = cv2.imread(mer_nkarner)
# @jit(target_backend='cuda',forceobj=True)
# @jit(target_backend="cuda")
# @cuda.jit
def face_tilt(crop_image):
    try:
        results = mp_face_mesh.process(crop_image)
        demqi_tequtyun = False
        keteri_qanak = 0
        achqeri_cord = []
        qti_cord = [0,0]
        # Check if landmarks were detected
        if results.multi_face_landmarks:
            # Iterate over the detected faces
            for face_landmarks in results.multi_face_landmarks:
                # Get the specific unique landmarks by index
                if demqi_tequtyun == False:
                    landmarks = [
                                face_landmarks.landmark[71],  #qnerak
                                face_landmarks.landmark[301], #qnerak
                                # face_landmarks.landmark[168],  #Nose tip
                                face_landmarks.landmark[4], #qit
                                # face_landmarks.landmark[359],  #Right eye outer corner
                                # face_landmarks.landmark[130],#Right eye inner corner
                                # face_landmarks.landmark[57], #beran
                                # face_landmarks.landmark[287], #beran
                                face_landmarks.landmark[199], #kzak
                                # face_landmarks.landmark[133],  # Left eye outer corner
                                # face_landmarks.landmark[362]
                                ]  
                    # Draw the landmarks on the image
                    hert = 0
                    for landmark in landmarks:
                        if demqi_tequtyun == False:
                            hert += 1
                            if landmark.x > 1 or landmark.x < -1 or landmark.y > 1 or landmark.y < -1:
                                keteri_qanak += 1
                                demqi_tequtyun = True
                                break
                            if hert == 3:
                                qti_cord = (landmark.x,landmark.y)
                            elif hert == 4:
                                color = (255,0,0)
                            else:
                                achqeri_cord.append((landmark.x,landmark.y))
                            # cv2.circle(crop_image, (landmark.x, landmark.y), 2, color, -1)
                    try:
                        # print("QITTTTT",qti_cord)
                        dzax_d = qti_cord[0] - achqeri_cord[0][0]
                        aj_d   = achqeri_cord[1][0] - qti_cord[0]
                        print("dzax_d < 0 ",dzax_d < 0 ,"aj_d < 0 ", aj_d < 0 ," dzax_d/aj_d > 4", dzax_d/aj_d > 4 ," aj_d/dzax_d > 4", aj_d/dzax_d > 4)
                        if dzax_d < 0 or aj_d < 0 or dzax_d/aj_d > 4 or aj_d/dzax_d > 4:
                            demqi_tequtyun = True
                    except:
                        demqi_tequtyun = True
                    # else:
                    #     demqi_tequtyun = False
        else:
            demqi_tequtyun = True
        return demqi_tequtyun
    except:
        demqi_tequtyun = True
        # cv2.imwrite(f'/home/art/Downloads/yolov8-face-main/two_images_landmarks/{time.time()}.jpg',crop_image)
        return demqi_tequtyun


