# import sys
# import cv2
# import time
# import os
# import numpy as np
# sys.path.insert(0, '/home/quadro/Documents/FaceX-Zoo-main/face_sdk/api_usage/')
# from face_pipline import face_reg
# start = time.time()
# for hertakan_id in os.listdir("/home/quadro/Documents/Finally_yolov8_face/PHOTOS"):  
#     try:
#         mer_nkarner = "/home/quadro/Documents/Finally_yolov8_face/PHOTOS/"+hertakan_id
#         image1 = cv2.imread(mer_nkarner)
#         image2 = cv2.imread(mer_nkarner)
#         image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

#         combined_image = np.hstack((image2, image1))
#         # Display the resulting frame
#         score = face_reg(combined_image)
#         print("name - ",hertakan_id)
#         print("score  -  ",score)
#     except:
#         continue
# end = time.time()
# print("time - ", end- start)
import ast

string_representation = "[486 267 627 461]"
list_representation = ast.literal_eval(string_representation)

print(type(string_representation))
