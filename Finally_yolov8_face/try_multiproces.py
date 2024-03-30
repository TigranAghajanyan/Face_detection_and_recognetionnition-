import multiprocessing as mp
from multiprocessing import Process
import time
import cv2
# from numba import cuda,jit
import os
import numpy as np
import sys
sys.path.insert(0, '/home/quadro/Documents/FaceX-Zoo-main/face_sdk/api_usage/')
from face_pipline import face_reg

'''
# def calculat():
#     i = 0
#     for i in range(1000000000):
#         i += i
#     return i

# def print_func(continent='Asia'):
#     print("calculat - ",calculat())
#     print('The name of continent is : ', continent)

# if __name__ == "__main__":  # confirms that the code is under main function
#     start = time.time()
#     names = ['America', 'Europe', 'Africa','America', 'Europe', 'Africa']
#     procs = []
#     proc = Process(target=print_func)  # instantiating without any argument
#     procs.append(proc)
#     proc.start()

#     # instantiating process with arguments
    
#     for name in names:
#         # print(name)
#         proc = Process(target=print_func, args=(name,))
#         procs.append(proc)
#         proc.start()
    

#     # for proc in procs:
#     #     proc.join()
#     for i  in range(6):
#         print("calculat - ",calculat())
#     end = time.time()
#     print("--time--",end - start)

# start = time.time()
# calculat()
# end = time.time()
# print("time of one calculat - ",end - start)
'''
# @jit(target ="parallel")
# @numba.jit(nopython=True, nogil=True)
def matching(image1):
    image2 = cv2.imread("/home/quadro/Documents/Finally_yolov8_face/all_ids_help/Woman/ArmSH/12:00:00.007.jpg")
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    combined_image = np.hstack((image2, image1))
    score = face_reg(combined_image)
    print("score :",score)
    return score
# if __name__ == "__main__":  # confirms that the code is under main function
#     my_dict = {}
#     start = time.time()
#     mp.set_start_method("spawn")
#     procs = []
#     proc = Process(target=matching)  # instantiating without any argument
#     procs.append(proc)
#     proc.start()
#     path = "/home/quadro/Documents/Finally_yolov8_face/all_ids_help/"
#     gend_names = os.listdir(path)

#     for ind1, gender in enumerate(gend_names):
#         ind1 *= 100
#         for ind2, human_name in enumerate(os.listdir(path+gender)):
#             image_name = os.listdir(path + gender + "/" + human_name)
#             image_name = path + gender + "/" + human_name + "/" + image_name[0]

#             print("----image_name----",image_name)
#             image1 = cv2.imread(image_name)
#             image2 = cv2.imread("/home/quadro/Documents/Finally_yolov8_face/all_ids_help/Woman/ArmSH/12:00:00.007.jpg")

#             proc = Process(target=matching, args=(image1,image2))
#             my_dict[ind1+ind2] = proc
#             print("=-=-=-=-=-proc :",proc)

#     for proc_ind in my_dict:
#         print("*/*/*/*/* proc :",my_dict[proc_ind])
#         procs.append(my_dict[proc_ind])
#         my_dict[proc_ind].start()
        
#     for proc in procs:
#         proc.join()
        
#     end = time.time()
#     print("--time--",end - start)


######################################################3333333333333333333333333333333333333333333333333333333333333333333

# import threading

# path = "/home/quadro/Documents/Finally_yolov8_face/all_ids_help/"
# gend_names = os.listdir(path)
# threads = []
# start = time.time()
# my_dict = {}
# # for num in range((10)):
# #      my_dict[num] = 0
# for ind1, gender in enumerate(gend_names):
#         ind1 *= 100
#         for ind2, human_name in enumerate(os.listdir(path+gender)):
#             image_name = os.listdir(path + gender + "/" + human_name)
#             image_name = path + gender + "/" + human_name + "/" + image_name[0]

#             print("----image_name----",image_name)
#             image1 = cv2.imread(image_name)
#             image2 = cv2.imread("/home/quadro/Documents/Finally_yolov8_face/all_ids_help/Woman/ArmSH/12:00:00.007.jpg")

#             # image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
#             # combined_image = np.hstack((image2, image1))
#             # score = face_reg(combined_image)
#             # print("score :",score)

#             thread = threading.Thread(target=matching, args=(image1,image2))
#             my_dict[ind1+ind2] = thread
#             print("=-=-=-=-=-thread :",thread)
# for thread_ind in my_dict:
#     print("*/*/*/*/* thread :",my_dict[thread_ind])
#     my_dict[thread_ind].start()
#     threads.append(my_dict[thread_ind])
# print("---my dict ---", my_dict)
# print("---len dict ---",len(my_dict))
# # Wait for all threads to complete
# for thread in threads:
#     thread.join()

# end = time.time()
# print("--time--",end - start)

######################################################3333333333333333333333333333333333333333333333333333333333333333333

import concurrent.futures
import cv2
if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    images = []
    
    path = "/home/quadro/Documents/Finally_yolov8_face/all_ids_help/"
    gend_names = os.listdir(path)
# Create a VideoCapture object
    video_capture = cv2.VideoCapture(0)  # 0 refers to the default camera, you can provide a file path for video files

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        imagesor = []
        if not ret:
            print("Failed to capture frame")
            break


        start = time.time()
        # mp.set_start_method("spawn")
        for ind1, gender in enumerate(gend_names):
                ind1 *= 100
                for ind2, human_name in enumerate(os.listdir(path+gender)):
                    image_name = os.listdir(path + gender + "/" + human_name)
                    image_name = path + gender + "/" + human_name + "/" + image_name[0]

                    print("----image_name----",image_name)
                    image1 = cv2.imread(image_name)
                    images.append(image1)
                    imagesor.append(frame)

        image2 = cv2.imread("/home/quadro/Documents/Finally_yolov8_face/all_ids_help/Woman/ArmSH/12:00:00.007.jpg")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            score = executor.map(matching, (images,imagesor))
        cv2.putText(frame,
                        "score", 
                        (100,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0), 
                        thickness=2)
        cv2.imshow('Video', frame)

        end = time.time()

        print("--time--",end - start)






