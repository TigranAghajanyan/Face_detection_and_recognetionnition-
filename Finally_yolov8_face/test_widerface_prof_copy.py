import os
import argparse
from ultralytics import YOLO
# from human_gender.Face_info import det_gend
import cv2
import sys
import numpy as np
import shutil
import datetime
import ast
import time
import glob
from face_tilt_verjnakan import face_tilt
sys.path.insert(0, '/home/quadro/Documents/FaceX-Zoo-main/face_sdk/api_usage/')
from face_pipline import face_reg

# sys.path.insert(0, '/home/quadro/Downloads/Face_info-master/')
# from gender_detection import f_my_gender
# if __name__ == '__main__':
# from numba import jit, cuda

# gender_detector =  f_my_gender.Gender_Model()


def remove_empty_sublists(array):
    non_empty_sublists = [sublist for sublist in array if len(sublist) > 0]
    return non_empty_sublists

def creat_new_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
# @jit(target_backend='cuda',forceobj=True)
def copy_files(old_path,new_path):
    if old_path[-1] != "/":
        old_path += "/"
    if new_path[-1] != "/":
        new_path += "/"
    for imagename in  glob.glob(old_path+"*"):
        shutil.copytree(imagename, new_path+imagename.split("/")[-1])
# @jit(target_backend='cuda',forceobj=True)
def distance_between_the_two_boxes(box1,box2):
    y_mid1 = abs(box1[2]+box1[0])/2
    x_mid1 = abs(box1[3]+box1[1])/2

    y_mid2 = abs(box2[2]+box2[0])/2
    x_mid2 = abs(box2[3]+box2[1])/2 

    dist = (abs(x_mid1 - x_mid2)**2+abs(y_mid1 - y_mid2)**2)**(1/2)
    return dist
# @jit(target_backend='cuda',forceobj=True)
def crop_face(image,cords,with_more = 10):
    x1,x2,y1,y2 = int(cords[1]-with_more),int(cords[3]+with_more),int(cords[0]-with_more),int(cords[2]+with_more)
    if x1 < 0:
        x1 = 0
    elif y1 < 0:
        y1 = 0
    face = image[x1:x2,y1:y2]
    return face
def detect_face():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/quadro/Documents/Finally_yolov8_face/yolov8n-face.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu', help='augmented inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save_folder', default='/home/quadro/Documents/Finally_yolov8_face/widerface_evaluate/widerface_txt', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='/home/quadro/Documents/Finally_yolov8_face/data/widerface/val/imgs', type=str, help='dataset path')
    opt = parser.parse_args()
    # print(opt)

    folder_path = '/home/quadro/Documents/Finally_yolov8_face/all_ids'
    creat_new_folder(folder_path)
    
    folder_path1 = "/home/quadro/Documents/Finally_yolov8_face/two_images_TRUE"
    creat_new_folder(folder_path1)

    folder_path2 = "/home/quadro/Documents/Finally_yolov8_face/two_images_false"
    creat_new_folder(folder_path2)

    folder_path3 = "/home/quadro/Documents/Finally_yolov8_face/two_images_nitrual"
    creat_new_folder(folder_path3)


    source_folder = '/home/quadro/Documents/Finally_yolov8_face/all_ids_help/'
    copy_files(source_folder,folder_path)

    my_all_boxes = [[]]
    all_id = {}
    number_person = 0
    time_dict = {}
    handipats_mardik = []
    model = YOLO(opt.weights)
    out = []
    '''
    # # testing dataset
    # testset_folder = opt.dataset_folder
    # # testset_list = opt.dataset_folder[:-7] + "wider_val.txt"
    # testset_list = "/home/art/Downloads/yolov8-face-main/data/widerface/val/wider_val.txt"
    # with open(testset_list, 'r') as fr:
    #     test_dataset = fr.read().split()
    #     num_images = len(test_dataset)
    # # for img_name in test_dataset:
    # #     image_path = testset_folder + img_name
    # image_path = "/home/art/Downloads/yolov8-face-main/data/widerface/val/imgs/np_file_48598.jpeg"
    # results = model.predict(source=image_path, imgsz=opt.img_size, conf=opt.conf_thres, iou=opt.iou_thres, augment=opt.augment, device=opt.device)
    # img = cv2.imread('/home/art/Downloads/yolov8-face-main/data/widerface/val/imgs/np_file_48598.jpeg')

    # save_name = opt.save_folder + "IMAGE.txt" #img_name[:-4] + ".txt"
    # dirname = os.path.dirname(save_name)
    # if not os.path.isdir(dirname):
    #     os.makedirs(dirname)
    # with open(save_name, "w") as fd:
    #     result = results[0].cpu().numpy()
    #     file_name = os.path.basename(save_name)[:-4] + "\n"
    #     bboxs_num = str(result.boxes.shape[0]) + '\n'
    #     fd.write(file_name)
    #     fd.write(bboxs_num)
    #     for box in result.boxes:
    #         conf = box.conf[0]
    #         cls  = box.cls[0]
    #         xyxy = box.xyxy[0]
    #         x1 = int(xyxy[0] + 0.5)
    #         y1 = int(xyxy[1] + 0.5)
    #         x2 = int(xyxy[2] + 0.5)
    #         y2 = int(xyxy[3] + 0.5)
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    #         fd.write('%d %d %d %d %.03f' % (x1, y1, x2-x1, y2-y1, conf if conf <= 1 else 1) + '\n')
    # cv2.imwrite('/home/art/Downloads/yolov8-face-main/my_path/save1.jpg',img)

    '''
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture('/home/art/Downloads/yolov8-face-main/2.mp4')
    while True:
        # Read the current frame from the video stream
        ret, frame = video_capture.read()
        Falsh_frame=frame.copy()
        # if not ret:
        #     break
        if True:
            faces = model.predict(Falsh_frame)
            result = faces[0].cpu().numpy()
            
            # conf = box.conf[0]
            # cls  = box.cls[0]

            q = 0
            now = datetime.datetime.now()
            # Get the hour, minute, and second from the current time
            hour = now.hour
            minute = now.minute
            second = now.second
            chstacvats = []
            hneric = []
            ms = now.microsecond // 1000
            # new_fram_boxes  = result.boxes
            new_fram_boxes = [box.xyxy[0] for box in result.boxes]

            if new_fram_boxes:
                #fraymnery avelacnuma e my_all_boxes mej
                my_all_boxes.append(new_fram_boxes)
            norutyun = True
            # miayn erkrord framic sksats
            if len(my_all_boxes) > 1:

                # old framy vercnuma naxaverji framy
                old_fram_boxes = my_all_boxes[-2]
                #cikl enq frum verji frami bolor boxery vercnelov ev hamematelov naxkin frami bolor boxeri het
                for ind_box1,box1 in enumerate(new_fram_boxes):
                    if abs(box1[3]-box1[1]) < 18 or abs(box1[2]-box1[0]) < 18:
                        chstacvats.append(f"{box1}")
                        norutyun = False
                    else:
                        print('boxxxxx>>>>>',box1)
                        norutyun = True
                    score = 0
                    for box2 in old_fram_boxes:
                        if norutyun == False:
                            break
                        dist = distance_between_the_two_boxes(box1,box2)
                        print("====distance====",dist)
                        if dist < 48 :
                            try:
                                #ete heravorutyuny poqr e 48 ic nor boxy poxarinum e naxkin boxin vorpes "kay" dictonaryum
                                all_id[str(box1)] = all_id.pop(str(box2))
                                norutyun = False
                                break
                            except:
                                #naxaverji framum mot box chgtneluc heto stugum enq verjin 10ic , 10 baci [-2] framneri het
                                for framner in my_all_boxes:
                                    if norutyun:
                                        for hamar in range(len(framner)-1):
                                            if norutyun:
                                                her = distance_between_the_two_boxes(box1,framner[hamar])
                                                if her < 48:
                                                    print("hnerica")
                                                    try:
                                                        all_id[str(box1)] = all_id.pop(str(framner[hamar]))
                                                        norutyun = False
                                                        break
                                                    except:
                                                        for my_box in all_id:
                                                            print("+++++++++++++++++++++++",my_box)
                                            

                                                            string_representation = my_box
                                                            string_representation = string_representation.strip("[]")  #Remove square brackets
                                                            list_repr = [int(x) for x in string_representation.split()]
                                                            # list_repr = ast.literal_eval(string_representation)
                                                            print(my_box)
                                                            her = distance_between_the_two_boxes(box1,list_repr)
                                                            if her < 13:
                                                                all_id[str(box1)] = all_id.pop(my_box)
                                                                norutyun = False
                                                                break 
                                                        if norutyun:
                                                            chstacvats.append(f"{box1}")
                                                            print("---------3",my_all_boxes[-1][ind_box1])
                                                            try:
                                                                del my_all_boxes[-1][ind_box1]
                                                                my_all_boxes = remove_empty_sublists(my_all_boxes)
                                                            except:
                                                                pass
                                                            norutyun = False
                                                            break
                    if norutyun:
                        image1 = crop_face(Falsh_frame,box1)
                        if face_tilt(image1) == False:
                            print("teq chi *****************")
                            try:
                                gender = det_gend(image1)
                            except:
                                gender = False
                            
                            for gender_path in os.listdir("/home/quadro/Documents/Finally_yolov8_face/all_ids"):
                                if gender:
                                    gender_path = gender
                                for idneri_path in os.listdir("/home/quadro/Documents/Finally_yolov8_face/all_ids/"+gender_path):
                                    for hertakan_id in os.listdir("/home/quadro/Documents/Finally_yolov8_face/all_ids/"+gender_path+"/"+idneri_path):  
                                        if norutyun == False:
                                            break 
                                        q += 1                                
                                        mer_nkarner = f"/home/quadro/Documents/Finally_yolov8_face/all_ids/{gender_path}/{idneri_path}/{hertakan_id}"
                                        image2 = cv2.imread(mer_nkarner)
                                        print(f"{mer_nkarner}>>>>>>>",image2.shape)
                                        try:
                                            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
                                            combined_image = np.hstack((image2, image1))
                                            # Display the resulting frame
                                            score = face_reg(combined_image)
                                            print("score >>>>>>>",score)
                                            if (score)*100 > 45:
                                                if not "Id_" in idneri_path and not gender_path+"/"+idneri_path in handipats_mardik:
                                                    handipats_mardik.append(gender_path+"/"+idneri_path)
                                                    all_id[str(box1)] = [gender_path+"/"+idneri_path]
                                                    cv2.imwrite(f'/home/quadro/Documents/Finally_yolov8_face/two_images_TRUE/{gender_path+"/"+idneri_path}--{score}.jpg',combined_image)
                                                    norutyun = False
                                                    time_dict[f"/home/quadro/Documents/Finally_yolov8_face/all_ids/{gender_path}/{idneri_path}"] = [f'{hour}:{minute}:{second}']
                                                    hneric.append(f"{box1}")
                                                    break
                                                else:
                                                    if not gender_path+"/"+idneri_path in handipats_mardik:
                                                        handipats_mardik.append(gender_path+"/"+idneri_path)
                                                    all_id_copy = all_id.copy()
                                                    for key in all_id_copy:
                                                        print("idnery ==========",all_id[key][0],gender_path+"/"+idneri_path)
                                                        if all_id[key][0] == gender_path+"/"+idneri_path:
                                                            # if idneri_path in stugum:
                                                            #     break
                                                            all_id[str(box1)] = all_id.pop(key)
                                                            cv2.imwrite(f'/home/quadro/Documents/Finally_yolov8_face/two_images_TRUE/{idneri_path}--{score}.jpg',combined_image)
                                                            norutyun = False
                                                            hneric.append(f"{box1}")
                                                            break
                                            elif (score)*100 > 15 and  (score)*100 < 45:
                                                chstacvats.append(f"{box1}")
                                                # del my_all_boxes[-1][ind_box1]
                                                cv2.imwrite(f'/home/quadro/Documents/Finally_yolov8_face/two_images_nitrual/{idneri_path}--{score}.jpg',combined_image)
                                                # norutyun=False
                                                # break
                                            else:
                                                print("================00000000000000",score)
                                                cv2.imwrite(f'/home/quadro/Documents/Finally_yolov8_face/two_images_false/{idneri_path}--{score}.jpg',combined_image)

                                        except:
                                            chstacvats.append(f"{box1}")
                                            try:
                                                del my_all_boxes[-1][ind_box1]
                                                my_all_boxes = remove_empty_sublists(my_all_boxes)
                                            except:
                                                pass
                                            # norutyun = False
                                if gender:
                                    break
                        else:
                            norutyun = False
                            try:
                                cv2.imwrite('/home/quadro/Documents/Finally_yolov8_face/two_images_TRUE/1111111111.jpg',Falsh_frame[ int(box1[1])-10:int(box1[3])+10,int(box1[0])-10:int(box1[2])+10])
                            except:
                                pass
                            print("=============================%",score*100)
                    ####################### 
                    if norutyun:
                        if not f"{box1}" in chstacvats or f"{box1}" in hneric:
                            try:
                                gender = det_gend(image1)
                            except:
                                gender = "unknown"
                            number_person += 1
                            print("avelacav nor id")
                            all_id[str(box1)] = [f"{gender}/Id_{number_person}"] 
                        # norutyun = False
            if len(my_all_boxes) == 10:             
                my_all_boxes.pop(0)
            # for box in result.boxes:
            if len(new_fram_boxes):
                for box in my_all_boxes[-1]:
                    label = ""
                    # conf = round(box.conf[0]*100,2)
                    # box = box.xyxy[0]
                    # try:
                    print("--------------",box)
                    print("555555 chstacvatsner",chstacvats)
                    print("all_id",all_id)
                    if not f"{box}" in chstacvats or f"{box}" in hneric:
                        try:
                            folder_path = f"/home/quadro/Documents/Finally_yolov8_face/all_ids/{all_id[str(box)][0]}"
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)
                            folder_contents = os.listdir(folder_path)
                            # Calculate the length of the folder
                            folder_length = len(folder_contents)
                            if 0 < folder_length and folder_length < 2:
                                if folder_contents[-1].split(".jpg")[0].split(".")[0] < f'{hour}:{minute}:{second-4}':
                                    face_img = crop_face(Falsh_frame,box)
                                    if time_dict[folder_path][0] < f'{hour}:{minute}:{second-4}':
                                        if face_tilt(face_img) == False:
                                            cv2.imwrite(f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg',face_img)
                                            time_dict[folder_path] = [f'{hour}:{minute}:{second}']
                                    print("pahpanuma bolor boxery",f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg')

                            elif folder_length == 0:
                                face_img = crop_face(Falsh_frame,box)
                                cv2.imwrite(f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg',face_img)
                                time_dict[folder_path] = [f'{hour}:{minute}:{second}']
                                print("pahpanuma araji nkary n_id_um",f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg')
                            my_id = all_id[str(box)]
                            label = " "+my_id[0] # +f" ?"
                        except:
                            pass
                            
                    cv2.putText(Falsh_frame,
                                label, 
                                ((int(box[0])+int(box[2]))//2-70,(int(box[1]) + int(box[3]))//2-50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,0), 
                                thickness=2)
                # cv2.imwrite("/home/quadro/Documents/Finally_yolov8_face/two_images_landmarks/video.jpg",Falsh_frame)
            cv2.imshow('Face Detection', Falsh_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # except:
            #     continue

    # Release the video capture
    video_capture.release()
    cv2.destroyAllWindows()

detect_face()

