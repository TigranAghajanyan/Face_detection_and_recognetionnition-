import os
import argparse
from ultralytics import YOLO
import cv2
import sys
import numpy as np
sys.path.append('.')
import shutil
import datetime
sys.path.insert(1, '/home/quadro/Downloads/FaceX-Zoo-main/face_sdk/api_usage/')
# from face_pipline import face_reg
# export PYTHONPATH=/'/home/quadro/Downloads/FaceX-Zoo-main/face_sdk/api_usage'/to/face_pipline
# if __name__ == '__main__':
def detect_face():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/quadro/Documents/Finally_yolov8_face/yolov8n-face.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', type=str, default='cpu', help='augmented inference')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save_folder', default='/home/quadro/Downloads/yolov8-face-main/widerface_evaluate/widerface_txt', type=str, help='Dir to save txt results')
    parser.add_argument('--dataset_folder', default='/home/quadro/Downloads/yolov8-face-main/data/widerface/val/imgs', type=str, help='dataset path')
    opt = parser.parse_args()
    # print(opt)

    folder_path = '/home/quadro/Documents/Finally_yolov8_face/all_ids'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    folder_path1 = "/home/quadro/Documents/Finally_yolov8_face/all_ids/two_images_TRUE"
    if os.path.exists(folder_path1):
        shutil.rmtree(folder_path1)
    os.makedirs(folder_path1)
    folder_path2 = "/home/quadro/Documents/Finally_yolov8_face/all_ids/two_images_false"
    if os.path.exists(folder_path2):
        shutil.rmtree(folder_path2)
    os.makedirs(folder_path2)
    folder_path3 = "/home/quadro/Documents/Finally_yolov8_face/all_ids/two_images_nitrual"
    if os.path.exists(folder_path3):
        shutil.rmtree(folder_path3)
    os.makedirs(folder_path3)
    my_all_boxes = []
    all_id = {}
    number_person = 0
    time_dict = {}

    model = YOLO(opt.weights)
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
    video_capture = cv2.VideoCapture(9)
    # video_capture = cv2.VideoCapture('/home/art/Downloads/yolov8-face-main/2.mp4')


    while True:
            # Read the current frame from the video stream
            ret, frame = video_capture.read()
            Falsh_frame=frame.copy()

            # Perform face detection using YOLOv3
            faces = model.predict(frame)
            result = faces[0].cpu().numpy()
            radius = 2
            # Define the color of the circle (blue in this case)
            color = (255, 0, 0)

            # conf = box.conf[0]
            # cls  = box.cls[0]

            q = 0
            stugum = []
            now = datetime.datetime.now()
            # Get the hour, minute, and second from the current time
            hour = now.hour
            minute = now.minute
            second = now.second
            chstacvats = []
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
                    print('boxxxxx>>>>>',box1)
                    # box1 = box1.xyxy[0]
                    #stanum enq x mijin ev y mijiny nor frami hertakan boxi hamar
                    norutyun = True
                    # print("-------",box1)
                    y_mid1 = abs(box1[2]+box1[0])/2
                    x_mid1 = abs(box1[3]+box1[1])/2
                    dist = 0
                    score = 0
                    for box2 in old_fram_boxes:
                        # if norutyun == False:
                        #     break
                        #stanum enq x mijin ev y mijiny naxkin frami hertakan boxi hamar
                        y_mid2 = abs(box2[2]+box2[0])/2
                        x_mid2 = abs(box2[3]+box2[1])/2 
                        #hashvum enq hertakan boxeri mijev heravorutyun
                        dist = (abs(x_mid1 - x_mid2)**2+abs(y_mid1 - y_mid2)**2)**(1/2)
                        print("==distance===",dist)
                        if dist < 48 :
                            try:
                                #ete heravorutyuny poqr e 48 ic nor boxy poxarinum e naxkin boxin vorpes "kay" dictonaryum
                                all_id[str(box1)] = all_id.pop(str(box2))
                                norutyun = False
                                break
                            except:
                                # #naxaverji framum mot box chgtneluc heto stugum enq verjin 10ic , 10 baci [-2] framneri het
                                for framner in my_all_boxes:
                                    if norutyun:
                                        for hamar in range(len(framner)-1):
                                            if norutyun:
                                                # box2 = box2.xyxy[0]
                                                y_mid2= abs(framner[hamar][2]+framner[hamar][0])/2
                                                x_mid2 = abs(framner[hamar][3]+framner[hamar][1])/2 
                                                her = (abs(x_mid1 - x_mid2)**2+abs(y_mid1 - y_mid2)**2)**(1/2)
                                                if her < 48:
                                                    print("hnerica")
                                                    try:
                                                        all_id[str(box1)] = all_id.pop(str(framner[hamar]))
                                                        norutyun = False
                                                        break
                                                    except:
                                                        chstacvats.append(f"{box1}")
                                                        del my_all_boxes[-1][ind_box1]
                                                        norutyun = False
                                                        break
                    if norutyun:
                        for idneri_path in os.listdir("all_ids"):
                            for hertakan_id in os.listdir("all_ids/"+idneri_path):  
                                if norutyun == False:
                                    break 
                                q += 1
                                # print("===================",int(box1[1]-40),int(box1[3]+40),int(box1[0]-40),int(box1[2]+40))
                                x1,x2,y1,y2 = int(box1[1]-20),int(box1[3]+20),int(box1[0]-20),int(box1[2]+20)
                                if x1 < 0:
                                    x1 = 0
                                elif y1 < 0:
                                    y1 = 0
                                image1 = frame[x1:x2,y1:y2]
                                mer_nkarner = "all_ids/"+idneri_path+"/"+hertakan_id
                                image2 = cv2.imread(mer_nkarner)
                                print(f"{mer_nkarner}>>>>>>>",image2.shape)
                                try:
                                    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
                                    combined_image = np.hstack((image2, image1))
                                    # Display the resulting frame
                                    score = face_reg(combined_image)
                                    if (score)*100 > 40:
                                        all_id_copy = all_id.copy()
                                        for key in all_id_copy:
                                            print("idnery ==========",all_id[key][0],idneri_path)
                                            if all_id[key][0] == idneri_path:
                                                # if idneri_path in stugum:
                                                #     break
                                                all_id[str(box1)] = all_id.pop(key)
                                                cv2.imwrite(f'/home/quadro/Documents/Finally_yolov8_face/all_ids/two_images_TRUE/{idneri_path}--{score}.jpg',combined_image)
                                                norutyun = False
                                                break
                                    elif (score)*100 > 15 and  (score)*100 < 40:
                                        # chstacvats.append(f"{box1}")
                                        # del my_all_boxes[-1][ind_box1]
                                        cv2.imwrite(f'/home/art/Downloads/yolov8-face-main/two_images_nitrual/{idneri_path}--{score}.jpg',combined_image)
                                        norutyun=False
                                        # break
                                    else:
                                        print("================00000000000000",score)
                                        cv2.imwrite(f'/home/quadro/Documents/Finally_yolov8_face/all_ids/two_images_false/{idneri_path}--{score}.jpg',combined_image)

                                except:
                                    chstacvats.append(f"{box1}")
                                    del my_all_boxes[-1][ind_box1]
                                    norutyun = False
                                print("=============================%",score*100)
                    ####################### 
                    if norutyun:
                        number_person += 1
                        print("avelacav nor id")
                        all_id[str(box1)] = [f"Id_{number_person}"] 
                        # norutyun = False
            # miayn erb stanum enq  araji frami boxery
            else: 
                if len(my_all_boxes) == 1:
                    # print("arajin ID",new_fram_boxes)
                    for ind_box1,boxs in enumerate(new_fram_boxes):
                        # boxs = boxs.xyxy[0]
                        x1,x2,y1,y2 = int(boxs[1]-20),int(boxs[3]+20),int(boxs[0]-20),int(boxs[2]+20)
                        if x1 < 0:
                            x1 = 0
                        elif y1 < 0:
                            y1 = 0
                        image_1 = frame[x1:x2,y1:y2]
                        combined_image = np.hstack((image_1, image_1))
                        try:

                            score = face_reg(combined_image)
                            print("avelacav id amenaskzbum")
                            cv2.imwrite(f'/home/quadro/Documents/Finally_yolov8_face/all_ids/two_images_TRUE/{number_person + 1}-{ind_box1}-{score}.jpg',combined_image)
                            number_person += 1  
                            all_id[str(boxs)] = [f"Id_{number_person}"]
                        except:
                            chstacvats.append(f"{boxs}")
                            my_all_boxes.pop(0)
                            break
                            # pass

            if len(my_all_boxes) == 10:             
                my_all_boxes.pop(0)
            for box in result.boxes:
                conf = round(box.conf[0]*100,2)
                box = box.xyxy[0]
                # try:
                print("--------------",box)
                print("555555 chstacvatsner",chstacvats)
                if not f"{box}" in chstacvats:
                    try:
                        folder_path = f"all_ids/{all_id[str(box)][0]}"
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        folder_contents = os.listdir(folder_path)
                        # Calculate the length of the folder
                        folder_length = len(folder_contents)
                        if 0 < folder_length and folder_length < 5:
                            if folder_contents[-1].split(".jpg")[0].split(".")[0] < f'{hour}:{minute}:{second-4}':
                                x1,x2,y1,y2 = int(box[1]-20),int(box[3]+20),int(box[0]-20),int(box[2]+20)
                                if x1 < 0:
                                    x1 = 0
                                elif y1 < 0:
                                    y1 = 0
                                face_img = frame[x1:x2,y1:y2]
                                if time_dict[folder_path][0] < f'{hour}:{minute}:{second-4}':
                                    cv2.imwrite(f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg',face_img)
                                    time_dict[folder_path] = [f'{hour}:{minute}:{second}']
                                print("pahpanuma bolor boxery",f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg')

                        elif folder_length == 0:
                            x1,x2,y1,y2 = int(box[1]-20),int(box[3]+20),int(box[0]-20),int(box[2]+20)
                            if x1 < 0:
                                x1 = 0
                            elif y1 < 0:
                                y1 = 0
                            # face_img = frame[int(box[1]-40) : int(box[3]+40),int(box[0]-40): int(box[2]+40)]
                            face_img = frame[x1:x2,y1:y2]
                            print("xyxy",x1,x2,y1,y2)
                            # print("Face_img",x1, int(box[3]+40),int(box[0]-40), int(box[2]+40))
                            cv2.imwrite(f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg',face_img)
                            time_dict[folder_path] = [f'{hour}:{minute}:{second}']
                            print("pahpanuma araji nkary n_id_um",f'{folder_path}/{hour}:{minute}:{second}.{ms:03d}.jpg')

                        my_id = all_id[str(box)]
                        label = " "+my_id[0]
                        cv2.putText(Falsh_frame,
                                    label+f" {conf}%", 
                                    ((int(box[0])+int(box[2]))//2-70,(int(box[1]) + int(box[3]))//2-50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0,
                                    (0,255,0),
                                    thickness=2)
                    except:
                        pass
            print("================================================",my_all_boxes)
            '''

            # center = ((x1+x2)//2,(y1+y2)//2)
            # cv2.circle(frame, center, radius, color, thickness=2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            # print(frame)
            # image1 = frame[y1:y2,x1:x2]
            # # print(image1)
            # path = "/home/art/Downloads/yolov8-face-main/photos/"
            # for mer_nkarner in os.listdir(path):
            #     image2 = cv2.imread(path + mer_nkarner)
            #     image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
            #     combined_image = np.hstack((image2, image1))
            #     # Display the resulting frame
            #     try:
            #         if face_reg(combined_image) > 0.25:
            #             boolean = True
            #         else:
            #             boolean = False
            #     except:
            #         boolean = "unknown"
            #         pass
            #     if boolean == True:
            #         name = mer_nkarner.split(".jpg")[0]
            #         boolean = f"{name}"
            #         break
            # org = (x1, y1) # Coordinates of the bottom-left corner of the text
            # fontFace = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            # fontScale = 1.5  # Font scale
            # color = (255, 0, 0)  # BGR color (blue)
            # thickness = 2 
            # cv2.putText(frame, f"{boolean}", org, fontFace, fontScale, color, thickness)
            '''
            cv2.imshow('Face Detection', Falsh_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture
    video_capture.release()
    cv2.destroyAllWindows()

detect_face()

