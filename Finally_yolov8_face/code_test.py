# from ultralytics import YOLO
# from human_gender.Face_info import det_gend
import cv2
import numpy as np
# from face_tilt_verjnakan import face_tilt
# sys.path.insert(0, '/home/quadro/Documents/FaceX-Zoo-main/face_sdk/api_usage/')
from test_widerface_prof import detect_face
import cv2
video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture('/home/art/Downloads/yolov8-face-main/2.mp4')
while True:
        # Read the current frame from the video stream
        ret, Falsh_frame = video_capture.read()
        Falsh_frame=detect_face(Falsh_frame)


        cv2.imshow('Face Detection', Falsh_frame)
                    # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                    # except:
                    #     continue

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()


# import pycuda.autoinit
# import pycuda.driver as drv
# import numpy as np
# from pycuda.compiler import SourceModule

# # Sample data
# data = np.random.rand(100).astype(np.float32)

# # Create a CUDA kernel
# mod = SourceModule("""
#     __global__ void square(float *data)
#     {
#         int idx = threadIdx.x + blockDim.x * blockIdx.x;
#         data[idx] = data[idx] * data[idx];
#     }
# """)

# # Get the kernel function
# square_kernel = mod.get_function("square")

# # Launch the kernel on the GPU
# block_size = 256
# grid_size = (data.size + block_size - 1) // block_size
# square_kernel(drv.InOut(data), block=(block_size, 1, 1), grid=(grid_size, 1))

