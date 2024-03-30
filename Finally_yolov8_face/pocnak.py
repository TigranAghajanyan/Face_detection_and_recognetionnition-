import cv2
import mediapipe as mp
# Initialize the MediaPipe face mesh model
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize the drawing utility to visualize the landmarks
mp_drawing = mp.solutions.drawing_utils
# Read an image or capture a frame from a video stream
image = cv2.imread(r"E:\downloads\for_me_face_rec\for_me\Finally_yolov8_face_for_save\Finally_yolov8_face\all_ids_help\Man\Tigo\12_00_00.007.jpg")
# OR
# Capture a frame from a video stream
# ret, frame = cap.read()
# Read an image or capture a frame from a video stream
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with the MediaPipe face mesh model
results = mp_face_mesh.process(image_rgb)
# Check if landmarks were detected
if results.multi_face_landmarks:
    # Iterate over the detected faces
    for face_landmarks in results.multi_face_landmarks:
        # Draw the landmarks on the image
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION
        )

# Display the image with landmarks
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OR

# Save the image with landmarks
cv2.imwrite("output_image.jpg", image)
