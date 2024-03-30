# from PIL import Image

# def combine_images(image2_path, output_path):
#     image1_path = '/home/art/Downloads/yolov8-face-main/2023-06-27-19-35-41.jpg'

#     # Open the first image
#     image1 = Image.open(image1_path)

#     # Open the second image
#     image2 = Image.open(image2_path)

#     # Resize the second image to match the size of the first image
#     image2 = image2.resize(image1.size)

#     # Create a new image with the same size as the first image
#     combined_image = Image.new('RGB', (image1.width * 2, image1.height))

#     # Paste the first image on the left side
#     combined_image.paste(image1, (0, 0))

#     # Paste the second image on the right side
#     combined_image.paste(image2, (image1.width, 0))

#     # Save the combined image
#     combined_image.save(output_path)
#     return output_path
# Example usage
# image1_path = '/home/art/Downloads/yolov8-face-main/2023-06-27-19-35-41.jpg'
# image2_path = '/home/art/Desktop/Photo/all_ids/Id_4/18:37:22.766.jpg'
# output_path = '/home/art/Desktop/Photo/Argine_Lian1.jpg'

# combine_images(image1_path, image2_path, output_path)
import cv2
import numpy as np
image1 = cv2.imread('/home/art/Downloads/yolov8-face-main/2023-06-27-19-35-41.jpg')
image2 = cv2.imread('/home/art/Downloads/yolov8-face-main/2023-06-27-19-35-41.jpg')
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
combined_image = np.hstack((image2, image1))
cv2.imshow('Composite Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
