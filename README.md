# Name of Project - "Text Binarize Defender in Watermark Images Project"

This project combines multiple techniques to detect, remove background, and binarize watermark text from images. It utilizes YOLOv8 for watermark text detection, followed by background removal using image processing techniques, and finally, text binarization using OTSU and wavelet transform methods.

Features Watermark Text Detection: Utilizes YOLOv8 to accurately detect watermark text in images. Background Removal: Removes the background from the detected watermark text region to isolate it. Text Binarization: Applies OTSU and wavelet transform techniques to binarize the watermark text, enhancing its visibility and usability.
#Code of this project


# WATERMARK TEXT DETECTION BY YOLOV8s - 1st TECHNIQUE

!nvidia-smi

#Pip install method

!pip install ultralytics==8.0.181

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/YOLOv8s

!ls

%cd /content/drive/MyDrive/YOLOv8s

!yolo task=detect mode=train model=yolov8s.pt data= data.yaml epochs=100 imgsz=640 plots=True

!ls runs/detect/train24/

Image(filename='/content/drive/MyDrive/YOLOv8s/runs/detect/train24/confusion_matrix.png' , width=500)

Image(filename='runs/detect/train24/results.png' , width=900)

Image(filename='runs/detect/train24/val_batch0_pred.jpg' , width=600)

#Validate Custom Model
!yolo task=detect mode=val model=runs/detect/train24/weights/best.pt data=data.yaml

#Inference with Custom Model
!yolo task=detect mode=predict model=/content/drive/MyDrive/YOLOv8s/runs/detect/train24/weights/best.pt conf=0.25 source=/content/drive/MyDrive/YOLOv8s/Major/test/images

#Text Detection on unseen images
!yolo task=detect mode=predict model=/content/drive/MyDrive/YOLOv8s/runs/detect/train24/weights/best.pt conf=0.25 source=/content/drive/MyDrive/Major_project_dataset/images/train

# BACKGROUND REMOVAL - 2nd TECHNIQUE

!pip install rembg

!pip install pillow

from rembg import remove
from PIL import Image
import os

#Path to the folder containing the images
predict_folder = "/content/drive/MyDrive/YOLOv8s/runs/detect/predict"

#Path to the folder where you want to save the result
output_folder = "/content/drive/MyDrive/Background_removal"

#Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

#Iterate through each file in the predict folder
for filename in os.listdir(predict_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(predict_folder, filename)
        img = Image.open(img_path)

        # Perform background removal
        R = remove(img)

        # Convert the image to RGB mode before saving as JPEG
        R = R.convert("RGB")

        # Save the result in the output folder
        output_path = os.path.join(output_folder, f"removed_{filename.replace('.png', '.jpg')}")
        R.save(output_path)

print("Background removal and saving completed.")


#Backgorund removal of unseen test images

from rembg import remove
from PIL import Image
import os

#Path to the folder containing the images
predict_folder = "/content/drive/MyDrive/YOLOv8s/runs/detect/predict18"

#Path to the folder where you want to save the results
output_folder = "/content/drive/MyDrive/background_unseen"

#Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

#Iterate through each file in the predict folder
for filename in os.listdir(predict_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(predict_folder, filename)
        img = Image.open(img_path)

        # Perform background removal
        R = remove(img)

        # Convert the image to RGB mode before saving as JPEG
        R = R.convert("RGB")

        # Save the result in the output folder
        output_path = os.path.join(output_folder, f"removed_{filename.replace('.png', '.jpg')}")
        R.save(output_path)

print("Background removal and saving completed.")


# WATERMARK TEXT BINARIZATION BY WAVELET TRANSFORM AND OTSU - 3rd TECHNIQUE 

#Convert Extracted Image Into Binarization By Wavelet Transform

!pip install PyWavelets

import cv2
import os
import numpy as np
import pywt

#Path to the folder containing watermark images
input_folder = "/content/drive/MyDrive/Background_removal"

#Output folder for wavelet transformed images
output_folder = "/content/drive/MyDrive/Wavelet_Transform_Images"

#Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#List all files in the input folder
image_files = os.listdir(input_folder)

#Loop through the watermark image files and apply wavelet transform
for image_file in image_files:
    # Create the full path for the input watermark image
    input_image_path = os.path.join(input_folder, image_file)

    # Read the input watermark image in grayscale mode
    watermark_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Perform wavelet transform
    coeffs = pywt.dwt2(watermark_image, 'bior1.3')
    cA, (cH, cV, cD) = coeffs

    # Thresholding to remove noise or retain important features
    threshold = 30  # You may need to adjust this threshold based on your image
    cA_thresholded = pywt.threshold(cA, threshold, mode='soft')
    cH_thresholded = pywt.threshold(cH, threshold, mode='soft')
    cV_thresholded = pywt.threshold(cV, threshold, mode='soft')
    cD_thresholded = pywt.threshold(cD, threshold, mode='soft')

    # Reconstruct the image from the thresholded coefficients
    reconstructed_image = pywt.idwt2((cA_thresholded, (cH_thresholded, cV_thresholded, cD_thresholded)), 'bior1.3')

    # Convert the reconstructed image to binary format
    _, wavelet_binary_image = cv2.threshold(reconstructed_image, 128, 255, cv2.THRESH_BINARY)

    # Create the full path for the output wavelet transformed image
    output_image_path = os.path.join(output_folder, f"wavelet_transformed_{image_file}")

    # Save the wavelet transformed image
    cv2.imwrite(output_image_path, wavelet_binary_image)

print("Wavelet transformation of watermark images completed.")


#Wavelet Tranformation of unseen test images

import cv2
import os
import numpy as np
import pywt

#Path to the folder containing watermark images
input_folder = "/content/drive/MyDrive/background_unseen"

#Output folder for wavelet transformed images
output_folder = "/content/drive/MyDrive/wavelet_unseen"

#Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#List all files in the input folder
image_files = os.listdir(input_folder)

#Loop through the watermark image files and apply wavelet transform
for image_file in image_files:
    # Create the full path for the input watermark image
    input_image_path = os.path.join(input_folder, image_file)

    # Read the input watermark image in grayscale mode
    watermark_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Perform wavelet transform
    coeffs = pywt.dwt2(watermark_image, 'bior1.3')
    cA, (cH, cV, cD) = coeffs

    # Thresholding to remove noise or retain important features
    threshold = 30  # You may need to adjust this threshold based on your image
    cA_thresholded = pywt.threshold(cA, threshold, mode='soft')
    cH_thresholded = pywt.threshold(cH, threshold, mode='soft')
    cV_thresholded = pywt.threshold(cV, threshold, mode='soft')
    cD_thresholded = pywt.threshold(cD, threshold, mode='soft')

    # Reconstruct the image from the thresholded coefficients
    reconstructed_image = pywt.idwt2((cA_thresholded, (cH_thresholded, cV_thresholded, cD_thresholded)), 'bior1.3')

    # Convert the reconstructed image to binary format
    _, wavelet_binary_image = cv2.threshold(reconstructed_image, 128, 255, cv2.THRESH_BINARY)

    # Create the full path for the output wavelet transformed image
    output_image_path = os.path.join(output_folder, f"wavelet_transformed_{image_file}")

    # Save the wavelet transformed image
    cv2.imwrite(output_image_path, wavelet_binary_image)

print("Wavelet transformation of watermark images completed.")


#OTSU Binarization

pip install opencv-python

import cv2
import glob
import os

#Path to the folder containing watermark images
input_folder = "/content/drive/MyDrive/Background_removal"

#Output folder for binarized images
output_folder = "/content/drive/MyDrive/OTSU_Binarization"

#Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#List all files in the input folder
image_files = glob.glob(os.path.join(input_folder, "*.jpg"))

#Loop through the image files and convert each to binary format using Otsu's thresholding
for image_file in image_files:
    # Create the full path for the input image
    input_image_path = os.path.join(input_folder, image_file)

    # Read the input image in grayscale mode
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Otsu's thresholding to binarize the image
    _, binary_image = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create the full path for the output binary image
    output_image_path = os.path.join(output_folder, f"binarized_{image_file}")

    # Save the binary image
    cv2.imwrite(output_image_path, binary_image)

print("Extracted Image Binarization completed.")


#OTSU Binarization on Unseen Images

import cv2
import os

#Path to the folder containing watermark images
input_folder = "/content/drive/MyDrive/background_unseen"

#Output folder for binarized images
output_folder = "/content/drive/MyDrive/otsu_unseen_new"

#Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#List all files in the input folder
image_files = os.listdir(input_folder)

#Loop through the image files and convert each to binary format using Otsu's thresholding
for image_file in image_files:
    # Create the full path for the input image
    input_image_path = os.path.join(input_folder, image_file)

    # Read the input image in grayscale mode
    original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Otsu's thresholding to binarize the image
    _, binary_image = cv2.threshold(original_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create the full path for the output binary image
    output_image_path = os.path.join(output_folder, f"binarized_{image_file}")

    # Save the binary image
    cv2.imwrite(output_image_path, binary_image)

print("Extracted Image Binarization completed.")
















