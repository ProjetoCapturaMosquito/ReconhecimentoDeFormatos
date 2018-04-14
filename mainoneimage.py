import os
import cv2
import glob
from picturesManager import save_images, resize_images, clean_pictures
from PIL import Image

clean_pictures()

# Get user supplied values
imagePath = '3x4.png'
cascPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    image,
    minNeighbors=20,
    minSize=(30, 30),
    maxSize=(300,300)
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

save_images(faces, image)
resize_images(faces)

cv2.imshow("Faces found", image)
cv2.waitKey(0)