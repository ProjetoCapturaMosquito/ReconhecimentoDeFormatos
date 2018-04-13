import os
import cv2
import glob
from PIL import Image
from subprocess import call

def print_menu():
    print("Press one of the following commands")
    print("  c - crop and save rectangles")
    print("  r - resize cropped rectangles")
    print("  Space - save entire picture")
    print("  q - quit")

# Saves rectangles from entire picture 
def save_images(faces, imagem):
    img_counter = 0
    for (x, y, w, h) in faces:
            cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 4)
            crop_rectangle = imagem[y: y+h, x: x+w].copy()
            img_counter += 1
            img_name = "face_{}.png".format(img_counter)
            cv2.imwrite("./Pictures/" + img_name, crop_rectangle)
            print("{} written on Pictures folder!".format(img_name))

# Resize to one specific size rectangles found
def resize_images(faces):
    img_counter = 0
    # Choose size
    img_width = 100
    img_height = 200
    filelist=os.listdir('Pictures')
    for foto in filelist[:]: # filelist[:] makes a copy of filelist.
        if foto.endswith(".png"):
            img_counter += 1
            resize_name = "resize_{}.png".format(img_counter)
            img_name = "face_{}.png".format(img_counter)
            img = Image.open("./Pictures/" + img_name)
            img = img.resize((img_width, img_height), Image.ANTIALIAS)
            img.save('./ResizedPictures/' + resize_name, 'PNG')
            print("{} written on Resized folder!".format(resize_name))
    print("All images resized!")

arqCasc1 = 'haarcascade_frontalface_default.xml'
arqCasc2 = 'haarcascade_eye.xml'
# Face classifier
faceCascade1 = cv2.CascadeClassifier(arqCasc1)
# Eyes classifier
faceCascade2 = cv2.CascadeClassifier(arqCasc2)
webcam = cv2.VideoCapture(0)

# Clean older pictures
files = glob.glob('Pictures/*.png')
for f in files:
    os.remove(f)

files = glob.glob('ResizedPictures/*.png')
for f in files:
    os.remove(f)

print_menu()

while True:
    img_counter = 0
    s, imagem = webcam.read()
    imagem = cv2.flip(imagem,180)
    ret, frame = webcam.read()

    faces = faceCascade1.detectMultiScale(
        imagem,
        minNeighbors=20,
        minSize=(30, 30),
        maxSize=(300,300)
    )

    olhos = faceCascade2.detectMultiScale(
        imagem,
        minNeighbors=20,
        minSize=(10, 10),
        maxSize=(90,90)
    )

    # Draw rectangle on faces and eyes detected
    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 4)
    for (x, y, w, h) in olhos:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Shows picture on screen
    cv2.imshow('Video', imagem)
    
    k = cv2.waitKey(1)

    if k & 0xFF == ord('q'):
        # Close program
        print("Closing...")
        break
    elif k%256 == 32:
        # Save picture hitting space
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
    elif k & 0xFF == ord('c'):
        # crop faces
        save_images(faces, imagem)
    elif k & 0xFF == ord('r'):
        # Resize pictures
        resize_images(faces)

webcam.release()
cv2.destroyAllWindows()