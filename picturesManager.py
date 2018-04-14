import os
import cv2
import glob
from PIL import Image

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

def clean_pictures():
    # Clean older pictures
    files = glob.glob('Pictures/*.png')
    for f in files:
        os.remove(f)

    files = glob.glob('ResizedPictures/*.png')
    for f in files:
        os.remove(f)