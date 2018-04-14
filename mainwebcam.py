import cv2
from picturesManager import save_images, print_menu, resize_images, clean_pictures

arqCasc1 = 'haarcascade_frontalface_default.xml'
arqCasc2 = 'haarcascade_eye.xml'

# Face and eyes classifier
faceCascade1 = cv2.CascadeClassifier(arqCasc1)
faceCascade2 = cv2.CascadeClassifier(arqCasc2)

webcam = cv2.VideoCapture(0)

clean_pictures()

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