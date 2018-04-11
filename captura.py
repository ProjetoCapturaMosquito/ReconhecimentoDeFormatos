import cv2
 
arqCasc1 = 'haarcascade_frontalface_default.xml'
arqCasc2 = 'haarcascade_eye.xml'
faceCascade1 = cv2.CascadeClassifier(arqCasc1) #classificador para o rosto
faceCascade2 = cv2.CascadeClassifier(arqCasc2) #classificador para os olhos
 
webcam = cv2.VideoCapture(0)  #instancia o uso da webcam

img_counter = 0

while True:
    s, imagem = webcam.read() #pega efeticamente a imagem da webcam
    imagem = cv2.flip(imagem,180) #espelha a imagem
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
 
    # Desenha um retangulo nas faces e olhos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (0, 255, 0), 4)
 
    for (x, y, w, h) in olhos:
        cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)
 
    cv2.imshow('Video', imagem) #mostra a imagem captura na janela
    
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    elif k & 0xFF == ord('q'):
        #o trecho seguinte e apenas para parar o codigo e fechar a janela
        break
 
webcam.release() #dispensa o uso da webcam
cv2.destroyAllWindows() #fecha todas a janelas abertas