import cv2



cap = cv2.VideoCapture(0) # трансляция с камеры
cap.set(3, 500) #ширина картинки c камеры
cap.set(4, 300) #длинна картинки с камеры

faces = cv2.CascadeClassifier('face.xml')

while True:  #
    success, img = cap.read()
    img = cv2.flip(img, 1)
    results = faces.detectMultiScale(img, scaleFactor=1.104, minNeighbors=5)
    for (x, y, w, h) in results:
        cv2.circle(img, (x + (w // 2), y + (h // 2)), h // 2, (0, 255, 0), thickness=2)

    cv2.imshow('Result', img)

    # if cv2.waitKey(1) == ord('q'):
    #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


