from keras.models import load_model
import numpy as np
import cv2

# Cargar el modelo
model = load_model('keras_model.h5')
# Crear el array de la forma adecuada para alimentar el modelo keras
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Cargar las clases de objetos
labels = list()
file_labels = open('labels.txt', 'r')
for i in file_labels: 
    labels.append(i.split()[1])
camara = cv2.VideoCapture(0)

while True:
    c, frame = camara.read()
    # Voltear la imagen para que coincida con la entrada de entrenamiento
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # redimensionar la imagen a un 224x224 con la misma estrategia que en TM2:
    #image = cv2.resize(frame, (224, 224), fx=0, fy=0, interpolation = cv2.INTER_CUBICq)
    #image = cv2.resize(frame, (224, 224), fx=0, fy=0, interpolation = cv2.INTER_AREA)
    #image = cv2.resize(frame, (224, 224), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    image = cv2.resize(frame, (224, 224), fx=0, fy=0, interpolation = cv2.INTER_AREA)

    # convertir la imagen en un array de numpy
    image_array = np.asarray(image)
    # normalizar la imagen
    data[0] = (image_array.astype(np.float32) / 127.0) - 1
    # realizar reconocimiento de objetos
    prediction = model.predict(data)
    for i in range(len(prediction[0])):
        # Solo mostrar objetos que tengan una precisión a partir del 40 %
        if (prediction[0][i] >= 0.40):
            cv2.putText(frame, str(labels[i]) + ' - prob: ' + str(prediction[0][i]), (20, 20 + (i * 28)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Visualizar por pantalla el video
    cv2.imshow('Video', cv2.resize(frame,(1600, 740), interpolation = cv2.INTER_AREA))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break