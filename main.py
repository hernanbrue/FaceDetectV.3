import dlib
import cv2
import os
import openpyxl
from openpyxl import Workbook
import numpy as np
from datetime import date

# Cargar imágenes de referencia
faces_folder_path = "faces"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Lista de características faciales de las imágenes de referencia
face_descriptors = []
for file_name in os.listdir(faces_folder_path):
    if file_name.endswith(".jpg"):
        image_path = os.path.join(faces_folder_path, file_name)
        image = cv2.imread(image_path)
        faces = detector(image, 1)
        for face in faces:
            shape = sp(image, face)
            face_descriptor = facerec.compute_face_descriptor(image, shape)
            face_descriptors.append(face_descriptor)

# Iniciar la cámara web
video_capture = cv2.VideoCapture(0)

# Crear archivo de Excel
wb = Workbook()
ws = wb.active
ws.title = "Reconocimientos"
ws.append(["Nombre", "Fecha", "Distancia"])

# Identificar caras y comparar con las imágenes de referencia
detected_faces = {}
face_in_list = []
name = "" 
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    for face in faces:
        shape = sp(frame, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)

        # Comparar con las imágenes de referencia
        distances = []
        for i in range(len(face_descriptors)):
            distance = np.linalg.norm(np.array(face_descriptor) - np.array(face_descriptors[i]))
            distances.append(distance)

        # Obtener el nombre de la imagen de referencia más cercana
        min_distance = min(distances)
        if min_distance < 0.5:
            index = distances.index(min_distance)
            name = os.listdir(faces_folder_path)[index].split(".")[0]
            face_in_list.append(name)
        else:
            name = "Desconocido"

        # Mostrar el nombre en la imagen y agregar al archivo de Excel. En rojo el desconocido y en azul el conocido.
        if name == "Desconocido":
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if face_in_list.count(name) <= 1 and name != "Desconocido":
            ws.append([name, date.today(), min_distance])


    # Mostrar la imagen y esperar a que se presione la tecla 'q' para salir
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Guardar el archivo de Excel y liberar

#guarda
wb.save("reconocimientos.xlsx")

#libera
video_capture.release()
cv2.destroyAllWindows()