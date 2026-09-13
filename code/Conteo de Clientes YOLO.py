# %% [markdown]
# ##1. Libraries
!pip install ultralytics -q
import cv2
import numpy as np
from ultralytics import YOLO
from google.colab import drive
from google.colab.patches import cv2_imshow

drive.mount('/content/drive')
video_path = path_video = "/content/drive/MyDrive/Data/Curso Computer Vision/people-detection.mp4" 
output_path = "/content/drive/MyDrive/Data/Curso Computer Vision/output_video.avi"


def signed_distance(point, line):
    """
    Calcula la distancia firmada de un punto a una línea definida por dos puntos.
    Permite saber de qué lado de la línea se encuentra el punto.
    point: (x, y)
    line: ((x1, y1), (x2, y2))
    """
    x, y = point
    (x1, y1), (x2, y2) = line
    num = (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / den if den != 0 else 0

# Definir las líneas de conteo
line1 = ((130, 120), (25, 300)) #((130, 180), (25, 300))
line2 = ((650, 175), (720, 275))

# Contadores para cada línea
count_line1 = 0
count_line2 = 0

# Umbral para asociar detecciones entre frames
distance_threshold = 25 #50

# Lista para almacenar los centroides del frame anterior
prev_centroids = []

# Cargar el modelo YOLO (se asume que "person" es la clase 0 en COCO)
model = YOLO("yolo11n.pt")

# Abrir video de entrada y configurar VideoWriter para el video de salida
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("No se pudo abrir el video de entrada.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección con YOLOv11
    results = model(frame, conf=0.7)
    # Accedemos al primer resultado de la lista
    boxes_obj = results[0].boxes

    current_centroids = []
    if boxes_obj is not None and len(boxes_obj) > 0:
        # Extraer las cajas y las clases como arrays NumPy
        bboxes = boxes_obj.xyxy.cpu().numpy()  # Array de shape (N,4)
        classes = boxes_obj.cls.cpu().numpy()    # Array de shape (N,)
        # Filtrar detecciones de "person" comprobando que la clase sea 0
        for i in range(len(bboxes)):
            if int(classes[i]) == 0:
                x1, y1, x2, y2 = map(int, bboxes[i])
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_centroids.append(centroid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

    # Dibujar las líneas de conteo
    cv2.line(frame, line1[0], line1[1], (255, 0, 0), 2)
    cv2.line(frame, line2[0], line2[1], (0, 0, 255), 2)

    # Comparar cada centroide actual con los del frame anterior para detectar cruces
    for curr in current_centroids:
        best_distance = float('inf')
        best_prev = None
        for prev in prev_centroids:
            d = np.linalg.norm(np.array(curr) - np.array(prev))
            if d < best_distance and d < distance_threshold:
                best_distance = d
                best_prev = prev
        if best_prev is not None:
            # Verificar cruce de la línea 1
            prev_side1 = signed_distance(best_prev, line1)
            curr_side1 = signed_distance(curr, line1)
            if prev_side1 * curr_side1 < 0:
                count_line1 += 1
            # Verificar cruce de la línea 2
            prev_side2 = signed_distance(best_prev, line2)
            curr_side2 = signed_distance(curr, line2)
            if prev_side2 * curr_side2 < 0:
                count_line2 += 1

    # Dibujar los contadores en el frame
    cv2.putText(frame, f"Seccion Futbol: {count_line1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Seccion Tenis: {count_line2}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Escribir el frame procesado en el video de salida
    writer.write(frame)

    # Actualizar los centroides del frame anterior
    prev_centroids = current_centroids.copy()

cap.release()
writer.release()
print(f"Video procesado y guardado en: {output_path}")