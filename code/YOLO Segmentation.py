# %% [markdown]
# ## 1. Libraries
!pip install ultralytics -q
import cv2
import numpy as np
from ultralytics import YOLO
from google.colab import drive
from google.colab.patches import cv2_imshow
from IPython.display import clear_output

# %% [markdown]
# ## 2. Video Loading
drive.mount('/content/drive')
path_video = "/content/drive/MyDrive/Data/Curso Computer Vision/park_detection.avi"

# %% [markdown]
# ## 3. Open the video
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("Error: No se pudo abrir el archivo de video. Verifica la ruta.")

# %% [markdown]
# ## 4. Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,          # Número de frames usados para construir el fondo.
    varThreshold=16,      # Sensibilidad para detectar cambios
    detectShadows=True,   # Detección de sombras
)

heatmap_refined = None

# %% [markdown]
# ## 5. YOLOv11 model loading for segmentation
model = YOLO("yolo11n-seg")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error de lectura.")
        break

    # Inicializar el acumulador del heatmap en el primer frame
    if heatmap_refined is None:
        heatmap_refined = np.zeros(frame.shape[:2], dtype=np.float32)

    # --- Paso 1: Sustracción de Fondo ---
    fgmask = bg_subtractor.apply(frame)
    # Umbral para obtener una máscara binaria limpia
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # --- Paso 2: Segmentación con YOLO ---
    # Realizamos la detección con segmentación sobre el frame completo.
    results = model(frame, verbose=False)[0]

    # Crear una máscara vacía para acumular las segmentaciones de la clase "persona"
    segmentation_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if results.masks is not None:
        # Extraer las máscaras y las clases
        masks = results.masks.data.cpu().numpy() if hasattr(results.masks.data, 'cpu') else results.masks.data
        classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes.cls, 'cpu') else results.boxes.cls

        for mask, cls in zip(masks, classes):
            if int(cls) == 0:  # Filtramos detecciones de persona (en COCO, "person" es la clase 0)
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
                # Redimensionar mask_bin a las dimensiones del frame (o segmentation_mask)
                mask_bin_resized = cv2.resize(mask_bin, (segmentation_mask.shape[1], segmentation_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                segmentation_mask = cv2.bitwise_or(segmentation_mask, mask_bin_resized)

    # --- Paso 3: Combinación de Máscaras ---
    # Se realiza una intersección entre la máscara de movimiento y la máscara de segmentación de personas
    refined_mask = cv2.bitwise_and(fgmask, segmentation_mask)

    # Acumulamos la máscara refinada en el heatmap
    heatmap_refined = cv2.add(heatmap_refined, refined_mask.astype(np.float32))

    # --- VISUALIZACIÓN ADAPTADA PARA COLAB ---
    # Limpiar la celda antes de mostrar el siguiente frame
    clear_output(wait=True)

    # Convertir máscaras de 1 canal a 3 canales (BGR) para poder concatenarlas
    fgmask_3c = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    seg_mask_3c = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
    refined_mask_3c = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)

    # Redimensionar al 50% para formar una cuadrícula que quepa en pantalla
    h, w = frame.shape[:2]
    h_half, w_half = h // 2, w // 2

    frame_rs = cv2.resize(frame, (w_half, h_half))
    fgmask_rs = cv2.resize(fgmask_3c, (w_half, h_half))
    seg_mask_rs = cv2.resize(seg_mask_3c, (w_half, h_half))
    refined_mask_rs = cv2.resize(refined_mask_3c, (w_half, h_half))

    # Añadir títulos a cada imagen redimensionada
    cv2.putText(frame_rs, "Frame Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(fgmask_rs, "Movimiento (FG)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(seg_mask_rs, "Segmentacion (Personas)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(refined_mask_rs, "Mascara Refinada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Concatenar en una cuadrícula 2x2
    top_row = np.hstack((frame_rs, fgmask_rs))
    bottom_row = np.hstack((seg_mask_rs, refined_mask_rs))
    grid = np.vstack((top_row, bottom_row))

    # Mostrar el resultado final
    cv2_imshow(grid)

# Liberar captura de video al finalizar
cap.release()
# %%
