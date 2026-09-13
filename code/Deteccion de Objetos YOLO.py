# %% [markdown]
# ## 1. Libraries
!pip install ultralytics -q
from ultralytics import YOLO
import cv2
from google.colab import drive
from google.colab.patches import cv2_imshow
from IPython.display import clear_output
# %% [markdown]
# ## 2. Model Loading
model = YOLO('yolo11n')
# ## 3. Video Loading
drive.mount('/content/drive')
# CAMBIA ESTA RUTA por la ruta de un video que tengas en tu Drive
path_video = "/content/drive/MyDrive/Data/Curso Computer Vision/store-aisle-detection.mp4" 

cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("Error: No se pudo abrir el archivo de video.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video.")
            break

        # Realizar la detección en el frame
        # Realizar la detección en el frame con configuración personalizada
        results = model(
            frame, 
            conf=0.6,        # Nivel de confianza mínimo (60%)
            classes=[0, 2],  # Lista de clases permitidas (ej. 0=persona, 2=carro)
            verbose=False
        )
        
        # Acceder al primer resultado
        boxes_obj = results[0].boxes

        # Si se detectaron objetos, dibujar sobre el frame
        if boxes_obj is not None and len(boxes_obj) > 0:
            bboxes = boxes_obj.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
            confs = boxes_obj.conf.cpu().numpy()    # Confianza
            classes = boxes_obj.cls.cpu().numpy()   # Clases
            
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(classes[i])] if hasattr(model, 'names') else str(int(classes[i]))
                label = f'{class_name} {confs[i]:.2f}'
                
                # Dibujar bounding box y etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # --- VISUALIZACIÓN PARA COLAB ---
        clear_output(wait=True) # Limpiar el fotograma anterior
        
        # Redimensionar el frame para que no sea inmenso en el navegador
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (w // 2, h // 2))
        
        cv2_imshow(frame_resized) # Mostrar en la celda

    cap.release()

# %%
