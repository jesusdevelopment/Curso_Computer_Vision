# %% [markdown]
# ## 1) Montar Drive e Instalación de Dependencias
# %%
from google.colab import drive
drive.mount('/content/drive')

!apt-get install unrar -y -q
!pip install ultralytics -q

import os
import cv2
import numpy as np
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from IPython.display import Image, display

# %% [markdown]
# ## 2) Definición de Rutas y Descompresión (Optimizado)
# %%
# Rutas en Google Drive (Para guardar y leer el .rar original)
drive_dir = "/content/drive/MyDrive/Data/Curso Computer Vision"
rar_path = os.path.join(drive_dir, "defectos_de_soldaduras.rar")
save_dir = os.path.join(drive_dir, "testing")

# Rutas Locales en Colab (¡MUCHO más rápido para leer miles de imágenes!)
local_dir = "/content"
dataset_dir = os.path.join(local_dir, "defectos_de_soldaduras")
yaml_path = os.path.join(dataset_dir, "data.yaml")

# Extraer en almacenamiento local SOLO si no se ha extraído antes
if not os.path.exists(dataset_dir):
    print("Extrayendo archivos localmente para mayor velocidad...")
    !unrar x -o+ "{rar_path}" "{local_dir}/"
else:
    print("✅ El dataset ya está extraído en el almacenamiento local.")

# %% [markdown]
# ## 3) Entrenamiento del Modelo ⚙️🔍
# %%
model = YOLO("yolo11n.pt")

# Entrenamos con los datos locales (yaml_path), pero guardamos en Drive (project)

results = model.train(
    data=yaml_path,
    epochs=10,
    imgsz=640,
    augment=True,
    patience=5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.5,
    perspective=0.0005,
    degrees=5.0,
    shear=2.0,
    scale=0.5,    
    project=os.path.join(base_dir, "runs"),
    name="soldaduras_exp",
    exist_ok=True
)
# %% [markdown]
# ## 4) Cargar el Modelo Guardado en Drive 💾
# %%
best_model_path = os.path.join(drive_dir, "runs/soldaduras_exp/weights/best.pt")
mi_modelo = YOLO(best_model_path)

# %% [markdown]
# ## 5) Predicciones y Guardado en Drive 🤙
# %%
os.makedirs(save_dir, exist_ok=True)
# Leemos la imagen de prueba desde la carpeta local rápida
image_path = os.path.join(dataset_dir, "valid/images/SampleV1_1_mp4-24_jpg.rf.8487d87bb4c1d1ab9059da84ac881495.jpg")

# 📍 1) Visualizar resultados estándar
results = mi_modelo(image_path)

for result in results:
    annotated_frame = result.plot()
    result_image_path = os.path.join(save_dir, "pred_" + os.path.basename(result.path))
    cv2.imwrite(result_image_path, annotated_frame)
    display(Image(filename=result_image_path))

# 📍 2) Filtrar por confianza mayor al 30%
results_conf = mi_modelo(image_path, conf=0.30)

for result in results_conf:
    annotated_frame = result.plot()
    result_image_path = os.path.join(save_dir, "conf30_" + os.path.basename(result.path))
    cv2.imwrite(result_image_path, annotated_frame)
    display(Image(filename=result_image_path))

# 📍 3) Filtrar por clase específica
results_class = mi_modelo(image_path, conf=0.30, classes=[1])

for result in results_class:
    annotated_frame = result.plot()
    result_image_path = os.path.join(save_dir, "class1_" + os.path.basename(result.path))
    cv2.imwrite(result_image_path, annotated_frame)
    display(Image(filename=result_image_path))

# %% [markdown]
# ## 6) Función de Extracción de Datos 📍
# %%
def detectar_objetos(image_path, model):
    results = model(image_path)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    classes_idx = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    names = result.names if hasattr(result, "names") else model.names
    class_names = [names[i] for i in classes_idx]
    annotated_img = result.plot()

    return boxes, class_names, confidences, annotated_img

# Prueba de la función (leyendo desde local)
image_path_test = os.path.join(dataset_dir, "valid/images/SampleV2_1_mp4-26_jpg.rf.a8431cb1acce1c55d32861c982c2a16a.jpg")

boxes, classes, confs, img_anotada = detectar_objetos(image_path_test, mi_modelo)

print("Bounding Boxes:\n", boxes)
print("Class Names:", classes)
print("Confidences:", confs)

cv2_imshow(img_anotada)