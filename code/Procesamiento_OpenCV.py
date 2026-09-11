# %% [markdown]
# ## 1. Libraries Installation and Importation
# !pip install opencv-python
import cv2 
import matplotlib.pyplot as plt
from google.colab import drive
from google.colab.patches import cv2_imshow
from IPython.display import clear_output
import numpy as np

# %% [markdown]
# ## 2. Image Loading and Display
# Load an image from Google Drive
drive.mount('/content/drive')
image_path = '/content/drive/MyDrive/Data/Curso Computer Vision/cctv_image.jpg'
image = cv2.imread(image_path)
# %%
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
else:
    # Convert image from BGR (OpenCV format) to RGB (matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Image Visualization with matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title("Imagen Capturada por CCTV")
    plt.axis("off")
    plt.show()
# %% [markdown]
# ## 3. Real Time Video Capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Para Windows
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break

        # Mostrar el frame en una ventana llamada 'Frame de CCTV'
        cv2.imshow("Frame de CCTV", frame)

        # Esperar 1 ms para detectar si se ha presionado la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar el objeto de captura y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

# %% [markdown]
# ## 4. Prerecorded videos
# Path video

'''path_video = "/content/drive/MyDrive/Data/Curso Computer Vision/store-aisle-detection.mp4"

# Cargar video
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("Error: No se pudo abrir el archivo de video.")
else:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Fin del video o no se pudo leer el frame.")
            break

        # Limpiar el frame anterior en la celda de Colab
        clear_output(wait=True)
        
        # Mostrar el frame actual en Colab
        cv2_imshow(frame)

    # Liberar el archivo de video al terminar
    cap.release()'''
# %% [markdown]
# ## 5.Bright and Contrast Tunning
image_path = '/content/drive/MyDrive/Data/Curso Computer Vision/cctv_image.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Error: No se pudo cargar la imagen.")
    exit()
    
# Ajuste de brillo y contraste: new_image = image * alpha + beta
alpha = 1.2  # Factor de contraste (mayor a 1 aumenta contraste)
beta = 30    # Valor de brillo (positivo aumenta el brillo)
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
# Convertir la imagen de BGR (formato OpenCV) a RGB (formato matplotlib)
adjusted = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
# Mostrar la imagen usando matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(adjusted)
plt.title("Imagen Capturada por CCTV con ajuste de brillo y contraste")
plt.axis("off")
plt.show()
# %% [markdown]
# ## 6. Normalization and Color Correction
# Convertir la imagen ajustada a espacio HSV para modificar la saturación
hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
# Aumentar la saturación
saturacion = 5
s = cv2.multiply(s, saturacion)  
s = np.clip(s, 0, 255).astype(np.uint8)
hsv_adjusted = cv2.merge([h, s, v])
color_corrected = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
# Mostrar la imagen usando matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(color_corrected)
plt.title("Imagen Capturada por CCTV con ajuste de saturación")
plt.axis("off")
plt.show()
# Convertir la imagen de BGR (formato OpenCV) a RGB (formato matplotlib)
color_corrected = cv2.cvtColor(color_corrected, cv2.COLOR_BGR2RGB)
# Guardar imagen para comparar
output_file = "./data/cctv_image_con_ajuste_saturacion.jpg"
cv2.imwrite(output_file, color_corrected)
print(f"Imagen guardada exitosamente en: {output_file}")
# %% [markdown]
# ## 7. Aplicación de Anotaciones
# Hacer una copia de una imagen ya cargada
annotated = image.copy()
# Dibujar una línea: desde (150,250) hasta (350,250) en azul BGR: (255, 0, 0) con grosor de línea 3 píxeles
cv2.line(annotated, (150, 250), (350, 250), (255, 0, 0), 3)
# Dibujar un rectángulo: esquina superior izquierda (150, 300), inferior derecha (350,600) en verde BGR: (0, 255, 0)
cv2.rectangle(annotated, (150, 300), (350, 600), (0, 255, 0), 3)
# Escribir texto: 'Persona' en rojo BGR: (0, 0, 255), ubicado en (200,290)
cv2.putText(annotated, 'Persona', (200, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# %%
# Convertir de BGR a RGB para visualizar con matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
adjusted_rgb = adjusted
color_corrected_rgb = cv2.cvtColor(color_corrected, cv2.COLOR_BGR2RGB)
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(adjusted_rgb)
plt.title("Brillo y Contraste Ajustados")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(color_corrected_rgb)
plt.title("Corrección de Color (Saturación)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(annotated_rgb)
plt.title("Imagen Anotada")
plt.axis("off")

plt.tight_layout()
plt.show()