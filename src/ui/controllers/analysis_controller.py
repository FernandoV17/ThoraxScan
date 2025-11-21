import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class AnalysisController:
    def __init__(self):
        pass

    def detect_anomalies(self, pil_image):
        """
        Segmentación por K-Means y Otsu para el PIA.
        """
        try:
            # 1. Convertir imagen PIL a OpenCV
            img_np = np.array(pil_image)
            # Convertir de RGB (PIL) a BGR (OpenCV)
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            # Convertir a RGB para mostrar en Matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 2. Preprocesamiento
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)

            # 3. Otsu
            val_otsu, segmented_otsu = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 4. K-Means
            pixel_values = img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            k = 3 
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            clustered_img = centers[labels.flatten()].reshape(img.shape)
            clustered_img_rgb = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)

            # --- VISUALIZACIÓN ---
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_rgb)
            plt.title("Original")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(segmented_otsu, cmap='gray')
            plt.title(f"Otsu (T={val_otsu:.0f})")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(clustered_img_rgb)
            plt.title(f"K-Means (k={k})")
            plt.axis('off')

            plt.tight_layout()
            plt.show(block=False)

            return f"ANÁLISIS EXITOSO\nUmbral Otsu: {val_otsu}\nClusters K-Means: {k}\n\nGráficas generadas en ventana externa."

        except Exception as e:
            print(f"Error detallado: {e}")
            return f"Error en análisis: {str(e)}"

    def detect_fractures(self, pil_image):
        return "Módulo de Fracturas en desarrollo"

    def detect_cardiomegaly(self, pil_image):
        return "Módulo de Cardiomegalia en desarrollo"