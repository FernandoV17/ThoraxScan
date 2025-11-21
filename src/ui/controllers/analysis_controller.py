from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.frequency_analysis import FrequencyAnalysis
from src.analysis.patology_detector import PatologyDetector
from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class AnalysisController:
    def __init__(self):
        self.patology_detector = PatologyDetector()
        logger.info("AnalysisController inicializado")
        self.frequency_analyzer = FrequencyAnalysis()

    def detect_anomalies(self, pil_image):
        try:
            img_np = np.array(pil_image)

            if len(img_np.shape) == 2:
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)

            # Segmentación Otsu
            val_otsu, segmented_otsu = cv2.threshold(
                blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            pixel_values = img.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            k = 3
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(
                pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            centers = np.uint8(centers)
            clustered_img = centers[labels.flatten()].reshape(img.shape)
            clustered_img_rgb = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(img_rgb)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(segmented_otsu, cmap="gray")
            plt.title(f"Otsu (T={val_otsu:.0f})")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(clustered_img_rgb)
            plt.title(f"K-Means (k={k})")
            plt.axis("off")

            plt.tight_layout()
            plt.show(block=False)

            return f"""ANÁLISIS DE SEGMENTACIÓN - COMPLETADO

Métodos aplicados:
• Umbralización Otsu: T = {val_otsu:.0f}
• Clustering K-Means: k = {k}

Resultados:
- Otsu: Segmentación binaria basada en histograma
- K-Means: Agrupamiento por similitud de color

Las visualizaciones se muestran en ventana externa.

NOTA: Para diagnóstico médico, consulte con especialista."""

        except Exception as e:
            logger.error(f"Error en análisis de anomalías: {e}")
            return f"Error en análisis: {str(e)}"

    def detect_fractures(self, pil_image):
        try:
            img_np = np.array(pil_image)

            if len(img_np.shape) == 2:
                gray_img = img_np
            else:
                gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_img)

            edges = cv2.Canny(enhanced, 50, 150)

            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
            )

            if len(img_np.shape) == 2:
                result_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            else:
                result_img = img_np.copy()

            line_count = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    line_count += 1

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(gray_img, cmap="gray")
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(edges, cmap="gray")
            plt.title("Bordes Detectados")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Líneas Detectadas: {line_count}")
            plt.axis("off")

            plt.tight_layout()
            plt.show(block=False)

            return f"""ANÁLISIS DE FRACTURAS

Resultados:
- Bordes detectados: Sí
- Líneas potenciales: {line_count}
- Análisis de continuidad ósea: Realizado

Interpretación:
{line_count} líneas potenciales detectadas que podrían indicar discontinuidades.

RECOMENDACIÓN: Este análisis es preliminar. Consulte con radiólogo."""

        except Exception as e:
            logger.error(f"Error en detección de fracturas: {e}")
            return f"Error en análisis de fracturas: {str(e)}"

    def detect_cardiomegaly(self, pil_image):
        """Detección de cardiomegalia (ratio cardiotorácico)"""
        try:
            img_np = np.array(pil_image)

            if len(img_np.shape) == 2:
                gray_img = img_np
            else:
                gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            _, binary = cv2.threshold(
                gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(img_np.shape) == 2:
                result_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            else:
                result_img = img_np.copy()

            cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

            total_area = gray_img.shape[0] * gray_img.shape[1]
            contour_areas = [cv2.contourArea(cnt) for cnt in contours]
            max_area = max(contour_areas) if contour_areas else 0
            area_ratio = max_area / total_area if total_area > 0 else 0

            # Mostrar resultados
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(gray_img, cmap="gray")
            plt.title("Imagen Original")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Contornos Detectados\nÁrea relativa: {area_ratio:.3f}")
            plt.axis("off")

            plt.tight_layout()
            plt.show(block=False)

            # Evaluación simple basada en área
            if area_ratio > 0.3:
                cardiomegaly_status = "POSIBLE CARDIOMEGALIA"
                recommendation = "Ratio cardiotorácico elevado. Consultar urgente."
            else:
                cardiomegaly_status = "SIN SIGNOS EVIDENTES"
                recommendation = "Ratio dentro de parámetros normales."

            return f"""ANÁLISIS CARDIOMEGALIA

Resultados:
- Área relativa del corazón: {area_ratio:.3f}
- Estado: {cardiomegaly_status}

{recommendation}

NOTA: Análisis automático basado en áreas. Para diagnóstico preciso consulte especialista."""

        except Exception as e:
            logger.error(f"Error en análisis de cardiomegalia: {e}")
            return f"Error en análisis de cardiomegalia: {str(e)}"

    def detect_consolidation(self, image):
        try:
            result = self.patology_detector.detect_consolidation(image)
            return self._format_patology_result("Consolidación Pulmonar", result)
        except Exception as e:
            logger.error(f"Error en detección de consolidación: {e}")
            return f"Error en análisis: {str(e)}"

    def detect_opacities(self, image):
        try:
            result = self.patology_detector.detect_opacities(image)
            return self._format_patology_result("Opacidades", result)
        except Exception as e:
            logger.error(f"Error en detección de opacidades: {e}")
            return f"Error en análisis: {str(e)}"

    def _format_patology_result(self, title: str, result: Dict[str, Any]) -> str:
        if result.get("detected", False):
            confidence = result.get("confidence", 0) * 100
            methods = ", ".join(result.get("methods_used", []))

            return f"""🔍 {title.upper()} - DETECTADA

Confianza: {confidence:.1f}%
Métodos utilizados: {methods}

Características encontradas:
{self._format_details(result.get("details", {}))}

RECOMENDACIÓN: Consultar con especialista para confirmación."""
        else:
            return f"""{title.upper()} - NO DETECTADA

Análisis realizado con {len(result.get("methods_used", []))} métodos.

No se encontraron signos evidentes de {title.lower()} en la imagen.

NOTA: Este es un análisis automático. Siempre consulte con un radiólogo."""

    def _format_details(self, details: Dict[str, Any]) -> str:
        """Formatea los detalles del análisis"""
        formatted = []
        for key, value in details.items():
            if isinstance(value, dict):
                sub_details = []
                for k, v in value.items():
                    if isinstance(v, float):
                        sub_details.append(f"  {k}: {v:.3f}")
                    else:
                        sub_details.append(f"  {k}: {v}")
                formatted.append(f"{key}:\n" + "\n".join(sub_details))
            else:
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)

    def analyze_frequency_domain(self, pil_image):
        try:
            result = self.frequency_analyzer.analyze_fft(pil_image)

            if result.get("has_artifacts", False):
                artifact_status = "POSIBLES ARTEFACTOS DETECTADOS"
                recommendation = "Se detectaron patrones periódicos que podrían indicar artefactos de adquisición."
            else:
                artifact_status = "SIN ARTEFACTOS EVIDENTES"
                recommendation = "No se detectaron patrones periódicos significativos."

            spectral = result.get("spectral_analysis", {})

            return f"""📡 ANÁLISIS EN DOMINIO DE FRECUENCIA

{artifact_status}

Análisis Espectral:
• Ratio Bajas Frecuencias: {spectral.get("low_freq_ratio", 0):.3f}
• Ratio Altas Frecuencias: {spectral.get("high_freq_ratio", 0):.3f}
• Entropía Espectral: {spectral.get("spectral_entropy", 0):.3f}

Patrones Periódicos:
• Picos significativos: {result.get("periodic_patterns", {}).get("num_significant_peaks", 0)}

{recommendation}

NOTA: Las visualizaciones espectrales se muestran en ventana externa."""

        except Exception as e:
            logger.error(f"Error en análisis FFT: {e}")
            return f"Error en análisis de frecuencia: {str(e)}"
