from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class FrequencyAnalysis:
    def __init__(self):
        logger.info("FrequencyAnalysis inicializado")

    def analyze_fft(self, pil_image):
        """Analiza la imagen en el dominio de frecuencia"""
        try:
            # Convertir a array numpy y escala de grises
            if isinstance(pil_image, Image.Image):
                img_array = np.array(pil_image.convert("L"))
            else:
                img_array = np.array(pil_image)

            # Calcular FFT
            f = np.fft.fft2(img_array)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

            # Análisis espectral
            spectral_analysis = self._analyze_spectrum(magnitude_spectrum)

            # Detectar artefactos
            has_artifacts = self._detect_artifacts(magnitude_spectrum)

            # Visualizar
            self._visualize_fft(img_array, magnitude_spectrum)

            return {
                "has_artifacts": has_artifacts,
                "spectral_analysis": spectral_analysis,
                "magnitude_spectrum": magnitude_spectrum.tolist(),
            }

        except Exception as e:
            logger.error(f"Error en análisis FFT: {e}")
            return {"has_artifacts": False, "spectral_analysis": {}, "error": str(e)}

    def _analyze_spectrum(self, magnitude_spectrum):
        """Analiza el espectro de frecuencia"""
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2

        # Definir regiones de frecuencia
        center_region = magnitude_spectrum[crow - 30 : crow + 30, ccol - 30 : ccol + 30]
        low_freq_region = magnitude_spectrum[
            crow - 60 : crow + 60, ccol - 60 : ccol + 60
        ]
        high_freq_region = magnitude_spectrum

        # Calcular ratios
        total_energy = np.sum(magnitude_spectrum)
        low_freq_energy = np.sum(low_freq_region) if total_energy > 0 else 0
        high_freq_energy = total_energy - low_freq_energy

        return {
            "low_freq_ratio": low_freq_energy / total_energy if total_energy > 0 else 0,
            "high_freq_ratio": high_freq_energy / total_energy
            if total_energy > 0
            else 0,
            "spectral_entropy": self._calculate_spectral_entropy(magnitude_spectrum),
            "total_energy": float(total_energy),
        }

    def _calculate_spectral_entropy(self, magnitude_spectrum):
        """Calcula la entropía espectral"""
        # Normalizar el espectro
        spectrum_normalized = magnitude_spectrum / np.sum(magnitude_spectrum)
        # Calcular entropía
        entropy = -np.sum(spectrum_normalized * np.log(spectrum_normalized + 1e-10))
        return float(entropy)

    def _detect_artifacts(self, magnitude_spectrum):
        """Detecta artefactos periódicos en el espectro"""
        # Buscar patrones repetitivos (líneas en el espectro)
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2

        # Excluir el centro (bajas frecuencias)
        mask = np.ones_like(magnitude_spectrum)
        mask[crow - 20 : crow + 20, ccol - 20 : ccol + 20] = 0

        masked_spectrum = magnitude_spectrum * mask

        # Buscar picos significativos fuera del centro
        threshold = np.mean(masked_spectrum) + 2 * np.std(masked_spectrum)
        high_peaks = np.sum(masked_spectrum > threshold)

        return high_peaks > 10  # Si hay más de 10 picos significativos

    def _visualize_fft(self, original, magnitude_spectrum):
        """Visualiza la FFT"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap="gray")
        plt.title("Imagen Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap="hot")
        plt.title("Espectro de Frecuencia (FFT)")
        plt.axis("off")

        plt.tight_layout()
        plt.show(block=False)

    def apply_frequency_filter(
        self, pil_image, filter_type="high", cutoff=30, strength=1.0
    ):
        """Aplica filtro en dominio de frecuencia con parámetros configurables"""
        try:
            # Convertir a array
            img_array = np.array(pil_image.convert("L"))

            # FFT
            f = np.fft.fft2(img_array)
            fshift = np.fft.fftshift(f)

            rows, cols = img_array.shape
            crow, ccol = rows // 2, cols // 2

            # Crear máscara según el tipo de filtro
            if filter_type == "high":  # Pasa altas - elimina bajas frecuencias
                mask = np.ones((rows, cols), np.float32)
                cutoff_adj = int(cutoff * strength)
                mask[
                    crow - cutoff_adj : crow + cutoff_adj,
                    ccol - cutoff_adj : ccol + cutoff_adj,
                ] = 0

            elif filter_type == "low":  # Pasa bajas - elimina altas frecuencias
                mask = np.zeros((rows, cols), np.float32)
                cutoff_adj = int(cutoff * strength)
                mask[
                    crow - cutoff_adj : crow + cutoff_adj,
                    ccol - cutoff_adj : ccol + cutoff_adj,
                ] = 1

            elif filter_type == "band":  # Pasa banda
                mask = np.zeros((rows, cols), np.float32)
                outer_cutoff = int(cutoff * strength)
                inner_cutoff = int(cutoff * 0.5 * strength)
                mask[
                    crow - outer_cutoff : crow + outer_cutoff,
                    ccol - outer_cutoff : ccol + outer_cutoff,
                ] = 1
                mask[
                    crow - inner_cutoff : crow + inner_cutoff,
                    ccol - inner_cutoff : ccol + inner_cutoff,
                ] = 0

            elif filter_type == "band_stop":  # Rechaza banda
                mask = np.ones((rows, cols), np.float32)
                outer_cutoff = int(cutoff * strength)
                inner_cutoff = int(cutoff * 0.5 * strength)
                mask[
                    crow - outer_cutoff : crow + outer_cutoff,
                    ccol - outer_cutoff : ccol + outer_cutoff,
                ] = 0
                mask[
                    crow - inner_cutoff : crow + inner_cutoff,
                    ccol - inner_cutoff : ccol + inner_cutoff,
                ] = 1

            # Aplicar filtro
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)

            # Normalizar y convertir
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)

            return Image.fromarray(img_back)

        except Exception as e:
            logger.error(f"Error aplicando filtro frecuencia: {e}")
            return pil_image

    def get_spectrum_image(self, pil_image):
        """Obtiene la imagen del espectro para visualización"""
        try:
            img_array = np.array(pil_image.convert("L"))

            # Calcular FFT
            f = np.fft.fft2(img_array)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

            # Normalizar para visualización
            spectrum_normalized = cv2.normalize(
                magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
            )

            return Image.fromarray(spectrum_normalized.astype(np.uint8))

        except Exception as e:
            logger.error(f"Error obteniendo espectro: {e}")
            return pil_image
