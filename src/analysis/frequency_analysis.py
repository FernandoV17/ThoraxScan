from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
from PIL import Image

from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class FrequencyAnalysis:
    def __init__(self):
        logger.info("FrequencyAnalysis inicializado")

    def analyze_fft(self, image: Image.Image) -> Dict[str, Any]:
        try:
            img_array = np.array(image)

            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            fft_result = self._compute_fft(img_array)

            spectral_analysis = self._spectral_analysis(fft_result)

            periodic_patterns = self._detect_periodic_patterns(fft_result)

            self._visualize_fft_results(img_array, fft_result)

            return {
                "spectral_analysis": spectral_analysis,
                "periodic_patterns": periodic_patterns,
                "has_artifacts": periodic_patterns["has_periodic_patterns"],
            }

        except Exception as e:
            logger.error(f"Error en análisis FFT: {e}")
            return {}

    def _compute_fft(self, img_array: np.ndarray) -> np.ndarray:
        rows, cols = img_array.shape
        window = np.outer(np.hanning(rows), np.hanning(cols))
        windowed_img = img_array * window

        fft = fftpack.fft2(windowed_img)
        fft_shifted = fftpack.fftshift(fft)  # Centrar las bajas frecuencias

        magnitude_spectrum = np.log(1 + np.abs(fft_shifted))

        return magnitude_spectrum

    def _spectral_analysis(self, fft_result: np.ndarray) -> Dict[str, float]:
        mean_intensity = np.mean(fft_result)
        std_intensity = np.std(fft_result)
        max_intensity = np.max(fft_result)

        center_y, center_x = fft_result.shape[0] // 2, fft_result.shape[1] // 2

        low_freq_radius = min(center_x, center_y) // 4
        low_freq_mask = self._create_circular_mask(
            fft_result.shape, center_x, center_y, low_freq_radius
        )
        low_freq_energy = np.sum(fft_result[low_freq_mask])

        high_freq_energy = np.sum(fft_result[~low_freq_mask])
        total_energy = low_freq_energy + high_freq_energy

        return {
            "mean_intensity": float(mean_intensity),
            "std_intensity": float(std_intensity),
            "max_intensity": float(max_intensity),
            "low_freq_ratio": float(low_freq_energy / total_energy)
            if total_energy > 0
            else 0,
            "high_freq_ratio": float(high_freq_energy / total_energy)
            if total_energy > 0
            else 0,
            "spectral_entropy": float(self._calculate_spectral_entropy(fft_result)),
        }

    def _detect_periodic_patterns(self, fft_result: np.ndarray) -> Dict[str, Any]:
        try:
            threshold = np.mean(fft_result) + 2 * np.std(fft_result)
            peaks = fft_result > threshold

            center_y, center_x = fft_result.shape[0] // 2, fft_result.shape[1] // 2
            exclusion_radius = min(center_x, center_y) // 8

            center_mask = self._create_circular_mask(
                fft_result.shape, center_x, center_y, exclusion_radius
            )
            significant_peaks = peaks & ~center_mask

            num_peaks = np.sum(significant_peaks)

            return {
                "has_periodic_patterns": num_peaks > 5,
                "num_significant_peaks": int(num_peaks),
                "peak_intensity_threshold": float(threshold),
            }
        except Exception as e:
            logger.warning(f"Error detectando patrones periódicos: {e}")
            return {"has_periodic_patterns": False, "num_significant_peaks": 0}

    def _create_circular_mask(self, shape, center_x, center_y, radius):
        y, x = np.ogrid[: shape[0], : shape[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        return mask

    def _calculate_spectral_entropy(self, spectrum: np.ndarray) -> float:
        spectrum_flat = spectrum.flatten()
        spectrum_flat = spectrum_flat / np.sum(spectrum_flat)

        spectrum_flat = spectrum_flat[spectrum_flat > 0]
        entropy = -np.sum(spectrum_flat * np.log2(spectrum_flat))

        return entropy

    def _visualize_fft_results(self, original: np.ndarray, fft_result: np.ndarray):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Imagen original
        axes[0, 0].imshow(original, cmap="gray")
        axes[0, 0].set_title("Imagen Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(fft_result, cmap="hot")
        axes[0, 1].set_title("Espectro de Frecuencia (FFT)")
        axes[0, 1].axis("off")

        axes[1, 0].hist(fft_result.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title("Distribución de Intensidades Espectrales")
        axes[1, 0].set_xlabel("Intensidad")
        axes[1, 0].set_ylabel("Frecuencia")

        self._plot_radial_analysis(fft_result, axes[1, 1])

        plt.tight_layout()
        plt.show(block=False)

    def _plot_radial_analysis(self, fft_result: np.ndarray, ax):
        center_y, center_x = fft_result.shape[0] // 2, fft_result.shape[1] // 2
        max_radius = min(center_x, center_y)

        radial_profile = []
        radii = range(1, max_radius)

        for r in radii:
            mask = self._create_circular_mask(fft_result.shape, center_x, center_y, r)
            prev_mask = self._create_circular_mask(
                fft_result.shape, center_x, center_y, r - 1
            )
            ring_mask = mask & ~prev_mask

            if np.any(ring_mask):
                radial_profile.append(np.mean(fft_result[ring_mask]))
            else:
                radial_profile.append(0)

        ax.plot(radii, radial_profile)
        ax.set_title("Perfil Radial del Espectro")
        ax.set_xlabel("Radio (píxeles)")
        ax.set_ylabel("Intensidad Promedio")
        ax.grid(True, alpha=0.3)
