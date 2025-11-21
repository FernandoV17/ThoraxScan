import numpy as np
from PIL import Image


class FourierAnalysis:
    def __init__(self):
        self.logger = get_module_logger(__name__)

    def compute_fft(self, image):
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert("L"))
        else:
            img_array = np.array(image)

        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        return fshift, magnitude_spectrum

    def apply_frequency_filter(self, image, filter_type="high", cutoff=30):
        fshift, magnitude_spectrum = self.compute_fft(image)

        rows, cols = fshift.shape
        crow, ccol = rows // 2, cols // 2

        # Crear máscara
        mask = np.ones((rows, cols), np.uint8)

        if filter_type == "high":  # Pasa altas
            mask[crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff] = 0
        elif filter_type == "low":  # Pasa bajas
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff] = 1
        elif filter_type == "band":  # Pasa banda
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow - cutoff : crow + cutoff, ccol - cutoff : ccol + cutoff] = 1
            mask[
                crow - cutoff // 2 : crow + cutoff // 2,
                ccol - cutoff // 2 : ccol + cutoff // 2,
            ] = 0

        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return Image.fromarray(img_back.astype(np.uint8)), magnitude_spectrum, mask

    def analyze_frequency_domain(self, image):
        fshift, magnitude_spectrum = self.compute_fft(image)

        analysis = {
            "dominant_frequencies": np.sum(
                magnitude_spectrum > np.mean(magnitude_spectrum)
            ),
            "energy": np.sum(magnitude_spectrum),
            "mean_frequency": np.mean(magnitude_spectrum),
            "max_frequency": np.max(magnitude_spectrum),
            "recommendations": self.generate_recommendations(magnitude_spectrum),
        }

        return analysis

    def generate_recommendations(self, magnitude_spectrum):
        mean_val = np.mean(magnitude_spectrum)
        max_val = np.max(magnitude_spectrum)

        recommendations = []

        if max_val > mean_val * 3:
            recommendations.append(
                "Alto contenido de frecuencias altas - considere filtro pasa-bajas"
            )

        if np.sum(magnitude_spectrum > mean_val) < magnitude_spectrum.size * 0.1:
            recommendations.append(
                "Baja variación espectral - imagen posiblemente suavizada"
            )

        return recommendations
