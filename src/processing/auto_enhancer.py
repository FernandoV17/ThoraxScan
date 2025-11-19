import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class AutoEnhancer:
    def __init__(self):
        pass

    def enhance(self, image):
        try:
            if not image:
                return image

            enhanced = image.copy()

            enhanced = self._enhance_contrast(enhanced)

            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))

            enhanced = enhanced.filter(
                ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
            )

            return enhanced

        except Exception as e:
            print(f"Error en auto-enhance: {e}")
            return image

    def _enhance_contrast(self, image):
        try:
            img_array = np.array(image)

            if len(img_array.shape) == 2:  # Escala de grises
                hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * float(hist.max()) / cdf.max()

                cdf_m = np.ma.masked_equal(cdf, 0)
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype("uint8")

                img_enhanced = cdf[img_array]
                return Image.fromarray(img_enhanced)
            else:
                return image

        except:
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
