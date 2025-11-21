import json
import os

from src.analysis.frequency_analysis import FrequencyAnalysis
from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class UIStyleManager:
    def __init__(self):
        path = os.path.join("assets", "ui", "styles.json")
        with open(path, "r") as f:
            self.styles = json.load(f)

    def get(self, key_path):
        keys = key_path.split(".")
        value = self.styles
        for k in keys:
            value = value.get(k, None)
            if value is None:
                return None
        return value

    def get_font(self, category="global"):
        fam = self.get(f"{category}.font_family") or self.get("global.font_family")
        size = self.get(f"{category}.font_size") or self.get("global.font_size")
        weight = self.get(f"{category}.font_weight") or "normal"
        return (fam, size, weight)

    def apply_background(self, widget, category="panel"):
        color = self.get(f"{category}.bg_color") or self.get("global.bg_color")
        widget.configure(bg=color)

    def apply_button_style(self, button):
        btn = self.styles["button"]
        button.configure(
            bg=btn["bg_color"],
            fg=btn["fg_color"],
            font=(self.styles["global"]["font_family"], btn["font_size"]),
        )


class AnalysisController:
    def __init__(self):
        self.frequency_analyzer = (
            FrequencyAnalysis()
        )
        logger.info("AnalysisController inicializado")

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

            return f"""ANÁLISIS EN DOMINIO DE FRECUENCIA

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
