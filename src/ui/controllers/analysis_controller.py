from src.analysis.controller import AnalysisController as AnalysisEngine
from src.helpers.logger import get_module_logger

logger = get_module_logger(__name__)


class AnalysisController:
    def __init__(self):
        self.analysis_engine = AnalysisEngine()
        logger.info("AnalysisController inicializado")

    def detect_anomalies(self, image):
        try:
            return self.analysis_engine.detect_anomalies(image)
        except Exception as e:
            logger.error(f"Error en detección de anomalías: {e}")
            return f"Error en análisis: {str(e)}"

    def detect_fractures(self, image):
        try:
            return self.analysis_engine.detect_fractures(image)
        except Exception as e:
            logger.error(f"Error en detección de fracturas: {e}")
            return f"Error en análisis: {str(e)}"

    def detect_cardiomegaly(self, image):
        try:
            return self.analysis_engine.detect_cardiomegaly(image)
        except Exception as e:
            logger.error(f"Error en análisis de cardiomegalia: {e}")
            return f"Error en análisis: {str(e)}"
