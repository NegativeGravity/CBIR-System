from django.apps import AppConfig
import joblib
import os
import logging
from rembg import new_session
import onnxruntime as ort
from ml_engine.feature_extractor import ImageFeatureExtractor

logger = logging.getLogger('cbir_app')

class ImageSearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'image_search'
    ml_artifacts = None
    extractor = None
    rembg_session = None

    def ready(self):
        if os.environ.get('RUN_MAIN') != 'true':
            return
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_path = os.path.join(base_dir, 'ml_engine', 'artifacts', 'system_core.pkl')
        if os.path.exists(artifacts_path):
            self.ml_artifacts = joblib.load(artifacts_path)
            self.extractor = ImageFeatureExtractor()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.rembg_session = new_session("u2net", providers=providers)