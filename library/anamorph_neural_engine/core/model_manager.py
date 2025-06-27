"""
🧠 Model Manager - Enterprise Edition
Управление нейронными моделями
"""

import torch
import torch.nn as nn
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class ModelManager:
    """
    🧠 Enterprise Model Manager
    Управление жизненным циклом моделей
    """
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Loaded models registry
        self.loaded_models = {}
        self.model_metadata = {}
    
    def load_model(self, model_path: str, model_name: str = None) -> bool:
        """Загрузка модели"""
        try:
            model_name = model_name or Path(model_path).stem
            
            checkpoint = torch.load(model_path, map_location='cpu')
            self.loaded_models[model_name] = checkpoint
            
            # Store metadata
            self.model_metadata[model_name] = {
                'path': model_path,
                'loaded_at': datetime.now().isoformat(),
                'size_mb': Path(model_path).stat().st_size / 1024 / 1024
            }
            
            self.logger.info(f"Model loaded: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Получение модели"""
        return self.loaded_models.get(model_name)
    
    def list_models(self) -> List[str]:
        """Список загруженных моделей"""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Выгрузка модели"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.model_metadata[model_name]
            self.logger.info(f"Model unloaded: {model_name}")
            return True
        return False 