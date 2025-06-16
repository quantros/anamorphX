"""
AnamorphX Neural Translator
Главный транслятор AnamorphX → PyTorch
"""

import os
from typing import Dict, Any, List
from .network_parser import NetworkParser
from .pytorch_generator import PyTorchGenerator


class NeuralTranslator:
    """Главный транслятор AnamorphX в PyTorch"""
    
    def __init__(self, output_dir: str = "generated_models"):
        self.parser = NetworkParser()
        self.generator = PyTorchGenerator()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def translate_code(self, anamorphx_code: str) -> Dict[str, Any]:
        """Трансляция AnamorphX кода в PyTorch"""
        try:
            networks = self.parser.parse_code(anamorphx_code)
            
            if not networks:
                return {
                    'success': False,
                    'error': 'No networks found in code',
                    'networks': [],
                    'files_generated': []
                }
            
            # Генерация файлов
            generated_files = []
            for network in networks:
                files = self._generate_network_files(network)
                generated_files.extend(files)
            
            return {
                'success': True,
                'networks': [net.name for net in networks],
                'files_generated': generated_files,
                'output_directory': self.output_dir
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Translation failed: {str(e)}',
                'networks': [],
                'files_generated': []
            }
    
    def _generate_network_files(self, network) -> List[str]:
        """Генерация файлов для одной сети"""
        generated_files = []
        class_name = network.name
        
        # Файл модели
        model_code = self.generator.generate_model_class(network)
        model_file = os.path.join(self.output_dir, f"{class_name.lower()}_model.py")
        
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(model_code)
        generated_files.append(model_file)
        
        # Скрипт обучения
        train_code = self.generator.generate_training_script(network)
        train_file = os.path.join(self.output_dir, f"train_{class_name.lower()}.py")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(train_code)
        generated_files.append(train_file)
        
        return generated_files 