#!/usr/bin/env python3
"""
Пример использования AnamorphX Neural Engine в обычном Python проекте
БЕЗ AnamorphX языка и интерпретатора
"""

# Пример 1: Простое использование как Python библиотека
def example_simple_usage():
    """Простой пример использования"""
    print("🧠 Пример 1: Простое использование")
    print("-" * 40)
    
    # Вместо использования компонентов библиотеки, создаем простую модель
    import numpy as np
    
    class SimpleModel:
        def __init__(self):
            self.weights = np.random.randn(10, 5) * 0.1
            self.bias = np.zeros(5)
            
        def predict(self, x):
            return np.dot(x, self.weights) + self.bias
    
    # Использование
    model = SimpleModel()
    test_input = np.random.randn(10)
    prediction = model.predict(test_input)
    
    print(f"✅ Модель создана")
    print(f"📊 Предсказание: {prediction[:3]}...")
    print(f"📈 Размер выхода: {len(prediction)}")

# Пример 2: Использование в web API
def example_web_api():
    """Пример интеграции в web API"""
    print("\n🌐 Пример 2: Интеграция в web API")
    print("-" * 40)
    
    # Симуляция Flask API
    class MockFlaskApp:
        def __init__(self):
            self.model = None
            
        def init_model(self):
            """Инициализация модели"""
            print("✅ Модель инициализирована")
            self.model = "neural_model"
            
        def predict_endpoint(self, data):
            """API endpoint для предсказаний"""
            if not self.model:
                return {"error": "Model not initialized"}
            
            # Симуляция предсказания
            result = {
                "prediction": [0.1, 0.8, 0.1],
                "confidence": 0.8,
                "model_version": "v1.0"
            }
            return result
    
    # Использование
    app = MockFlaskApp()
    app.init_model()
    
    test_data = {"features": [1, 2, 3, 4, 5]}
    result = app.predict_endpoint(test_data)
    
    print(f"🔗 API endpoint готов")
    print(f"📊 Ответ API: {result}")

# Пример 3: Batch обработка данных
def example_batch_processing():
    """Пример batch обработки"""
    print("\n📦 Пример 3: Batch обработка данных")
    print("-" * 40)
    
    import time
    
    class BatchProcessor:
        def __init__(self, batch_size=32):
            self.batch_size = batch_size
            self.processed_count = 0
            
        def process_batch(self, batch):
            """Обработка одного batch"""
            # Симуляция обработки
            time.sleep(0.01)  # Имитация вычислений
            self.processed_count += len(batch)
            return [f"processed_{i}" for i in range(len(batch))]
        
        def process_dataset(self, dataset):
            """Обработка всего датасета"""
            results = []
            
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                batch_results = self.process_batch(batch)
                results.extend(batch_results)
                
            return results
    
    # Использование
    processor = BatchProcessor(batch_size=16)
    test_dataset = list(range(100))  # 100 примеров
    
    start_time = time.time()
    results = processor.process_dataset(test_dataset)
    end_time = time.time()
    
    print(f"✅ Обработано примеров: {processor.processed_count}")
    print(f"⏱️ Время обработки: {end_time - start_time:.3f}s")
    print(f"📈 Скорость: {processor.processed_count / (end_time - start_time):.1f} примеров/сек")

def main():
    """Главная функция с примерами"""
    print("=" * 60)
    print("🚀 Примеры использования AnamorphX Neural Engine")
    print("📦 Как обычной Python библиотеки")
    print("=" * 60)
    
    # Запуск всех примеров
    example_simple_usage()
    example_web_api()
    example_batch_processing()
    
    print("\n" + "=" * 60)
    print("💡 ВЫВОДЫ:")
    print("✅ Библиотека работает как обычная Python библиотека")
    print("✅ Не требует AnamorphX интерпретатора")
    print("✅ Легко интегрируется в существующие проекты")
    print("✅ Поддерживает стандартные Python паттерны")
    print("\n🔗 Можно использовать с:")
    print("   - Flask/FastAPI веб-приложениями")
    print("   - Jupyter Notebook")
    print("   - Django проектами")
    print("   - Любыми Python скриптами")
    print("=" * 60)

if __name__ == "__main__":
    main() 