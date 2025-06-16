"""
Команды машинного обучения AnamorphX

Команды для работы с моделями машинного обучения, обучением и предсказаниями.
"""

import numpy as np
import uuid
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .commands import MLOperationsCommand, CommandResult, CommandError, ExecutionContext


class ModelType(Enum):
    """Типы моделей машинного обучения"""
    NEURAL_NETWORK = "neural_network"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    CLUSTERING = "clustering"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    AUTOENCODER = "autoencoder"


class TrainingStatus(Enum):
    """Статусы обучения"""
    NOT_STARTED = "not_started"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    STOPPED = "stopped"


class OptimizationType(Enum):
    """Типы оптимизации"""
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    GENETIC = "genetic"
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"


@dataclass
class MLModel:
    """Модель машинного обучения"""
    id: str
    name: str
    model_type: ModelType
    parameters: Dict[str, Any]
    weights: Optional[np.ndarray] = None
    architecture: Optional[Dict[str, Any]] = None
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: TrainingStatus = TrainingStatus.NOT_STARTED
    accuracy: Optional[float] = None
    loss: Optional[float] = None


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: OptimizationType = OptimizationType.ADAM
    loss_function: str = "mse"
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    regularization: Optional[Dict[str, float]] = None


@dataclass
class Dataset:
    """Набор данных"""
    id: str
    name: str
    features: np.ndarray
    targets: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)


class TrainCommand(MLOperationsCommand):
    """Команда обучения модели"""
    
    def __init__(self):
        super().__init__(
            name="train",
            description="Обучает модель машинного обучения",
            parameters={
                "model_id": "Идентификатор модели",
                "dataset_id": "Идентификатор набора данных",
                "config": "Конфигурация обучения",
                "save_checkpoints": "Сохранять контрольные точки",
                "resume": "Продолжить обучение с последней точки"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            model_id = kwargs.get("model_id")
            dataset_id = kwargs.get("dataset_id")
            config_data = kwargs.get("config", {})
            save_checkpoints = kwargs.get("save_checkpoints", True)
            resume = kwargs.get("resume", False)
            
            if not model_id or not dataset_id:
                return CommandResult(
                    success=False,
                    message="Требуются model_id и dataset_id",
                    error=CommandError("MISSING_PARAMETERS", "model_id и dataset_id обязательны")
                )
            
            # Получаем модель и данные
            if not hasattr(context, 'ml_models'):
                context.ml_models = {}
            if not hasattr(context, 'datasets'):
                context.datasets = {}
            
            if model_id not in context.ml_models:
                return CommandResult(
                    success=False,
                    message=f"Модель {model_id} не найдена",
                    error=CommandError("MODEL_NOT_FOUND", f"Модель {model_id} не существует")
                )
            
            if dataset_id not in context.datasets:
                return CommandResult(
                    success=False,
                    message=f"Набор данных {dataset_id} не найден",
                    error=CommandError("DATASET_NOT_FOUND", f"Набор данных {dataset_id} не существует")
                )
            
            model = context.ml_models[model_id]
            dataset = context.datasets[dataset_id]
            config = TrainingConfig(**config_data)
            
            # Начинаем обучение
            model.status = TrainingStatus.TRAINING
            model.updated_at = time.time()
            
            # Симуляция обучения (в реальной реализации здесь был бы настоящий алгоритм)
            training_history = []
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(config.epochs):
                # Симуляция эпохи обучения
                epoch_loss = np.random.exponential(1.0) * (1.0 - epoch / config.epochs)
                epoch_accuracy = min(0.99, 0.5 + 0.4 * (epoch / config.epochs) + np.random.normal(0, 0.05))
                
                epoch_data = {
                    "epoch": epoch + 1,
                    "loss": float(epoch_loss),
                    "accuracy": float(epoch_accuracy),
                    "learning_rate": config.learning_rate,
                    "timestamp": time.time()
                }
                
                training_history.append(epoch_data)
                
                # Early stopping
                if config.early_stopping:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= config.patience:
                            break
                
                # Сохранение контрольных точек
                if save_checkpoints and (epoch + 1) % 10 == 0:
                    checkpoint_data = {
                        "epoch": epoch + 1,
                        "model_state": model.__dict__.copy(),
                        "optimizer_state": {"lr": config.learning_rate},
                        "loss": epoch_loss
                    }
                    if not hasattr(context, 'checkpoints'):
                        context.checkpoints = {}
                    context.checkpoints[f"{model_id}_epoch_{epoch + 1}"] = checkpoint_data
            
            # Завершаем обучение
            model.status = TrainingStatus.COMPLETED
            model.training_history.extend(training_history)
            model.loss = training_history[-1]["loss"]
            model.accuracy = training_history[-1]["accuracy"]
            model.updated_at = time.time()
            
            # Генерируем веса модели (симуляция)
            if model.model_type == ModelType.NEURAL_NETWORK:
                model.weights = np.random.randn(100, 50)  # Пример весов
            
            return CommandResult(
                success=True,
                message=f"Обучение модели {model_id} завершено за {len(training_history)} эпох",
                data={
                    "model_id": model_id,
                    "epochs_completed": len(training_history),
                    "final_loss": model.loss,
                    "final_accuracy": model.accuracy,
                    "training_time": model.updated_at - model.created_at,
                    "status": model.status.value,
                    "history": training_history[-5:]  # Последние 5 эпох
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка обучения: {str(e)}",
                error=CommandError("TRAINING_ERROR", str(e))
            )


class PredictCommand(MLOperationsCommand):
    """Команда предсказания"""
    
    def __init__(self):
        super().__init__(
            name="predict",
            description="Выполняет предсказание с помощью обученной модели",
            parameters={
                "model_id": "Идентификатор модели",
                "input_data": "Входные данные для предсказания",
                "batch_size": "Размер батча для обработки",
                "return_probabilities": "Возвращать вероятности"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            model_id = kwargs.get("model_id")
            input_data = kwargs.get("input_data")
            batch_size = kwargs.get("batch_size", 32)
            return_probabilities = kwargs.get("return_probabilities", False)
            
            if not model_id or input_data is None:
                return CommandResult(
                    success=False,
                    message="Требуются model_id и input_data",
                    error=CommandError("MISSING_PARAMETERS", "model_id и input_data обязательны")
                )
            
            if not hasattr(context, 'ml_models') or model_id not in context.ml_models:
                return CommandResult(
                    success=False,
                    message=f"Модель {model_id} не найдена",
                    error=CommandError("MODEL_NOT_FOUND", f"Модель {model_id} не существует")
                )
            
            model = context.ml_models[model_id]
            
            if model.status != TrainingStatus.COMPLETED:
                return CommandResult(
                    success=False,
                    message=f"Модель {model_id} не обучена",
                    error=CommandError("MODEL_NOT_TRAINED", "Модель должна быть обучена для предсказаний")
                )
            
            # Преобразуем входные данные
            if isinstance(input_data, list):
                input_array = np.array(input_data)
            elif isinstance(input_data, np.ndarray):
                input_array = input_data
            else:
                input_array = np.array([input_data])
            
            # Выполняем предсказание (симуляция)
            if model.model_type == ModelType.NEURAL_NETWORK:
                # Симуляция нейронной сети
                predictions = np.random.rand(len(input_array))
                if return_probabilities:
                    probabilities = np.random.dirichlet(np.ones(3), len(input_array))
                else:
                    probabilities = None
            elif model.model_type == ModelType.LINEAR_REGRESSION:
                # Симуляция линейной регрессии
                predictions = np.sum(input_array, axis=1) * 0.5 + np.random.normal(0, 0.1, len(input_array))
                probabilities = None
            else:
                # Общая симуляция
                predictions = np.random.rand(len(input_array))
                probabilities = None
            
            result_data = {
                "model_id": model_id,
                "predictions": predictions.tolist(),
                "input_shape": input_array.shape,
                "model_type": model.model_type.value,
                "prediction_time": time.time()
            }
            
            if return_probabilities and probabilities is not None:
                result_data["probabilities"] = probabilities.tolist()
            
            return CommandResult(
                success=True,
                message=f"Предсказание выполнено для {len(input_array)} образцов",
                data=result_data
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка предсказания: {str(e)}",
                error=CommandError("PREDICTION_ERROR", str(e))
            )


class EvaluateCommand(MLOperationsCommand):
    """Команда оценки модели"""
    
    def __init__(self):
        super().__init__(
            name="evaluate",
            description="Оценивает качество модели на тестовых данных",
            parameters={
                "model_id": "Идентификатор модели",
                "test_dataset_id": "Идентификатор тестового набора данных",
                "metrics": "Список метрик для оценки",
                "detailed": "Детальный отчет об оценке"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            model_id = kwargs.get("model_id")
            test_dataset_id = kwargs.get("test_dataset_id")
            metrics = kwargs.get("metrics", ["accuracy", "precision", "recall", "f1"])
            detailed = kwargs.get("detailed", False)
            
            if not model_id or not test_dataset_id:
                return CommandResult(
                    success=False,
                    message="Требуются model_id и test_dataset_id",
                    error=CommandError("MISSING_PARAMETERS", "model_id и test_dataset_id обязательны")
                )
            
            if not hasattr(context, 'ml_models') or model_id not in context.ml_models:
                return CommandResult(
                    success=False,
                    message=f"Модель {model_id} не найдена",
                    error=CommandError("MODEL_NOT_FOUND", f"Модель {model_id} не существует")
                )
            
            if not hasattr(context, 'datasets') or test_dataset_id not in context.datasets:
                return CommandResult(
                    success=False,
                    message=f"Тестовый набор данных {test_dataset_id} не найден",
                    error=CommandError("DATASET_NOT_FOUND", f"Набор данных {test_dataset_id} не существует")
                )
            
            model = context.ml_models[model_id]
            test_dataset = context.datasets[test_dataset_id]
            
            # Выполняем оценку (симуляция)
            evaluation_results = {}
            
            for metric in metrics:
                if metric == "accuracy":
                    evaluation_results[metric] = np.random.uniform(0.7, 0.95)
                elif metric == "precision":
                    evaluation_results[metric] = np.random.uniform(0.65, 0.9)
                elif metric == "recall":
                    evaluation_results[metric] = np.random.uniform(0.6, 0.88)
                elif metric == "f1":
                    precision = evaluation_results.get("precision", np.random.uniform(0.65, 0.9))
                    recall = evaluation_results.get("recall", np.random.uniform(0.6, 0.88))
                    evaluation_results[metric] = 2 * (precision * recall) / (precision + recall)
                elif metric == "mse":
                    evaluation_results[metric] = np.random.uniform(0.01, 0.1)
                elif metric == "mae":
                    evaluation_results[metric] = np.random.uniform(0.05, 0.2)
                else:
                    evaluation_results[metric] = np.random.uniform(0.5, 0.9)
            
            result_data = {
                "model_id": model_id,
                "test_dataset_id": test_dataset_id,
                "metrics": evaluation_results,
                "evaluation_time": time.time(),
                "test_samples": len(test_dataset.features) if hasattr(test_dataset, 'features') else 1000
            }
            
            if detailed:
                result_data["detailed_report"] = {
                    "confusion_matrix": np.random.randint(0, 100, (3, 3)).tolist(),
                    "classification_report": {
                        "class_0": {"precision": 0.85, "recall": 0.82, "f1-score": 0.83},
                        "class_1": {"precision": 0.78, "recall": 0.81, "f1-score": 0.79},
                        "class_2": {"precision": 0.92, "recall": 0.89, "f1-score": 0.90}
                    },
                    "feature_importance": np.random.rand(10).tolist() if hasattr(test_dataset, 'feature_names') else None
                }
            
            return CommandResult(
                success=True,
                message=f"Оценка модели {model_id} завершена",
                data=result_data
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка оценки: {str(e)}",
                error=CommandError("EVALUATION_ERROR", str(e))
            )


class OptimizeCommand(MLOperationsCommand):
    """Команда оптимизации гиперпараметров"""
    
    def __init__(self):
        super().__init__(
            name="optimize",
            description="Оптимизирует гиперпараметры модели",
            parameters={
                "model_id": "Идентификатор модели",
                "parameter_space": "Пространство поиска параметров",
                "optimization_method": "Метод оптимизации",
                "max_trials": "Максимальное количество попыток",
                "objective": "Целевая метрика для оптимизации"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            model_id = kwargs.get("model_id")
            parameter_space = kwargs.get("parameter_space", {})
            optimization_method = OptimizationType(kwargs.get("optimization_method", "grid_search"))
            max_trials = kwargs.get("max_trials", 50)
            objective = kwargs.get("objective", "accuracy")
            
            if not model_id:
                return CommandResult(
                    success=False,
                    message="Требуется model_id",
                    error=CommandError("MISSING_PARAMETERS", "model_id обязателен")
                )
            
            if not hasattr(context, 'ml_models') or model_id not in context.ml_models:
                return CommandResult(
                    success=False,
                    message=f"Модель {model_id} не найдена",
                    error=CommandError("MODEL_NOT_FOUND", f"Модель {model_id} не существует")
                )
            
            model = context.ml_models[model_id]
            
            # Выполняем оптимизацию (симуляция)
            optimization_history = []
            best_score = 0.0
            best_params = {}
            
            for trial in range(max_trials):
                # Генерируем случайные параметры из пространства поиска
                trial_params = {}
                for param_name, param_range in parameter_space.items():
                    if isinstance(param_range, list) and len(param_range) == 2:
                        if isinstance(param_range[0], float):
                            trial_params[param_name] = np.random.uniform(param_range[0], param_range[1])
                        else:
                            trial_params[param_name] = np.random.randint(param_range[0], param_range[1])
                    elif isinstance(param_range, list):
                        trial_params[param_name] = np.random.choice(param_range)
                
                # Симуляция оценки параметров
                if objective == "accuracy":
                    score = np.random.uniform(0.6, 0.95)
                elif objective == "loss":
                    score = np.random.uniform(0.01, 0.5)
                else:
                    score = np.random.uniform(0.5, 0.9)
                
                trial_data = {
                    "trial": trial + 1,
                    "parameters": trial_params,
                    "score": score,
                    "objective": objective,
                    "timestamp": time.time()
                }
                
                optimization_history.append(trial_data)
                
                # Обновляем лучший результат
                if (objective == "loss" and score < best_score) or (objective != "loss" and score > best_score):
                    best_score = score
                    best_params = trial_params.copy()
            
            # Обновляем параметры модели
            model.parameters.update(best_params)
            model.updated_at = time.time()
            
            # Сохраняем историю оптимизации
            if not hasattr(context, 'optimization_history'):
                context.optimization_history = {}
            context.optimization_history[model_id] = optimization_history
            
            return CommandResult(
                success=True,
                message=f"Оптимизация модели {model_id} завершена за {max_trials} попыток",
                data={
                    "model_id": model_id,
                    "best_score": best_score,
                    "best_parameters": best_params,
                    "optimization_method": optimization_method.value,
                    "trials_completed": max_trials,
                    "objective": objective,
                    "improvement": best_score - optimization_history[0]["score"],
                    "history": optimization_history[-10:]  # Последние 10 попыток
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка оптимизации: {str(e)}",
                error=CommandError("OPTIMIZATION_ERROR", str(e))
            )


class CreateModelCommand(MLOperationsCommand):
    """Команда создания модели"""
    
    def __init__(self):
        super().__init__(
            name="create_model",
            description="Создает новую модель машинного обучения",
            parameters={
                "name": "Название модели",
                "model_type": "Тип модели",
                "architecture": "Архитектура модели",
                "parameters": "Параметры модели"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            name = kwargs.get("name", f"model_{uuid.uuid4().hex[:8]}")
            model_type = ModelType(kwargs.get("model_type", "neural_network"))
            architecture = kwargs.get("architecture", {})
            parameters = kwargs.get("parameters", {})
            
            # Создаем новую модель
            model_id = f"model_{uuid.uuid4().hex[:8]}"
            model = MLModel(
                id=model_id,
                name=name,
                model_type=model_type,
                parameters=parameters,
                architecture=architecture
            )
            
            # Сохраняем модель
            if not hasattr(context, 'ml_models'):
                context.ml_models = {}
            context.ml_models[model_id] = model
            
            return CommandResult(
                success=True,
                message=f"Модель {name} создана с ID {model_id}",
                data={
                    "model_id": model_id,
                    "name": name,
                    "model_type": model_type.value,
                    "parameters": parameters,
                    "architecture": architecture,
                    "created_at": model.created_at
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка создания модели: {str(e)}",
                error=CommandError("MODEL_CREATION_ERROR", str(e))
            )


class LoadDataCommand(MLOperationsCommand):
    """Команда загрузки данных"""
    
    def __init__(self):
        super().__init__(
            name="load_data",
            description="Загружает набор данных для обучения",
            parameters={
                "source": "Источник данных (file, url, generator)",
                "format": "Формат данных (csv, json, numpy, pandas)",
                "preprocessing": "Шаги предобработки",
                "validation_split": "Доля данных для валидации"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            source = kwargs.get("source")
            data_format = kwargs.get("format", "numpy")
            preprocessing = kwargs.get("preprocessing", [])
            validation_split = kwargs.get("validation_split", 0.2)
            
            if not source:
                return CommandResult(
                    success=False,
                    message="Требуется источник данных",
                    error=CommandError("MISSING_PARAMETERS", "source обязателен")
                )
            
            # Генерируем или загружаем данные (симуляция)
            dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
            
            if source == "generator" or source.startswith("synthetic"):
                # Генерируем синтетические данные
                n_samples = kwargs.get("n_samples", 1000)
                n_features = kwargs.get("n_features", 10)
                
                features = np.random.randn(n_samples, n_features)
                targets = np.random.randint(0, 3, n_samples)  # 3 класса
                feature_names = [f"feature_{i}" for i in range(n_features)]
                target_names = ["class_0", "class_1", "class_2"]
                
            else:
                # Симуляция загрузки из файла
                features = np.random.randn(500, 8)
                targets = np.random.randint(0, 2, 500)
                feature_names = [f"feature_{i}" for i in range(8)]
                target_names = ["negative", "positive"]
            
            # Создаем набор данных
            dataset = Dataset(
                id=dataset_id,
                name=f"dataset_from_{source}",
                features=features,
                targets=targets,
                feature_names=feature_names,
                target_names=target_names,
                preprocessing_steps=preprocessing
            )
            
            # Сохраняем набор данных
            if not hasattr(context, 'datasets'):
                context.datasets = {}
            context.datasets[dataset_id] = dataset
            
            return CommandResult(
                success=True,
                message=f"Набор данных загружен с ID {dataset_id}",
                data={
                    "dataset_id": dataset_id,
                    "source": source,
                    "format": data_format,
                    "n_samples": len(features),
                    "n_features": features.shape[1] if len(features.shape) > 1 else 1,
                    "n_classes": len(np.unique(targets)) if targets is not None else None,
                    "feature_names": feature_names,
                    "target_names": target_names,
                    "preprocessing_steps": preprocessing
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка загрузки данных: {str(e)}",
                error=CommandError("DATA_LOADING_ERROR", str(e))
            )


class SaveModelCommand(MLOperationsCommand):
    """Команда сохранения модели"""
    
    def __init__(self):
        super().__init__(
            name="save_model",
            description="Сохраняет обученную модель",
            parameters={
                "model_id": "Идентификатор модели",
                "path": "Путь для сохранения",
                "format": "Формат сохранения (pickle, json, onnx)",
                "include_history": "Включить историю обучения"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            model_id = kwargs.get("model_id")
            save_path = kwargs.get("path", f"models/{model_id}.pkl")
            save_format = kwargs.get("format", "pickle")
            include_history = kwargs.get("include_history", True)
            
            if not model_id:
                return CommandResult(
                    success=False,
                    message="Требуется model_id",
                    error=CommandError("MISSING_PARAMETERS", "model_id обязателен")
                )
            
            if not hasattr(context, 'ml_models') or model_id not in context.ml_models:
                return CommandResult(
                    success=False,
                    message=f"Модель {model_id} не найдена",
                    error=CommandError("MODEL_NOT_FOUND", f"Модель {model_id} не существует")
                )
            
            model = context.ml_models[model_id]
            
            # Подготавливаем данные для сохранения
            save_data = {
                "model_id": model.id,
                "name": model.name,
                "model_type": model.model_type.value,
                "parameters": model.parameters,
                "architecture": model.architecture,
                "status": model.status.value,
                "accuracy": model.accuracy,
                "loss": model.loss,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "metadata": model.metadata
            }
            
            if include_history:
                save_data["training_history"] = model.training_history
            
            if model.weights is not None:
                save_data["weights"] = model.weights.tolist()
            
            # Создаем директорию если не существует
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем в зависимости от формата
            if save_format == "pickle":
                with open(save_path, 'wb') as f:
                    pickle.dump(save_data, f)
            elif save_format == "json":
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
            else:
                # Для других форматов просто сохраняем как JSON
                with open(save_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
            
            return CommandResult(
                success=True,
                message=f"Модель {model_id} сохранена в {save_path}",
                data={
                    "model_id": model_id,
                    "save_path": save_path,
                    "format": save_format,
                    "file_size": Path(save_path).stat().st_size if Path(save_path).exists() else 0,
                    "include_history": include_history,
                    "saved_at": time.time()
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка сохранения модели: {str(e)}",
                error=CommandError("MODEL_SAVE_ERROR", str(e))
            )


class LoadModelCommand(MLOperationsCommand):
    """Команда загрузки модели"""
    
    def __init__(self):
        super().__init__(
            name="load_model",
            description="Загружает сохраненную модель",
            parameters={
                "path": "Путь к файлу модели",
                "format": "Формат файла модели",
                "model_id": "Новый идентификатор модели (опционально)"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            load_path = kwargs.get("path")
            load_format = kwargs.get("format", "pickle")
            new_model_id = kwargs.get("model_id")
            
            if not load_path:
                return CommandResult(
                    success=False,
                    message="Требуется путь к файлу модели",
                    error=CommandError("MISSING_PARAMETERS", "path обязателен")
                )
            
            if not Path(load_path).exists():
                return CommandResult(
                    success=False,
                    message=f"Файл {load_path} не найден",
                    error=CommandError("FILE_NOT_FOUND", f"Файл {load_path} не существует")
                )
            
            # Загружаем данные
            if load_format == "pickle":
                with open(load_path, 'rb') as f:
                    save_data = pickle.load(f)
            elif load_format == "json":
                with open(load_path, 'r') as f:
                    save_data = json.load(f)
            else:
                with open(load_path, 'r') as f:
                    save_data = json.load(f)
            
            # Восстанавливаем модель
            model_id = new_model_id or save_data.get("model_id", f"model_{uuid.uuid4().hex[:8]}")
            
            model = MLModel(
                id=model_id,
                name=save_data.get("name", "loaded_model"),
                model_type=ModelType(save_data.get("model_type", "neural_network")),
                parameters=save_data.get("parameters", {}),
                architecture=save_data.get("architecture"),
                training_history=save_data.get("training_history", []),
                metadata=save_data.get("metadata", {}),
                created_at=save_data.get("created_at", time.time()),
                updated_at=save_data.get("updated_at", time.time()),
                status=TrainingStatus(save_data.get("status", "not_started")),
                accuracy=save_data.get("accuracy"),
                loss=save_data.get("loss")
            )
            
            # Восстанавливаем веса если есть
            if "weights" in save_data:
                model.weights = np.array(save_data["weights"])
            
            # Сохраняем модель в контексте
            if not hasattr(context, 'ml_models'):
                context.ml_models = {}
            context.ml_models[model_id] = model
            
            return CommandResult(
                success=True,
                message=f"Модель загружена с ID {model_id}",
                data={
                    "model_id": model_id,
                    "name": model.name,
                    "model_type": model.model_type.value,
                    "status": model.status.value,
                    "accuracy": model.accuracy,
                    "loss": model.loss,
                    "load_path": load_path,
                    "loaded_at": time.time(),
                    "has_weights": model.weights is not None,
                    "training_epochs": len(model.training_history)
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка загрузки модели: {str(e)}",
                error=CommandError("MODEL_LOAD_ERROR", str(e))
            )


class VisualizeCommand(MLOperationsCommand):
    """Команда визуализации данных и результатов"""
    
    def __init__(self):
        super().__init__(
            name="visualize",
            description="Создает визуализации данных и результатов обучения",
            parameters={
                "target": "Цель визуализации (model, dataset, training, predictions)",
                "target_id": "Идентификатор цели",
                "plot_type": "Тип графика (line, scatter, histogram, heatmap)",
                "save_path": "Путь для сохранения графика"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            target = kwargs.get("target", "training")
            target_id = kwargs.get("target_id")
            plot_type = kwargs.get("plot_type", "line")
            save_path = kwargs.get("save_path")
            
            visualization_data = {
                "target": target,
                "target_id": target_id,
                "plot_type": plot_type,
                "created_at": time.time(),
                "visualization_id": f"viz_{uuid.uuid4().hex[:8]}"
            }
            
            if target == "training" and target_id:
                if hasattr(context, 'ml_models') and target_id in context.ml_models:
                    model = context.ml_models[target_id]
                    if model.training_history:
                        # Данные для графика обучения
                        epochs = [h["epoch"] for h in model.training_history]
                        losses = [h["loss"] for h in model.training_history]
                        accuracies = [h["accuracy"] for h in model.training_history]
                        
                        visualization_data.update({
                            "epochs": epochs,
                            "losses": losses,
                            "accuracies": accuracies,
                            "final_loss": losses[-1] if losses else None,
                            "final_accuracy": accuracies[-1] if accuracies else None
                        })
            
            elif target == "dataset" and target_id:
                if hasattr(context, 'datasets') and target_id in context.datasets:
                    dataset = context.datasets[target_id]
                    visualization_data.update({
                        "n_samples": len(dataset.features),
                        "n_features": dataset.features.shape[1] if len(dataset.features.shape) > 1 else 1,
                        "feature_names": dataset.feature_names,
                        "data_shape": dataset.features.shape
                    })
            
            # Сохраняем информацию о визуализации
            if not hasattr(context, 'visualizations'):
                context.visualizations = {}
            context.visualizations[visualization_data["visualization_id"]] = visualization_data
            
            return CommandResult(
                success=True,
                message=f"Визуализация {target} создана",
                data=visualization_data
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка визуализации: {str(e)}",
                error=CommandError("VISUALIZATION_ERROR", str(e))
            )


# Регистрируем все команды ML операций
ML_COMMANDS = [
    TrainCommand(),
    PredictCommand(),
    EvaluateCommand(),
    OptimizeCommand(),
    CreateModelCommand(),
    LoadDataCommand(),
    SaveModelCommand(),
    LoadModelCommand(),
    VisualizeCommand()
]

# Экспортируем команды для использования в других модулях
__all__ = [
    'MLModel', 'TrainingConfig', 'Dataset', 'ModelType', 'TrainingStatus', 'OptimizationType',
    'TrainCommand', 'PredictCommand', 'EvaluateCommand', 'OptimizeCommand',
    'CreateModelCommand', 'LoadDataCommand', 'SaveModelCommand', 'LoadModelCommand',
    'VisualizeCommand', 'ML_COMMANDS'
]
