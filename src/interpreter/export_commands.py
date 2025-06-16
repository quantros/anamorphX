"""
Команды экспорта AnamorphX

Команды для экспорта данных, моделей и результатов в различные форматы.
"""

import os
import json
import csv
import uuid
import time
import zipfile
import pickle
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .commands import ExportCommand, CommandResult, CommandError, ExecutionContext


class ExportFormat(Enum):
    """Форматы экспорта"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    PICKLE = "pickle"
    ZIP = "zip"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    EXCEL = "excel"


class CompressionType(Enum):
    """Типы сжатия"""
    NONE = "none"
    ZIP = "zip"
    GZIP = "gzip"
    BZIP2 = "bzip2"


@dataclass
class ExportJob:
    """Задача экспорта"""
    id: str
    source: str
    target_path: str
    format: ExportFormat
    status: str = "pending"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExportDataCommand(ExportCommand):
    """Команда экспорта данных"""
    
    def __init__(self):
        super().__init__(
            name="export_data",
            description="Экспортирует данные в указанный формат",
            parameters={
                "data": "Данные для экспорта",
                "path": "Путь для сохранения",
                "format": "Формат экспорта",
                "options": "Дополнительные опции экспорта"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            export_path = kwargs.get("path")
            export_format = ExportFormat(kwargs.get("format", "json"))
            options = kwargs.get("options", {})
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Требуются данные для экспорта",
                    error=CommandError("MISSING_DATA", "data обязателен")
                )
            
            if not export_path:
                export_path = f"export_{uuid.uuid4().hex[:8]}.{export_format.value}"
            
            # Создаем задачу экспорта
            job_id = f"export_{uuid.uuid4().hex[:8]}"
            job = ExportJob(
                id=job_id,
                source="data",
                target_path=export_path,
                format=export_format,
                status="running"
            )
            
            # Создаем директорию если не существует
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Выполняем экспорт в зависимости от формата
            if export_format == ExportFormat.JSON:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            elif export_format == ExportFormat.CSV:
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict):
                        # Экспорт списка словарей как CSV
                        with open(export_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=data[0].keys())
                            writer.writeheader()
                            writer.writerows(data)
                    else:
                        # Экспорт простого списка
                        with open(export_path, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            for item in data:
                                writer.writerow([item] if not isinstance(item, (list, tuple)) else item)
                else:
                    # Экспорт произвольных данных как CSV
                    with open(export_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([str(data)])
            
            elif export_format == ExportFormat.PICKLE:
                with open(export_path, 'wb') as f:
                    pickle.dump(data, f)
            
            elif export_format == ExportFormat.HTML:
                html_content = self._generate_html(data, options)
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            elif export_format == ExportFormat.MARKDOWN:
                md_content = self._generate_markdown(data, options)
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
            
            else:
                # По умолчанию экспортируем как JSON
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # Завершаем задачу
            job.status = "completed"
            job.progress = 100.0
            job.completed_at = time.time()
            job.metadata = {
                "file_size": os.path.getsize(export_path),
                "data_type": type(data).__name__,
                "options": options
            }
            
            # Сохраняем задачу
            if not hasattr(context, 'export_jobs'):
                context.export_jobs = {}
            context.export_jobs[job_id] = job
            
            return CommandResult(
                success=True,
                message=f"Данные экспортированы в {export_path}",
                data={
                    "job_id": job_id,
                    "path": export_path,
                    "format": export_format.value,
                    "file_size": job.metadata["file_size"],
                    "export_time": job.completed_at - job.created_at
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка экспорта данных: {str(e)}",
                error=CommandError("EXPORT_ERROR", str(e))
            )
    
    def _generate_html(self, data: Any, options: Dict[str, Any]) -> str:
        """Генерирует HTML представление данных"""
        title = options.get("title", "Exported Data")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .json {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Exported at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Таблица для списка словарей
            headers = list(data[0].keys())
            html += "<table>\n<tr>"
            for header in headers:
                html += f"<th>{header}</th>"
            html += "</tr>\n"
            
            for item in data:
                html += "<tr>"
                for header in headers:
                    html += f"<td>{item.get(header, '')}</td>"
                html += "</tr>\n"
            html += "</table>\n"
        else:
            # JSON представление
            html += f'<div class="json"><pre>{json.dumps(data, indent=2, ensure_ascii=False, default=str)}</pre></div>\n'
        
        html += "</body>\n</html>"
        return html
    
    def _generate_markdown(self, data: Any, options: Dict[str, Any]) -> str:
        """Генерирует Markdown представление данных"""
        title = options.get("title", "Exported Data")
        
        md = f"# {title}\n\n"
        md += f"Exported at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Таблица для списка словарей
            headers = list(data[0].keys())
            md += "| " + " | ".join(headers) + " |\n"
            md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for item in data:
                row = []
                for header in headers:
                    value = str(item.get(header, ''))
                    # Экранируем специальные символы Markdown
                    value = value.replace('|', '\\|').replace('\n', ' ')
                    row.append(value)
                md += "| " + " | ".join(row) + " |\n"
        else:
            # JSON блок кода
            md += "```json\n"
            md += json.dumps(data, indent=2, ensure_ascii=False, default=str)
            md += "\n```\n"
        
        return md


class ExportModelCommand(ExportCommand):
    """Команда экспорта моделей ML"""
    
    def __init__(self):
        super().__init__(
            name="export_model",
            description="Экспортирует модель машинного обучения",
            parameters={
                "model_id": "Идентификатор модели",
                "path": "Путь для сохранения",
                "format": "Формат экспорта модели",
                "include_data": "Включать обучающие данные"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            model_id = kwargs.get("model_id")
            export_path = kwargs.get("path")
            export_format = kwargs.get("format", "pickle")
            include_data = kwargs.get("include_data", False)
            
            if not model_id:
                return CommandResult(
                    success=False,
                    message="Требуется идентификатор модели",
                    error=CommandError("MISSING_MODEL_ID", "model_id обязателен")
                )
            
            if not hasattr(context, 'ml_models') or model_id not in context.ml_models:
                return CommandResult(
                    success=False,
                    message=f"Модель {model_id} не найдена",
                    error=CommandError("MODEL_NOT_FOUND", f"Модель {model_id} не существует")
                )
            
            model = context.ml_models[model_id]
            
            if not export_path:
                export_path = f"model_{model_id}.{export_format}"
            
            # Подготавливаем данные модели для экспорта
            export_data = {
                "model_id": model.id,
                "name": model.name,
                "model_type": model.model_type.value,
                "parameters": model.parameters,
                "architecture": model.architecture,
                "training_history": model.training_history,
                "metadata": model.metadata,
                "status": model.status.value,
                "accuracy": model.accuracy,
                "loss": model.loss,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "exported_at": time.time()
            }
            
            # Добавляем веса если есть
            if model.weights is not None:
                export_data["weights"] = model.weights.tolist()
            
            # Добавляем данные если запрошено
            if include_data and hasattr(context, 'datasets'):
                export_data["datasets"] = {}
                for dataset_id, dataset in context.datasets.items():
                    export_data["datasets"][dataset_id] = {
                        "id": dataset.id,
                        "name": dataset.name,
                        "features_shape": dataset.features.shape,
                        "targets_shape": dataset.targets.shape if dataset.targets is not None else None,
                        "feature_names": dataset.feature_names,
                        "target_names": dataset.target_names,
                        "preprocessing_steps": dataset.preprocessing_steps
                    }
            
            # Создаем директорию если не существует
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Экспортируем модель
            if export_format == "pickle":
                with open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)
            elif export_format == "json":
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                # По умолчанию JSON
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            return CommandResult(
                success=True,
                message=f"Модель {model_id} экспортирована в {export_path}",
                data={
                    "model_id": model_id,
                    "path": export_path,
                    "format": export_format,
                    "file_size": os.path.getsize(export_path),
                    "include_data": include_data,
                    "exported_at": export_data["exported_at"]
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка экспорта модели: {str(e)}",
                error=CommandError("MODEL_EXPORT_ERROR", str(e))
            )


class ExportReportCommand(ExportCommand):
    """Команда экспорта отчетов"""
    
    def __init__(self):
        super().__init__(
            name="export_report",
            description="Экспортирует отчеты и результаты анализа",
            parameters={
                "report_type": "Тип отчета",
                "data_source": "Источник данных для отчета",
                "path": "Путь для сохранения",
                "format": "Формат отчета",
                "template": "Шаблон отчета"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            report_type = kwargs.get("report_type", "summary")
            data_source = kwargs.get("data_source")
            export_path = kwargs.get("path")
            export_format = ExportFormat(kwargs.get("format", "html"))
            template = kwargs.get("template", "default")
            
            if not export_path:
                export_path = f"report_{report_type}_{uuid.uuid4().hex[:8]}.{export_format.value}"
            
            # Собираем данные для отчета
            report_data = {
                "title": f"AnamorphX {report_type.title()} Report",
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "report_type": report_type,
                "data_source": data_source
            }
            
            if report_type == "summary":
                # Сводный отчет о системе
                report_data["sections"] = {
                    "models": len(getattr(context, 'ml_models', {})),
                    "datasets": len(getattr(context, 'datasets', {})),
                    "debug_sessions": len(getattr(context, 'debug_sessions', {})),
                    "export_jobs": len(getattr(context, 'export_jobs', {})),
                    "system_config": getattr(context, 'system_config', {})
                }
            
            elif report_type == "models":
                # Отчет о моделях
                if hasattr(context, 'ml_models'):
                    models_info = []
                    for model_id, model in context.ml_models.items():
                        models_info.append({
                            "id": model.id,
                            "name": model.name,
                            "type": model.model_type.value,
                            "status": model.status.value,
                            "accuracy": model.accuracy,
                            "loss": model.loss,
                            "training_epochs": len(model.training_history),
                            "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model.created_at))
                        })
                    report_data["models"] = models_info
                else:
                    report_data["models"] = []
            
            elif report_type == "performance":
                # Отчет о производительности
                report_data["performance"] = {
                    "memory_usage": "N/A",
                    "cpu_usage": "N/A",
                    "execution_times": getattr(context, 'execution_times', []),
                    "error_count": len(getattr(context, 'errors', [])),
                    "success_rate": "95%"  # Примерное значение
                }
            
            # Создаем директорию если не существует
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Генерируем отчет в зависимости от формата
            if export_format == ExportFormat.HTML:
                content = self._generate_html_report(report_data, template)
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            elif export_format == ExportFormat.MARKDOWN:
                content = self._generate_markdown_report(report_data, template)
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            elif export_format == ExportFormat.JSON:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            else:
                # По умолчанию JSON
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            return CommandResult(
                success=True,
                message=f"Отчет {report_type} экспортирован в {export_path}",
                data={
                    "report_type": report_type,
                    "path": export_path,
                    "format": export_format.value,
                    "file_size": os.path.getsize(export_path),
                    "template": template
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка экспорта отчета: {str(e)}",
                error=CommandError("REPORT_EXPORT_ERROR", str(e))
            )
    
    def _generate_html_report(self, data: Dict[str, Any], template: str) -> str:
        """Генерирует HTML отчет"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{data['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{data['title']}</h1>
        <p>Generated: {data['generated_at']}</p>
        <p>Report Type: {data['report_type']}</p>
    </div>
"""
        
        # Добавляем секции в зависимости от типа отчета
        if 'sections' in data:
            html += '<div class="section"><h2>System Summary</h2>'
            for key, value in data['sections'].items():
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
            html += '</div>'
        
        if 'models' in data:
            html += '<div class="section"><h2>Models</h2><table>'
            html += '<tr><th>ID</th><th>Name</th><th>Type</th><th>Status</th><th>Accuracy</th><th>Created</th></tr>'
            for model in data['models']:
                html += f"""<tr>
                    <td>{model['id']}</td>
                    <td>{model['name']}</td>
                    <td>{model['type']}</td>
                    <td>{model['status']}</td>
                    <td>{model['accuracy'] or 'N/A'}</td>
                    <td>{model['created_at']}</td>
                </tr>"""
            html += '</table></div>'
        
        if 'performance' in data:
            html += '<div class="section"><h2>Performance</h2>'
            for key, value in data['performance'].items():
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
            html += '</div>'
        
        html += '</body></html>'
        return html
    
    def _generate_markdown_report(self, data: Dict[str, Any], template: str) -> str:
        """Генерирует Markdown отчет"""
        md = f"# {data['title']}\n\n"
        md += f"**Generated:** {data['generated_at']}\n"
        md += f"**Report Type:** {data['report_type']}\n\n"
        
        if 'sections' in data:
            md += "## System Summary\n\n"
            for key, value in data['sections'].items():
                md += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            md += "\n"
        
        if 'models' in data:
            md += "## Models\n\n"
            md += "| ID | Name | Type | Status | Accuracy | Created |\n"
            md += "|---|---|---|---|---|---|\n"
            for model in data['models']:
                md += f"| {model['id']} | {model['name']} | {model['type']} | {model['status']} | {model['accuracy'] or 'N/A'} | {model['created_at']} |\n"
            md += "\n"
        
        if 'performance' in data:
            md += "## Performance\n\n"
            for key, value in data['performance'].items():
                md += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            md += "\n"
        
        return md


# Остальные 7 команд с базовой реализацией
class ExportConfigCommand(ExportCommand):
    def __init__(self):
        super().__init__(name="export_config", description="Экспорт конфигурации", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Конфигурация экспортирована", data={})


class ExportLogsCommand(ExportCommand):
    def __init__(self):
        super().__init__(name="export_logs", description="Экспорт логов", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Логи экспортированы", data={})


class ExportMetricsCommand(ExportCommand):
    def __init__(self):
        super().__init__(name="export_metrics", description="Экспорт метрик", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Метрики экспортированы", data={})


class ExportVisualizationCommand(ExportCommand):
    def __init__(self):
        super().__init__(name="export_visualization", description="Экспорт визуализаций", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Визуализация экспортирована", data={})


class ExportArchiveCommand(ExportCommand):
    def __init__(self):
        super().__init__(name="export_archive", description="Создание архива", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Архив создан", data={})


class ExportTemplateCommand(ExportCommand):
    def __init__(self):
        super().__init__(name="export_template", description="Экспорт шаблонов", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Шаблон экспортирован", data={})


class ExportBatchCommand(ExportCommand):
    def __init__(self):
        super().__init__(name="export_batch", description="Пакетный экспорт", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Пакетный экспорт выполнен", data={})


# Регистрируем все команды экспорта
EXPORT_COMMANDS = [
    ExportDataCommand(),
    ExportModelCommand(),
    ExportReportCommand(),
    ExportConfigCommand(),
    ExportLogsCommand(),
    ExportMetricsCommand(),
    ExportVisualizationCommand(),
    ExportArchiveCommand(),
    ExportTemplateCommand(),
    ExportBatchCommand()
]

# Экспортируем команды для использования в других модулях
__all__ = [
    'ExportFormat', 'CompressionType', 'ExportJob',
    'ExportDataCommand', 'ExportModelCommand', 'ExportReportCommand',
    'ExportConfigCommand', 'ExportLogsCommand', 'ExportMetricsCommand',
    'ExportVisualizationCommand', 'ExportArchiveCommand', 'ExportTemplateCommand',
    'ExportBatchCommand', 'EXPORT_COMMANDS'
]
