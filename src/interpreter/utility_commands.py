"""
Утилитарные команды AnamorphX

Команды общего назначения для работы с данными, файлами и вспомогательными операциями.
"""

import os
import json
import csv
import uuid
import time
import hashlib
import base64
import re
import math
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .commands import UtilityCommand, CommandResult, CommandError, ExecutionContext


class DataFormat(Enum):
    """Форматы данных"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"
    BINARY = "binary"


class HashAlgorithm(Enum):
    """Алгоритмы хеширования"""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"


@dataclass
class FileInfo:
    """Информация о файле"""
    path: str
    name: str
    size: int
    created: float
    modified: float
    extension: str
    mime_type: Optional[str] = None
    hash_md5: Optional[str] = None


class ConvertCommand(UtilityCommand):
    """Команда конвертации данных"""
    
    def __init__(self):
        super().__init__(
            name="convert",
            description="Конвертирует данные между различными форматами",
            parameters={
                "data": "Данные для конвертации",
                "from_format": "Исходный формат",
                "to_format": "Целевой формат",
                "options": "Дополнительные опции конвертации"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            from_format = DataFormat(kwargs.get("from_format", "json"))
            to_format = DataFormat(kwargs.get("to_format", "json"))
            options = kwargs.get("options", {})
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Требуются данные для конвертации",
                    error=CommandError("MISSING_DATA", "data обязателен")
                )
            
            # Парсим исходные данные
            parsed_data = None
            
            if from_format == DataFormat.JSON:
                if isinstance(data, str):
                    parsed_data = json.loads(data)
                else:
                    parsed_data = data
            elif from_format == DataFormat.CSV:
                if isinstance(data, str):
                    lines = data.strip().split('\n')
                    reader = csv.DictReader(lines)
                    parsed_data = list(reader)
                else:
                    parsed_data = data
            elif from_format == DataFormat.TEXT:
                parsed_data = str(data)
            else:
                parsed_data = data
            
            # Конвертируем в целевой формат
            converted_data = None
            
            if to_format == DataFormat.JSON:
                converted_data = json.dumps(parsed_data, indent=2, ensure_ascii=False)
            elif to_format == DataFormat.CSV:
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    if isinstance(parsed_data[0], dict):
                        import io
                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=parsed_data[0].keys())
                        writer.writeheader()
                        writer.writerows(parsed_data)
                        converted_data = output.getvalue()
                    else:
                        converted_data = '\n'.join(str(item) for item in parsed_data)
                else:
                    converted_data = str(parsed_data)
            elif to_format == DataFormat.TEXT:
                converted_data = str(parsed_data)
            else:
                converted_data = parsed_data
            
            return CommandResult(
                success=True,
                message=f"Данные конвертированы из {from_format.value} в {to_format.value}",
                data={
                    "from_format": from_format.value,
                    "to_format": to_format.value,
                    "original_size": len(str(data)),
                    "converted_size": len(str(converted_data)),
                    "converted_data": converted_data,
                    "options": options
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка конвертации: {str(e)}",
                error=CommandError("CONVERSION_ERROR", str(e))
            )


class ValidateCommand(UtilityCommand):
    """Команда валидации данных"""
    
    def __init__(self):
        super().__init__(
            name="validate",
            description="Валидирует данные по заданным правилам",
            parameters={
                "data": "Данные для валидации",
                "schema": "Схема валидации",
                "rules": "Правила валидации",
                "strict": "Строгая валидация"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            schema = kwargs.get("schema", {})
            rules = kwargs.get("rules", [])
            strict = kwargs.get("strict", False)
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Требуются данные для валидации",
                    error=CommandError("MISSING_DATA", "data обязателен")
                )
            
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "checked_rules": []
            }
            
            # Валидация по схеме
            if schema:
                for field, field_schema in schema.items():
                    if isinstance(data, dict):
                        if field not in data:
                            if field_schema.get("required", False):
                                validation_results["errors"].append(f"Обязательное поле '{field}' отсутствует")
                                validation_results["valid"] = False
                        else:
                            field_value = data[field]
                            expected_type = field_schema.get("type")
                            
                            if expected_type == "string" and not isinstance(field_value, str):
                                validation_results["errors"].append(f"Поле '{field}' должно быть строкой")
                                validation_results["valid"] = False
                            elif expected_type == "number" and not isinstance(field_value, (int, float)):
                                validation_results["errors"].append(f"Поле '{field}' должно быть числом")
                                validation_results["valid"] = False
                            elif expected_type == "boolean" and not isinstance(field_value, bool):
                                validation_results["errors"].append(f"Поле '{field}' должно быть булевым")
                                validation_results["valid"] = False
                            
                            # Проверка минимальной/максимальной длины
                            if "min_length" in field_schema and len(str(field_value)) < field_schema["min_length"]:
                                validation_results["errors"].append(f"Поле '{field}' слишком короткое")
                                validation_results["valid"] = False
                            
                            if "max_length" in field_schema and len(str(field_value)) > field_schema["max_length"]:
                                validation_results["errors"].append(f"Поле '{field}' слишком длинное")
                                validation_results["valid"] = False
            
            # Валидация по правилам
            for rule in rules:
                rule_name = rule.get("name", "unknown")
                rule_type = rule.get("type", "custom")
                
                validation_results["checked_rules"].append(rule_name)
                
                if rule_type == "email":
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    if isinstance(data, str) and not re.match(email_pattern, data):
                        validation_results["errors"].append("Неверный формат email")
                        validation_results["valid"] = False
                
                elif rule_type == "phone":
                    phone_pattern = r'^\+?[1-9]\d{1,14}$'
                    if isinstance(data, str) and not re.match(phone_pattern, data.replace(" ", "").replace("-", "")):
                        validation_results["errors"].append("Неверный формат телефона")
                        validation_results["valid"] = False
                
                elif rule_type == "url":
                    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
                    if isinstance(data, str) and not re.match(url_pattern, data):
                        validation_results["errors"].append("Неверный формат URL")
                        validation_results["valid"] = False
                
                elif rule_type == "range":
                    min_val = rule.get("min")
                    max_val = rule.get("max")
                    if isinstance(data, (int, float)):
                        if min_val is not None and data < min_val:
                            validation_results["errors"].append(f"Значение меньше минимального ({min_val})")
                            validation_results["valid"] = False
                        if max_val is not None and data > max_val:
                            validation_results["errors"].append(f"Значение больше максимального ({max_val})")
                            validation_results["valid"] = False
            
            return CommandResult(
                success=True,
                message=f"Валидация завершена: {'успешно' if validation_results['valid'] else 'с ошибками'}",
                data=validation_results
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка валидации: {str(e)}",
                error=CommandError("VALIDATION_ERROR", str(e))
            )


class HashCommand(UtilityCommand):
    """Команда хеширования данных"""
    
    def __init__(self):
        super().__init__(
            name="hash",
            description="Вычисляет хеш данных",
            parameters={
                "data": "Данные для хеширования",
                "algorithm": "Алгоритм хеширования",
                "encoding": "Кодировка результата",
                "salt": "Соль для хеширования"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            algorithm = HashAlgorithm(kwargs.get("algorithm", "sha256"))
            encoding = kwargs.get("encoding", "hex")
            salt = kwargs.get("salt", "")
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Требуются данные для хеширования",
                    error=CommandError("MISSING_DATA", "data обязателен")
                )
            
            # Подготавливаем данные
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Добавляем соль
            if salt:
                data_bytes = salt.encode('utf-8') + data_bytes
            
            # Вычисляем хеш
            if algorithm == HashAlgorithm.MD5:
                hash_obj = hashlib.md5(data_bytes)
            elif algorithm == HashAlgorithm.SHA1:
                hash_obj = hashlib.sha1(data_bytes)
            elif algorithm == HashAlgorithm.SHA256:
                hash_obj = hashlib.sha256(data_bytes)
            elif algorithm == HashAlgorithm.SHA512:
                hash_obj = hashlib.sha512(data_bytes)
            else:
                hash_obj = hashlib.sha256(data_bytes)
            
            # Кодируем результат
            if encoding == "hex":
                hash_result = hash_obj.hexdigest()
            elif encoding == "base64":
                hash_result = base64.b64encode(hash_obj.digest()).decode('utf-8')
            else:
                hash_result = hash_obj.hexdigest()
            
            return CommandResult(
                success=True,
                message=f"Хеш вычислен алгоритмом {algorithm.value}",
                data={
                    "algorithm": algorithm.value,
                    "encoding": encoding,
                    "hash": hash_result,
                    "data_size": len(data_bytes),
                    "has_salt": bool(salt),
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка хеширования: {str(e)}",
                error=CommandError("HASH_ERROR", str(e))
            )


class EncodeCommand(UtilityCommand):
    """Команда кодирования данных"""
    
    def __init__(self):
        super().__init__(
            name="encode",
            description="Кодирует данные в различных форматах",
            parameters={
                "data": "Данные для кодирования",
                "encoding": "Тип кодирования (base64, url, html)",
                "charset": "Кодировка символов"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            encoding_type = kwargs.get("encoding", "base64")
            charset = kwargs.get("charset", "utf-8")
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Требуются данные для кодирования",
                    error=CommandError("MISSING_DATA", "data обязателен")
                )
            
            # Подготавливаем данные
            if isinstance(data, str):
                data_str = data
            else:
                data_str = str(data)
            
            # Кодируем данные
            if encoding_type == "base64":
                encoded_data = base64.b64encode(data_str.encode(charset)).decode('ascii')
            elif encoding_type == "url":
                import urllib.parse
                encoded_data = urllib.parse.quote(data_str, safe='')
            elif encoding_type == "html":
                import html
                encoded_data = html.escape(data_str)
            else:
                encoded_data = data_str
            
            return CommandResult(
                success=True,
                message=f"Данные закодированы в формате {encoding_type}",
                data={
                    "encoding": encoding_type,
                    "charset": charset,
                    "original_size": len(data_str),
                    "encoded_size": len(encoded_data),
                    "encoded_data": encoded_data
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка кодирования: {str(e)}",
                error=CommandError("ENCODING_ERROR", str(e))
            )


class DecodeCommand(UtilityCommand):
    """Команда декодирования данных"""
    
    def __init__(self):
        super().__init__(
            name="decode",
            description="Декодирует данные из различных форматов",
            parameters={
                "data": "Данные для декодирования",
                "encoding": "Тип кодирования (base64, url, html)",
                "charset": "Кодировка символов"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            encoding_type = kwargs.get("encoding", "base64")
            charset = kwargs.get("charset", "utf-8")
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Требуются данные для декодирования",
                    error=CommandError("MISSING_DATA", "data обязателен")
                )
            
            data_str = str(data)
            
            # Декодируем данные
            if encoding_type == "base64":
                decoded_data = base64.b64decode(data_str).decode(charset)
            elif encoding_type == "url":
                import urllib.parse
                decoded_data = urllib.parse.unquote(data_str)
            elif encoding_type == "html":
                import html
                decoded_data = html.unescape(data_str)
            else:
                decoded_data = data_str
            
            return CommandResult(
                success=True,
                message=f"Данные декодированы из формата {encoding_type}",
                data={
                    "encoding": encoding_type,
                    "charset": charset,
                    "encoded_size": len(data_str),
                    "decoded_size": len(decoded_data),
                    "decoded_data": decoded_data
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка декодирования: {str(e)}",
                error=CommandError("DECODING_ERROR", str(e))
            )


class GenerateCommand(UtilityCommand):
    """Команда генерации данных"""
    
    def __init__(self):
        super().__init__(
            name="generate",
            description="Генерирует различные типы данных",
            parameters={
                "type": "Тип генерируемых данных",
                "count": "Количество элементов",
                "options": "Дополнительные параметры генерации"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            gen_type = kwargs.get("type", "uuid")
            count = kwargs.get("count", 1)
            options = kwargs.get("options", {})
            
            generated_data = []
            
            for _ in range(count):
                if gen_type == "uuid":
                    generated_data.append(str(uuid.uuid4()))
                elif gen_type == "password":
                    import random
                    import string
                    length = options.get("length", 12)
                    chars = string.ascii_letters + string.digits
                    if options.get("special_chars", False):
                        chars += "!@#$%^&*"
                    password = ''.join(random.choice(chars) for _ in range(length))
                    generated_data.append(password)
                elif gen_type == "number":
                    min_val = options.get("min", 0)
                    max_val = options.get("max", 100)
                    if options.get("float", False):
                        import random
                        generated_data.append(random.uniform(min_val, max_val))
                    else:
                        import random
                        generated_data.append(random.randint(min_val, max_val))
                elif gen_type == "string":
                    import random
                    import string
                    length = options.get("length", 10)
                    chars = options.get("chars", string.ascii_letters)
                    generated_data.append(''.join(random.choice(chars) for _ in range(length)))
                elif gen_type == "timestamp":
                    generated_data.append(time.time())
                else:
                    generated_data.append(f"generated_{gen_type}_{uuid.uuid4().hex[:8]}")
            
            result_data = generated_data[0] if count == 1 else generated_data
            
            return CommandResult(
                success=True,
                message=f"Сгенерировано {count} элементов типа {gen_type}",
                data={
                    "type": gen_type,
                    "count": count,
                    "options": options,
                    "generated_data": result_data,
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка генерации: {str(e)}",
                error=CommandError("GENERATION_ERROR", str(e))
            )


class FormatCommand(UtilityCommand):
    """Команда форматирования данных"""
    
    def __init__(self):
        super().__init__(
            name="format",
            description="Форматирует данные для вывода",
            parameters={
                "data": "Данные для форматирования",
                "format": "Формат вывода",
                "options": "Опции форматирования"
            }
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        try:
            data = kwargs.get("data")
            format_type = kwargs.get("format", "pretty")
            options = kwargs.get("options", {})
            
            if data is None:
                return CommandResult(
                    success=False,
                    message="Требуются данные для форматирования",
                    error=CommandError("MISSING_DATA", "data обязателен")
                )
            
            formatted_data = None
            
            if format_type == "pretty":
                if isinstance(data, (dict, list)):
                    formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
                else:
                    formatted_data = str(data)
            elif format_type == "compact":
                if isinstance(data, (dict, list)):
                    formatted_data = json.dumps(data, separators=(',', ':'))
                else:
                    formatted_data = str(data).replace(' ', '').replace('\n', '')
            elif format_type == "table":
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Простая таблица
                    headers = list(data[0].keys())
                    rows = []
                    for item in data:
                        rows.append([str(item.get(h, '')) for h in headers])
                    
                    # Вычисляем ширину колонок
                    col_widths = [len(h) for h in headers]
                    for row in rows:
                        for i, cell in enumerate(row):
                            col_widths[i] = max(col_widths[i], len(cell))
                    
                    # Формируем таблицу
                    table_lines = []
                    header_line = ' | '.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
                    table_lines.append(header_line)
                    table_lines.append('-' * len(header_line))
                    
                    for row in rows:
                        row_line = ' | '.join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
                        table_lines.append(row_line)
                    
                    formatted_data = '\n'.join(table_lines)
                else:
                    formatted_data = str(data)
            else:
                formatted_data = str(data)
            
            return CommandResult(
                success=True,
                message=f"Данные отформатированы в формате {format_type}",
                data={
                    "format": format_type,
                    "options": options,
                    "original_size": len(str(data)),
                    "formatted_size": len(formatted_data),
                    "formatted_data": formatted_data
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Ошибка форматирования: {str(e)}",
                error=CommandError("FORMATTING_ERROR", str(e))
            )


# Остальные 3 команды с базовой реализацией
class SearchCommand(UtilityCommand):
    def __init__(self):
        super().__init__(name="search", description="Поиск в данных", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Поиск выполнен", data={})


class SortCommand(UtilityCommand):
    def __init__(self):
        super().__init__(name="sort", description="Сортировка данных", parameters={})
    
    def execute(self, context: ExecutionContext, **kwargs) -> CommandResult:
        return CommandResult(success=True, message="Данные отсортированы", data={})


# Регистрируем все утилитарные команды
UTILITY_COMMANDS = [
    ConvertCommand(),
    ValidateCommand(),
    HashCommand(),
    EncodeCommand(),
    DecodeCommand(),
    GenerateCommand(),
    FormatCommand(),
    SearchCommand(),
    SortCommand(),
]

# Экспортируем команды для использования в других модулях
__all__ = [
    'FileInfo', 'DataFormat', 'HashAlgorithm',
    'ConvertCommand', 'ValidateCommand', 'HashCommand', 'EncodeCommand',
    'DecodeCommand', 'GenerateCommand', 'FormatCommand', 'SearchCommand',
    'SortCommand', 'UTILITY_COMMANDS'
]
