#!/usr/bin/env python3
"""
Система интернационализации (i18n) для AnamorphX IDE
Поддержка русского и английского языков
"""

import json
import os
from typing import Dict, Any

class I18nManager:
    """Менеджер интернационализации"""
    
    def __init__(self):
        self.current_language = "ru"  # По умолчанию русский
        self.translations = {}
        self.load_translations()
    
    def load_translations(self):
        """Загрузка переводов"""
        # Русские переводы
        self.translations["ru"] = {
            # Меню
            "menu_file": "Файл",
            "menu_edit": "Правка", 
            "menu_run": "Выполнение",
            "menu_debug": "Отладка",
            "menu_tools": "Инструменты",
            "menu_help": "Справка",
            "menu_language": "Язык",
            
            # Файл
            "file_new": "Новый",
            "file_open": "Открыть",
            "file_save": "Сохранить",
            "file_save_as": "Сохранить как...",
            "file_exit": "Выход",
            
            # Правка
            "edit_undo": "Отменить",
            "edit_redo": "Повторить",
            "edit_cut": "Вырезать",
            "edit_copy": "Копировать",
            "edit_paste": "Вставить",
            "edit_find": "Найти",
            "edit_replace": "Заменить",
            
            # Выполнение
            "run_execute": "Запустить",
            "run_debug": "Отладка",
            "run_profile": "Профилировать",
            "run_stop": "Остановить",
            
            # Отладка
            "debug_step": "Шаг",
            "debug_step_into": "Шаг в",
            "debug_step_out": "Шаг из",
            "debug_continue": "Продолжить",
            "debug_breakpoint": "Точка останова",
            "debug_clear_breakpoints": "Очистить точки останова",
            
            # Панели
            "panel_variables": "Переменные",
            "panel_call_stack": "Стек вызовов",
            "panel_profiler": "Профайлер",
            "panel_console": "Консоль",
            "panel_output": "Вывод",
            
            # Кнопки
            "btn_run": "▶ Запустить",
            "btn_debug": "🐛 Отладка",
            "btn_profile": "📊 Профилировать",
            "btn_stop": "⏹ Стоп",
            "btn_step": "▶ Шаг",
            "btn_step_into": "↳ Шаг в",
            "btn_step_out": "↰ Шаг из",
            "btn_continue": "⏭ Продолжить",
            "btn_refresh": "Обновить",
            "btn_add": "Добавить",
            "btn_execute": "Выполнить",
            
            # Статусы
            "status_ready": "Готов",
            "status_running": "Выполняется...",
            "status_debugging": "Отладка...",
            "status_profiling": "Профилирование...",
            
            # Сообщения
            "msg_execution_started": "🚀 Запуск программы...",
            "msg_debug_started": "🐛 Начало отладки...",
            "msg_profile_started": "📊 Начало профилирования...",
            "msg_execution_completed": "✅ Выполнение завершено успешно!",
            "msg_breakpoint_set": "Точка останова установлена на строке",
            "msg_breakpoint_removed": "Точка останова удалена на строке",
            
            # Заголовки колонок
            "col_name": "Имя",
            "col_value": "Значение", 
            "col_type": "Тип",
            "col_function": "Функция",
            "col_time": "Время",
            "col_calls": "Вызовы",
            
            # Позиция курсора
            "cursor_position": "Строка: {line}, Столбец: {col}",
            
            # Команды консоли
            "console_help": "Доступные команды:",
            "console_vars": "vars - показать все переменные",
            "console_break": "break - показать точки останова",
            "console_print": "print <var> - показать значение переменной",
            "console_help_cmd": "help - показать эту справку",
            "console_unknown": "Неизвестная команда:",
            "console_var_not_found": "Переменная '{var}' не найдена",
            
            # Языки
            "lang_russian": "Русский",
            "lang_english": "English"
        }
        
        # Английские переводы
        self.translations["en"] = {
            # Menu
            "menu_file": "File",
            "menu_edit": "Edit",
            "menu_run": "Run", 
            "menu_debug": "Debug",
            "menu_tools": "Tools",
            "menu_help": "Help",
            "menu_language": "Language",
            
            # File
            "file_new": "New",
            "file_open": "Open",
            "file_save": "Save",
            "file_save_as": "Save As...",
            "file_exit": "Exit",
            
            # Edit
            "edit_undo": "Undo",
            "edit_redo": "Redo",
            "edit_cut": "Cut",
            "edit_copy": "Copy",
            "edit_paste": "Paste",
            "edit_find": "Find",
            "edit_replace": "Replace",
            
            # Run
            "run_execute": "Execute",
            "run_debug": "Debug",
            "run_profile": "Profile",
            "run_stop": "Stop",
            
            # Debug
            "debug_step": "Step",
            "debug_step_into": "Step Into",
            "debug_step_out": "Step Out",
            "debug_continue": "Continue",
            "debug_breakpoint": "Breakpoint",
            "debug_clear_breakpoints": "Clear Breakpoints",
            
            # Panels
            "panel_variables": "Variables",
            "panel_call_stack": "Call Stack",
            "panel_profiler": "Profiler",
            "panel_console": "Console",
            "panel_output": "Output",
            
            # Buttons
            "btn_run": "▶ Run",
            "btn_debug": "🐛 Debug",
            "btn_profile": "📊 Profile",
            "btn_stop": "⏹ Stop",
            "btn_step": "▶ Step",
            "btn_step_into": "↳ Step Into",
            "btn_step_out": "↰ Step Out",
            "btn_continue": "⏭ Continue",
            "btn_refresh": "Refresh",
            "btn_add": "Add",
            "btn_execute": "Execute",
            
            # Status
            "status_ready": "Ready",
            "status_running": "Running...",
            "status_debugging": "Debugging...",
            "status_profiling": "Profiling...",
            
            # Messages
            "msg_execution_started": "🚀 Starting program execution...",
            "msg_debug_started": "🐛 Starting debug session...",
            "msg_profile_started": "📊 Starting profiling...",
            "msg_execution_completed": "✅ Execution completed successfully!",
            "msg_breakpoint_set": "Breakpoint set at line",
            "msg_breakpoint_removed": "Breakpoint removed at line",
            
            # Column headers
            "col_name": "Name",
            "col_value": "Value",
            "col_type": "Type", 
            "col_function": "Function",
            "col_time": "Time",
            "col_calls": "Calls",
            
            # Cursor position
            "cursor_position": "Line: {line}, Column: {col}",
            
            # Console commands
            "console_help": "Available commands:",
            "console_vars": "vars - show all variables",
            "console_break": "break - show breakpoints",
            "console_print": "print <var> - show variable value",
            "console_help_cmd": "help - show this help",
            "console_unknown": "Unknown command:",
            "console_var_not_found": "Variable '{var}' not found",
            
            # Languages
            "lang_russian": "Русский",
            "lang_english": "English"
        }
    
    def set_language(self, language_code: str):
        """Установка языка"""
        if language_code in self.translations:
            self.current_language = language_code
            return True
        return False
    
    def get_language(self) -> str:
        """Получение текущего языка"""
        return self.current_language
    
    def t(self, key: str, **kwargs) -> str:
        """Получение перевода (translate)"""
        translation = self.translations.get(self.current_language, {}).get(key, key)
        
        # Форматирование с параметрами
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except:
                pass
        
        return translation
    
    def get_available_languages(self) -> Dict[str, str]:
        """Получение доступных языков"""
        return {
            "ru": self.t("lang_russian"),
            "en": self.t("lang_english")
        }

# Глобальный экземпляр менеджера
i18n = I18nManager()

def _(key: str, **kwargs) -> str:
    """Сокращенная функция для перевода"""
    return i18n.t(key, **kwargs)

def set_language(lang: str):
    """Установка языка"""
    return i18n.set_language(lang)

def get_language() -> str:
    """Получение текущего языка"""
    return i18n.get_language()

def get_available_languages() -> Dict[str, str]:
    """Получение доступных языков"""
    return i18n.get_available_languages()

# Пример использования
if __name__ == "__main__":
    print("=== Тест системы интернационализации ===")
    
    # Русский язык (по умолчанию)
    print(f"Текущий язык: {get_language()}")
    print(f"Меню 'Файл': {_('menu_file')}")
    print(f"Кнопка 'Запустить': {_('btn_run')}")
    print(f"Позиция курсора: {_('cursor_position', line=10, col=5)}")
    
    print("\n--- Переключение на английский ---")
    set_language("en")
    print(f"Current language: {get_language()}")
    print(f"Menu 'File': {_('menu_file')}")
    print(f"Button 'Run': {_('btn_run')}")
    print(f"Cursor position: {_('cursor_position', line=10, col=5)}")
    
    print(f"\nДоступные языки: {get_available_languages()}") 