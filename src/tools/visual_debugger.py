"""
Визуальный отладчик для AnamorphX IDE

Возможности:
- Визуальные точки останова в редакторе
- Панель переменных и стека вызовов
- Пошаговое выполнение с подсветкой
- Интерактивная консоль отладки
- Визуализация нейронных состояний
"""

import tkinter as tk
from tkinter import ttk, Text, Scrollbar, Frame, Label, Button
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import threading
import json
import time

from .debugger import AnamorphDebugger, DebugState, Breakpoint, StackFrame
from .interpreter_integration import CodeExecutor, ExecutionMode, ExecutionContext


@dataclass
class VisualBreakpoint:
    """Визуальная точка останова"""
    breakpoint: Breakpoint
    line_widget: Optional[tk.Widget] = None
    marker_widget: Optional[tk.Widget] = None
    is_active: bool = True


class BreakpointManager:
    """Менеджер визуальных точек останова"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.breakpoints: Dict[int, VisualBreakpoint] = {}
        self.debugger: Optional[AnamorphDebugger] = None
        
        # Настройка тегов для подсветки
        self._setup_breakpoint_tags()
        
        # Привязка событий
        self.text_widget.bind('<Button-1>', self._on_click)
        self.text_widget.bind('<Button-3>', self._on_right_click)
    
    def _setup_breakpoint_tags(self):
        """Настройка тегов для точек останова"""
        # Активная точка останова
        self.text_widget.tag_configure(
            'breakpoint_active',
            background='#FF4444',
            foreground='white'
        )
        
        # Неактивная точка останова
        self.text_widget.tag_configure(
            'breakpoint_inactive', 
            background='#888888',
            foreground='white'
        )
        
        # Текущая строка выполнения
        self.text_widget.tag_configure(
            'current_line',
            background='#FFFF00',
            foreground='black'
        )
        
        # Строка с ошибкой
        self.text_widget.tag_configure(
            'error_line',
            background='#FF8888',
            foreground='black'
        )
    
    def _on_click(self, event):
        """Обработка клика по редактору"""
        # Получение позиции
        index = self.text_widget.index(f"@{event.x},{event.y}")
        line_num = int(index.split('.')[0]) - 1
        
        # Проверка клика по области номеров строк (приблизительно)
        if event.x < 50:  # Примерная область номеров строк
            self.toggle_breakpoint(line_num)
    
    def _on_right_click(self, event):
        """Обработка правого клика"""
        index = self.text_widget.index(f"@{event.x},{event.y}")
        line_num = int(index.split('.')[0]) - 1
        
        # Контекстное меню
        context_menu = tk.Menu(self.text_widget, tearoff=0)
        
        if line_num in self.breakpoints:
            context_menu.add_command(
                label="Удалить точку останова",
                command=lambda: self.remove_breakpoint(line_num)
            )
            context_menu.add_command(
                label="Редактировать условие",
                command=lambda: self._edit_breakpoint_condition(line_num)
            )
        else:
            context_menu.add_command(
                label="Добавить точку останова",
                command=lambda: self.add_breakpoint(line_num)
            )
        
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def set_debugger(self, debugger: AnamorphDebugger):
        """Установить отладчик"""
        self.debugger = debugger
    
    def add_breakpoint(self, line_num: int, condition: str = None) -> bool:
        """Добавить точку останова"""
        if line_num in self.breakpoints:
            return False
        
        # Создание точки останова в отладчике
        if self.debugger:
            bp_id = self.debugger.add_line_breakpoint(
                "current_file.amph", 
                line_num + 1,  # Отладчик использует 1-based индексы
                condition
            )
            
            breakpoint = None
            for bp in self.debugger.breakpoint_manager.breakpoints.values():
                if bp.id == bp_id:
                    breakpoint = bp
                    break
            
            if breakpoint:
                visual_bp = VisualBreakpoint(breakpoint=breakpoint)
                self.breakpoints[line_num] = visual_bp
                self._update_breakpoint_visual(line_num)
                return True
        
        return False
    
    def remove_breakpoint(self, line_num: int) -> bool:
        """Удалить точку останова"""
        if line_num not in self.breakpoints:
            return False
        
        visual_bp = self.breakpoints[line_num]
        
        # Удаление из отладчика
        if self.debugger:
            self.debugger.remove_breakpoint(visual_bp.breakpoint.id)
        
        # Удаление визуальных элементов
        self._clear_breakpoint_visual(line_num)
        
        del self.breakpoints[line_num]
        return True
    
    def toggle_breakpoint(self, line_num: int) -> bool:
        """Переключить точку останова"""
        if line_num in self.breakpoints:
            return self.remove_breakpoint(line_num)
        else:
            return self.add_breakpoint(line_num)
    
    def _update_breakpoint_visual(self, line_num: int):
        """Обновить визуальное отображение точки останова"""
        if line_num not in self.breakpoints:
            return
        
        visual_bp = self.breakpoints[line_num]
        
        # Определение тега
        tag = 'breakpoint_active' if visual_bp.is_active else 'breakpoint_inactive'
        
        # Применение тега к строке
        line_start = f"{line_num + 1}.0"
        line_end = f"{line_num + 1}.end"
        
        # Очистка старых тегов
        self.text_widget.tag_remove('breakpoint_active', line_start, line_end)
        self.text_widget.tag_remove('breakpoint_inactive', line_start, line_end)
        
        # Применение нового тега
        self.text_widget.tag_add(tag, line_start, line_end)
    
    def _clear_breakpoint_visual(self, line_num: int):
        """Очистить визуальное отображение точки останова"""
        line_start = f"{line_num + 1}.0"
        line_end = f"{line_num + 1}.end"
        
        self.text_widget.tag_remove('breakpoint_active', line_start, line_end)
        self.text_widget.tag_remove('breakpoint_inactive', line_start, line_end)
    
    def highlight_current_line(self, line_num: int):
        """Подсветить текущую строку выполнения"""
        # Очистка предыдущей подсветки
        self.text_widget.tag_remove('current_line', '1.0', 'end')
        
        # Подсветка текущей строки
        if line_num >= 0:
            line_start = f"{line_num + 1}.0"
            line_end = f"{line_num + 1}.end"
            self.text_widget.tag_add('current_line', line_start, line_end)
            
            # Прокрутка к текущей строке
            self.text_widget.see(line_start)
    
    def highlight_error_line(self, line_num: int):
        """Подсветить строку с ошибкой"""
        if line_num >= 0:
            line_start = f"{line_num + 1}.0"
            line_end = f"{line_num + 1}.end"
            self.text_widget.tag_add('error_line', line_start, line_end)
    
    def _edit_breakpoint_condition(self, line_num: int):
        """Редактировать условие точки останова"""
        if line_num not in self.breakpoints:
            return
        
        visual_bp = self.breakpoints[line_num]
        current_condition = visual_bp.breakpoint.condition or ""
        
        # Диалог редактирования условия
        dialog = tk.Toplevel(self.text_widget)
        dialog.title("Условие точки останова")
        dialog.geometry("400x200")
        
        tk.Label(dialog, text="Условие (Python выражение):").pack(pady=5)
        
        condition_text = tk.Text(dialog, height=5, width=50)
        condition_text.pack(pady=5, padx=10, fill='both', expand=True)
        condition_text.insert('1.0', current_condition)
        
        def save_condition():
            new_condition = condition_text.get('1.0', 'end-1c').strip()
            visual_bp.breakpoint.condition = new_condition if new_condition else None
            dialog.destroy()
        
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Сохранить", command=save_condition).pack(side='left', padx=5)
        tk.Button(button_frame, text="Отмена", command=dialog.destroy).pack(side='left', padx=5)


class VariablesPanel:
    """Панель переменных отладчика"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Заголовок
        ttk.Label(self.frame, text="Переменные", font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        
        # Дерево переменных
        self.tree = ttk.Treeview(
            self.frame,
            columns=('type', 'value'),
            show='tree headings',
            height=10
        )
        
        self.tree.heading('#0', text='Имя')
        self.tree.heading('type', text='Тип')
        self.tree.heading('value', text='Значение')
        
        self.tree.column('#0', width=150)
        self.tree.column('type', width=100)
        self.tree.column('value', width=200)
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(self.frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещение
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Контекстное меню
        self.tree.bind('<Button-3>', self._on_right_click)
    
    def update_variables(self, stack_frame: Optional[StackFrame]):
        """Обновить отображение переменных"""
        # Очистка дерева
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not stack_frame:
            return
        
        # Локальные переменные
        if stack_frame.local_variables:
            local_item = self.tree.insert('', 'end', text='Локальные', values=('', ''))
            
            for name, value in stack_frame.local_variables.items():
                self._add_variable(local_item, name, value)
        
        # Аргументы функции
        if stack_frame.arguments:
            args_item = self.tree.insert('', 'end', text='Аргументы', values=('', ''))
            
            for name, value in stack_frame.arguments.items():
                self._add_variable(args_item, name, value)
        
        # Раскрытие узлов
        for item in self.tree.get_children():
            self.tree.item(item, open=True)
    
    def _add_variable(self, parent, name: str, value: Any):
        """Добавить переменную в дерево"""
        value_type = type(value).__name__
        value_str = str(value)
        
        # Ограничение длины значения
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."
        
        item = self.tree.insert(parent, 'end', text=name, values=(value_type, value_str))
        
        # Для сложных объектов добавляем дочерние элементы
        if hasattr(value, '__dict__') and value.__dict__:
            for attr_name, attr_value in value.__dict__.items():
                if not attr_name.startswith('_'):
                    self._add_variable(item, attr_name, attr_value)
        elif isinstance(value, dict) and len(value) <= 20:  # Ограничение для больших словарей
            for key, val in value.items():
                self._add_variable(item, str(key), val)
        elif isinstance(value, (list, tuple)) and len(value) <= 20:  # Ограничение для больших списков
            for i, val in enumerate(value):
                self._add_variable(item, f"[{i}]", val)
    
    def _on_right_click(self, event):
        """Контекстное меню для переменных"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        
        if not item:
            return
        
        context_menu = tk.Menu(self.tree, tearoff=0)
        context_menu.add_command(label="Копировать значение", command=lambda: self._copy_value(item))
        context_menu.add_command(label="Добавить в отслеживание", command=lambda: self._add_to_watch(item))
        
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def _copy_value(self, item):
        """Копировать значение переменной"""
        values = self.tree.item(item, 'values')
        if values:
            self.tree.clipboard_clear()
            self.tree.clipboard_append(values[1])  # Значение
    
    def _add_to_watch(self, item):
        """Добавить переменную в отслеживание"""
        var_name = self.tree.item(item, 'text')
        # TODO: Добавить в список отслеживания
        print(f"Добавлено в отслеживание: {var_name}")


class CallStackPanel:
    """Панель стека вызовов"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Заголовок
        ttk.Label(self.frame, text="Стек вызовов", font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        
        # Список стека
        self.listbox = tk.Listbox(self.frame, height=8)
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(self.frame, orient='vertical', command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        # Размещение
        self.listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # События
        self.listbox.bind('<Double-Button-1>', self._on_double_click)
        
        # Хранение кадров стека
        self.stack_frames: List[StackFrame] = []
        
        # Обработчик выбора кадра
        self.on_frame_selected: Optional[Callable[[StackFrame], None]] = None
    
    def update_call_stack(self, stack_frames: List[StackFrame]):
        """Обновить стек вызовов"""
        self.stack_frames = stack_frames
        
        # Очистка списка
        self.listbox.delete(0, tk.END)
        
        # Заполнение стека (от текущего к корневому)
        for i, frame in enumerate(reversed(stack_frames)):
            entry = f"{frame.function_name} ({frame.file_path}:{frame.line})"
            self.listbox.insert(tk.END, entry)
            
            # Выделение текущего кадра
            if i == 0:
                self.listbox.selection_set(0)
    
    def _on_double_click(self, event):
        """Обработка двойного клика по кадру стека"""
        selection = self.listbox.curselection()
        if selection and self.on_frame_selected:
            # Индекс в обратном порядке
            frame_index = len(self.stack_frames) - 1 - selection[0]
            if 0 <= frame_index < len(self.stack_frames):
                self.on_frame_selected(self.stack_frames[frame_index])


class DebugConsole:
    """Консоль отладки"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Заголовок
        ttk.Label(self.frame, text="Консоль отладки", font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        
        # Область вывода
        self.output_text = tk.Text(
            self.frame,
            height=10,
            background='black',
            foreground='green',
            font=('Consolas', 9),
            state='disabled'
        )
        
        # Поле ввода
        self.input_frame = ttk.Frame(self.frame)
        self.input_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.input_frame, text=">>>").pack(side='left')
        
        self.input_entry = ttk.Entry(self.input_frame)
        self.input_entry.pack(side='left', fill='x', expand=True, padx=5)
        self.input_entry.bind('<Return>', self._on_command)
        
        # Размещение
        self.output_text.pack(fill='both', expand=True)
        
        # Отладчик
        self.debugger: Optional[AnamorphDebugger] = None
        
        # История команд
        self.command_history: List[str] = []
        self.history_index = -1
        
        self.input_entry.bind('<Up>', self._history_up)
        self.input_entry.bind('<Down>', self._history_down)
        
        # Приветствие
        self._append_output("🐛 Консоль отладки AnamorphX\n")
        self._append_output("Команды: help, vars, eval <expr>, bt, step, next, cont\n\n")
    
    def set_debugger(self, debugger: AnamorphDebugger):
        """Установить отладчик"""
        self.debugger = debugger
    
    def _on_command(self, event=None):
        """Обработка команды"""
        command = self.input_entry.get().strip()
        if not command:
            return
        
        # Добавление в историю
        self.command_history.append(command)
        self.history_index = len(self.command_history)
        
        # Отображение команды
        self._append_output(f">>> {command}\n")
        
        # Очистка поля ввода
        self.input_entry.delete(0, tk.END)
        
        # Выполнение команды
        self._execute_command(command)
    
    def _execute_command(self, command: str):
        """Выполнить команду отладки"""
        try:
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'help':
                self._show_help()
            elif cmd == 'vars':
                self._show_variables()
            elif cmd == 'eval' and len(parts) > 1:
                expr = ' '.join(parts[1:])
                self._evaluate_expression(expr)
            elif cmd == 'bt':
                self._show_backtrace()
            elif cmd == 'step':
                self._step_into()
            elif cmd == 'next':
                self._step_over()
            elif cmd == 'cont':
                self._continue()
            elif cmd == 'bp' and len(parts) > 1:
                line_num = int(parts[1])
                self._add_breakpoint(line_num)
            else:
                self._append_output(f"Неизвестная команда: {cmd}\n")
                
        except Exception as e:
            self._append_output(f"Ошибка: {e}\n")
    
    def _show_help(self):
        """Показать справку"""
        help_text = """
Доступные команды:
  help          - показать справку
  vars          - показать переменные
  eval <expr>   - вычислить выражение
  bt            - показать стек вызовов
  step          - шаг с заходом в функции
  next          - шаг с пропуском функций
  cont          - продолжить выполнение
  bp <line>     - добавить точку останова
"""
        self._append_output(help_text)
    
    def _show_variables(self):
        """Показать переменные"""
        if not self.debugger or not self.debugger.current_frame:
            self._append_output("Нет текущего кадра\n")
            return
        
        frame = self.debugger.current_frame
        
        if frame.local_variables:
            self._append_output("Локальные переменные:\n")
            for name, value in frame.local_variables.items():
                self._append_output(f"  {name} = {value}\n")
        
        if frame.arguments:
            self._append_output("Аргументы:\n")
            for name, value in frame.arguments.items():
                self._append_output(f"  {name} = {value}\n")
    
    def _evaluate_expression(self, expression: str):
        """Вычислить выражение"""
        if not self.debugger:
            self._append_output("Отладчик не активен\n")
            return
        
        result = self.debugger.evaluate_expression(expression)
        self._append_output(f"=> {result}\n")
    
    def _show_backtrace(self):
        """Показать стек вызовов"""
        if not self.debugger:
            self._append_output("Отладчик не активен\n")
            return
        
        self._append_output("Стек вызовов:\n")
        for i, frame in enumerate(self.debugger.call_stack):
            marker = "=> " if i == len(self.debugger.call_stack) - 1 else "   "
            self._append_output(f"{marker}{frame.function_name} ({frame.file_path}:{frame.line})\n")
    
    def _step_into(self):
        """Шаг с заходом"""
        if self.debugger:
            self.debugger.step_into()
            self._append_output("Выполнен шаг с заходом\n")
    
    def _step_over(self):
        """Шаг с пропуском"""
        if self.debugger:
            self.debugger.step_over()
            self._append_output("Выполнен шаг с пропуском\n")
    
    def _continue(self):
        """Продолжить выполнение"""
        if self.debugger:
            self.debugger.resume()
            self._append_output("Выполнение продолжено\n")
    
    def _add_breakpoint(self, line_num: int):
        """Добавить точку останова"""
        if self.debugger:
            bp_id = self.debugger.add_line_breakpoint("current_file.amph", line_num)
            self._append_output(f"Добавлена точка останова на строке {line_num}\n")
    
    def _append_output(self, text: str):
        """Добавить текст в вывод"""
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, text)
        self.output_text.config(state='disabled')
        self.output_text.see(tk.END)
    
    def _history_up(self, event):
        """Навигация по истории (вверх)"""
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.command_history[self.history_index])
    
    def _history_down(self, event):
        """Навигация по истории (вниз)"""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.command_history[self.history_index])
        elif self.history_index == len(self.command_history) - 1:
            self.history_index = len(self.command_history)
            self.input_entry.delete(0, tk.END)


class VisualDebugger:
    """Главный класс визуального отладчика"""
    
    def __init__(self, ide_components):
        self.ide = ide_components
        self.debugger: Optional[AnamorphDebugger] = None
        self.executor: Optional[CodeExecutor] = None
        
        # Компоненты UI
        self.breakpoint_manager: Optional[BreakpointManager] = None
        self.variables_panel: Optional[VariablesPanel] = None
        self.call_stack_panel: Optional[CallStackPanel] = None
        self.debug_console: Optional[DebugConsole] = None
        
        # Состояние отладки
        self.is_debugging = False
        self.current_line = -1
        
        # Создание UI
        self._create_debug_ui()
        
        # Настройка обработчиков
        self._setup_event_handlers()
    
    def _create_debug_ui(self):
        """Создание интерфейса отладчика"""
        if not hasattr(self.ide, 'debug_frame'):
            return
        
        # Очистка существующего содержимого
        for widget in self.ide.debug_frame.winfo_children():
            widget.destroy()
        
        # Основной контейнер
        main_paned = ttk.PanedWindow(self.ide.debug_frame, orient='horizontal')
        main_paned.pack(fill='both', expand=True)
        
        # Левая панель (переменные и стек)
        left_paned = ttk.PanedWindow(main_paned, orient='vertical')
        main_paned.add(left_paned, weight=1)
        
        # Панель переменных
        variables_frame = ttk.Frame(left_paned)
        left_paned.add(variables_frame, weight=1)
        self.variables_panel = VariablesPanel(variables_frame)
        self.variables_panel.frame.pack(fill='both', expand=True)
        
        # Панель стека вызовов
        stack_frame = ttk.Frame(left_paned)
        left_paned.add(stack_frame, weight=1)
        self.call_stack_panel = CallStackPanel(stack_frame)
        self.call_stack_panel.frame.pack(fill='both', expand=True)
        
        # Правая панель (консоль отладки)
        console_frame = ttk.Frame(main_paned)
        main_paned.add(console_frame, weight=1)
        self.debug_console = DebugConsole(console_frame)
        self.debug_console.frame.pack(fill='both', expand=True)
        
        # Интеграция с редактором кода
        if hasattr(self.ide, 'code_editor') and hasattr(self.ide.code_editor, 'text_widget'):
            self.breakpoint_manager = BreakpointManager(self.ide.code_editor.text_widget)
    
    def _setup_event_handlers(self):
        """Настройка обработчиков событий"""
        if self.call_stack_panel:
            self.call_stack_panel.on_frame_selected = self._on_frame_selected
    
    def start_debugging(self, code: str, file_path: str = "main.amph"):
        """Начать отладку"""
        if self.is_debugging:
            self.stop_debugging()
        
        # Создание отладчика
        self.debugger = AnamorphDebugger()
        self.executor = CodeExecutor()
        
        # Настройка компонентов
        if self.breakpoint_manager:
            self.breakpoint_manager.set_debugger(self.debugger)
        
        if self.debug_console:
            self.debug_console.set_debugger(self.debugger)
        
        # Настройка обработчиков отладчика
        self.debugger.add_event_handler(self)
        
        # Создание контекста выполнения
        context = ExecutionContext(
            file_path=file_path,
            code=code,
            mode=ExecutionMode.DEBUG,
            debug_enabled=True,
            async_execution=True
        )
        
        # Запуск отладки
        self.executor.execute(context)
        self.is_debugging = True
        
        # Обновление UI
        self._update_debug_state()
    
    def stop_debugging(self):
        """Остановить отладку"""
        if self.debugger:
            self.debugger.stop()
            self.debugger = None
        
        if self.executor:
            self.executor.stop_execution()
            self.executor = None
        
        self.is_debugging = False
        self.current_line = -1
        
        # Очистка подсветки
        if self.breakpoint_manager:
            self.breakpoint_manager.highlight_current_line(-1)
        
        # Очистка панелей
        if self.variables_panel:
            self.variables_panel.update_variables(None)
        
        if self.call_stack_panel:
            self.call_stack_panel.update_call_stack([])
        
        self._update_debug_state()
    
    def step_into(self):
        """Шаг с заходом в функции"""
        if self.debugger:
            self.debugger.step_into()
    
    def step_over(self):
        """Шаг с пропуском функций"""
        if self.debugger:
            self.debugger.step_over()
    
    def step_out(self):
        """Шаг с выходом из функции"""
        if self.debugger:
            self.debugger.step_out()
    
    def continue_execution(self):
        """Продолжить выполнение"""
        if self.debugger:
            self.debugger.resume()
    
    def pause_execution(self):
        """Приостановить выполнение"""
        if self.debugger:
            self.debugger.pause()
    
    def _on_frame_selected(self, frame: StackFrame):
        """Обработка выбора кадра стека"""
        # Обновление панели переменных
        if self.variables_panel:
            self.variables_panel.update_variables(frame)
        
        # Переход к строке в редакторе
        if self.breakpoint_manager:
            self.breakpoint_manager.highlight_current_line(frame.line - 1)
    
    def _update_debug_state(self):
        """Обновление состояния UI отладчика"""
        # Обновление кнопок в toolbar
        if hasattr(self.ide, 'toolbar'):
            # TODO: Обновить состояние кнопок отладки
            pass
        
        # Обновление статуса
        if hasattr(self.ide, 'status_label'):
            if self.is_debugging:
                state = self.debugger.state.value if self.debugger else "unknown"
                self.ide.status_label.config(text=f"🐛 Отладка: {state}")
            else:
                self.ide.status_label.config(text="Готов")
    
    # Реализация DebugEventHandler
    def on_breakpoint_hit(self, breakpoint: Breakpoint, context: Dict):
        """Обработка попадания в точку останова"""
        line_num = context.get('line', 0) - 1
        
        # Подсветка текущей строки
        if self.breakpoint_manager:
            self.breakpoint_manager.highlight_current_line(line_num)
        
        # Обновление панелей
        self._update_debug_panels(context)
        
        # Вывод в консоль
        if self.debug_console:
            self.debug_console._append_output(f"🔴 Точка останова на строке {line_num + 1}\n")
    
    def on_step_complete(self, context: Dict):
        """Обработка завершения шага"""
        line_num = context.get('line', 0) - 1
        
        # Подсветка текущей строки
        if self.breakpoint_manager:
            self.breakpoint_manager.highlight_current_line(line_num)
        
        # Обновление панелей
        self._update_debug_panels(context)
    
    def on_variable_changed(self, name: str, old_value: Any, new_value: Any):
        """Обработка изменения переменной"""
        if self.debug_console:
            self.debug_console._append_output(f"🔄 {name}: {old_value} -> {new_value}\n")
    
    def on_exception(self, exception: Exception, context: Dict):
        """Обработка исключения"""
        line_num = context.get('line', 0) - 1
        
        # Подсветка строки с ошибкой
        if self.breakpoint_manager:
            self.breakpoint_manager.highlight_error_line(line_num)
        
        # Вывод в консоль
        if self.debug_console:
            self.debug_console._append_output(f"❌ Исключение: {exception}\n")
    
    def _update_debug_panels(self, context: Dict):
        """Обновление панелей отладки"""
        # Обновление стека вызовов
        if self.call_stack_panel and self.debugger:
            self.call_stack_panel.update_call_stack(self.debugger.call_stack)
        
        # Обновление переменных
        if self.variables_panel and self.debugger and self.debugger.current_frame:
            self.variables_panel.update_variables(self.debugger.current_frame)


def integrate_visual_debugger(ide_components) -> VisualDebugger:
    """Интеграция визуального отладчика с IDE"""
    visual_debugger = VisualDebugger(ide_components)
    
    # Перехват команд отладки в IDE
    if hasattr(ide_components, '_debug_program'):
        original_debug = ide_components._debug_program
        
        def new_debug_program():
            if hasattr(ide_components, 'code_editor'):
                code = ide_components.code_editor.get_content()
                file_path = ide_components.code_editor.current_file or "untitled.amph"
                visual_debugger.start_debugging(code, file_path)
        
        ide_components._debug_program = new_debug_program
    
    return visual_debugger


if __name__ == "__main__":
    # Тестирование визуального отладчика
    import tkinter as tk
    
    # Создание тестового окна
    root = tk.Tk()
    root.title("Тест визуального отладчика")
    root.geometry("800x600")
    
    # Создание тестового редактора
    editor_frame = ttk.Frame(root)
    editor_frame.pack(side='left', fill='both', expand=True)
    
    text_widget = tk.Text(editor_frame, font=('Consolas', 12))
    text_widget.pack(fill='both', expand=True)
    
    # Тестовый код
    test_code = '''def factorial(n):
    if n <= 1:
        return 1
    else:
        result = n * factorial(n - 1)
        return result

number = 5
result = factorial(number)
print(f"Factorial of {number} is {result}")'''
    
    text_widget.insert('1.0', test_code)
    
    # Создание панели отладки
    debug_frame = ttk.Frame(root)
    debug_frame.pack(side='right', fill='both', expand=True)
    
    # Имитация IDE
    class MockIDE:
        def __init__(self):
            self.debug_frame = debug_frame
            self.code_editor = MockEditor()
    
    class MockEditor:
        def __init__(self):
            self.text_widget = text_widget
            self.current_file = "test.amph"
        
        def get_content(self):
            return text_widget.get('1.0', 'end-1c')
    
    ide = MockIDE()
    
    # Создание визуального отладчика
    visual_debugger = VisualDebugger(ide)
    
    # Кнопки управления
    control_frame = ttk.Frame(root)
    control_frame.pack(side='bottom', fill='x', pady=5)
    
    ttk.Button(control_frame, text="🐛 Начать отладку", 
              command=lambda: visual_debugger.start_debugging(test_code)).pack(side='left', padx=5)
    ttk.Button(control_frame, text="⏸️ Пауза", 
              command=visual_debugger.pause_execution).pack(side='left', padx=5)
    ttk.Button(control_frame, text="▶️ Продолжить", 
              command=visual_debugger.continue_execution).pack(side='left', padx=5)
    ttk.Button(control_frame, text="🚪 Шаг", 
              command=visual_debugger.step_into).pack(side='left', padx=5)
    ttk.Button(control_frame, text="🛑 Стоп", 
              command=visual_debugger.stop_debugging).pack(side='left', padx=5)
    
    print("🔍 Визуальный отладчик готов к тестированию")
    print("Используйте кнопки для управления отладкой")
    
    root.mainloop() 