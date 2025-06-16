"""
Улучшенная IDE для AnamorphX с интеграцией всех инструментов разработки

Возможности:
- Асинхронное выполнение с управлением состояниями
- Интеграция отладчика с синхронизацией файлов
- Визуализация профайлера в реальном времени
- Инкрементальная подсветка с debounce
- Управление сессиями и таймаутами
- Потокобезопасный UI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import time
import uuid
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Импорты компонентов AnamorphX
from .interpreter_integration import CodeExecutor, ExecutionMode, ExecutionContext, IDEIntegration
from .incremental_highlighter import IncrementalHighlighter, TextWidgetHighlighter
from .visual_debugger import VisualDebugger, BreakpointManager
from .profiler_visualizer import ProfilerVisualizer
from .syntax_highlighter import AnamorphSyntaxHighlighter
from .ide_components import CodeEditor, FileExplorer, TerminalPanel


class ExecutionState(Enum):
    """Состояния выполнения"""
    IDLE = "idle"
    RUNNING = "running" 
    DEBUGGING = "debugging"
    PROFILING = "profiling"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ExecutionSession:
    """Сессия выполнения кода"""
    session_id: str
    file_path: str
    code: str
    mode: ExecutionMode
    state: ExecutionState = ExecutionState.IDLE
    start_time: float = field(default_factory=time.time)
    executor: Optional[CodeExecutor] = None
    context: Optional[ExecutionContext] = None
    output: List[str] = field(default_factory=list)
    error: Optional[str] = None
    timeout: float = 30.0  # Таймаут в секундах


class SessionManager:
    """Менеджер сессий выполнения"""
    
    def __init__(self):
        self.sessions: Dict[str, ExecutionSession] = {}
        self.active_session: Optional[str] = None
        self.session_callbacks: Dict[str, List[Callable]] = {}
        
        # Настройка логирования
        self.logger = logging.getLogger('SessionManager')
        self.logger.setLevel(logging.DEBUG)
        
        # Поток мониторинга сессий
        self.monitor_thread = threading.Thread(target=self._monitor_sessions, daemon=True)
        self.monitor_running = threading.Event()
        self.monitor_thread.start()
    
    def create_session(self, file_path: str, code: str, mode: ExecutionMode, 
                      timeout: float = 30.0) -> str:
        """Создать новую сессию"""
        session_id = str(uuid.uuid4())[:8]
        
        session = ExecutionSession(
            session_id=session_id,
            file_path=file_path,
            code=code,
            mode=mode,
            timeout=timeout
        )
        
        self.sessions[session_id] = session
        self.logger.info(f"Создана сессия {session_id}: {mode.value} для {file_path}")
        
        return session_id
    
    def start_session(self, session_id: str) -> bool:
        """Запустить сессию"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            # Создание исполнителя
            session.executor = CodeExecutor()
            
            # Создание контекста
            session.context = ExecutionContext(
                file_path=session.file_path,
                code=session.code,
                mode=session.mode,
                debug_enabled=(session.mode == ExecutionMode.DEBUG),
                profiling_enabled=(session.mode == ExecutionMode.PROFILE),
                async_execution=True
            )
            
            # Запуск выполнения
            session.executor.execute(session.context)
            session.state = ExecutionState.RUNNING
            self.active_session = session_id
            
            self.logger.info(f"Запущена сессия {session_id}")
            self._notify_callbacks(session_id, 'started')
            
            return True
            
        except Exception as e:
            session.state = ExecutionState.ERROR
            session.error = str(e)
            self.logger.error(f"Ошибка запуска сессии {session_id}: {e}")
            self._notify_callbacks(session_id, 'error', str(e))
            return False
    
    def stop_session(self, session_id: str) -> bool:
        """Остановить сессию"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            if session.executor:
                session.executor.stop_execution()
            
            session.state = ExecutionState.IDLE
            
            if self.active_session == session_id:
                self.active_session = None
            
            self.logger.info(f"Остановлена сессия {session_id}")
            self._notify_callbacks(session_id, 'stopped')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка остановки сессии {session_id}: {e}")
            return False
    
    def get_session_output(self, session_id: str) -> List[str]:
        """Получить вывод сессии"""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        
        # Получение нового вывода из исполнителя
        if session.executor and hasattr(session.executor, 'get_output'):
            new_output = session.executor.get_output()
            session.output.extend(new_output)
        
        return session.output.copy()
    
    def add_callback(self, session_id: str, callback: Callable):
        """Добавить callback для сессии"""
        if session_id not in self.session_callbacks:
            self.session_callbacks[session_id] = []
        
        self.session_callbacks[session_id].append(callback)
    
    def _notify_callbacks(self, session_id: str, event: str, data: Any = None):
        """Уведомить callbacks"""
        if session_id in self.session_callbacks:
            for callback in self.session_callbacks[session_id]:
                try:
                    callback(session_id, event, data)
                except Exception as e:
                    self.logger.error(f"Ошибка callback: {e}")
    
    def _monitor_sessions(self):
        """Мониторинг сессий (таймауты, завершения)"""
        self.monitor_running.set()
        
        while self.monitor_running.is_set():
            try:
                current_time = time.time()
                
                for session_id, session in list(self.sessions.items()):
                    # Проверка таймаута
                    if (session.state == ExecutionState.RUNNING and 
                        current_time - session.start_time > session.timeout):
                        
                        self.logger.warning(f"Таймаут сессии {session_id}")
                        self.stop_session(session_id)
                        session.state = ExecutionState.ERROR
                        session.error = "Превышен таймаут выполнения"
                        self._notify_callbacks(session_id, 'timeout')
                    
                    # Проверка завершения
                    if (session.executor and 
                        hasattr(session.executor, 'is_finished') and
                        session.executor.is_finished()):
                        
                        session.state = ExecutionState.IDLE
                        if self.active_session == session_id:
                            self.active_session = None
                        
                        self._notify_callbacks(session_id, 'finished')
                
                time.sleep(0.5)  # Проверка каждые 500мс
                
            except Exception as e:
                self.logger.error(f"Ошибка мониторинга: {e}")
                time.sleep(1.0)
    
    def cleanup(self):
        """Очистка и остановка мониторинга"""
        self.monitor_running.clear()
        
        # Остановка всех активных сессий
        for session_id in list(self.sessions.keys()):
            self.stop_session(session_id)
        
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)


class DebounceHandler:
    """Обработчик debounce для оптимизации событий"""
    
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.timers: Dict[str, threading.Timer] = {}
        self.lock = threading.Lock()
    
    def debounce(self, key: str, func: Callable, *args, **kwargs):
        """Вызвать функцию с задержкой debounce"""
        with self.lock:
            # Отмена предыдущего таймера
            if key in self.timers:
                self.timers[key].cancel()
            
            # Создание нового таймера
            timer = threading.Timer(self.delay, func, args, kwargs)
            self.timers[key] = timer
            timer.start()
    
    def cancel_all(self):
        """Отменить все активные таймеры"""
        with self.lock:
            for timer in self.timers.values():
                timer.cancel()
            self.timers.clear()


class EnhancedCodeEditor(CodeEditor):
    """Улучшенный редактор кода с инкрементальной подсветкой"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        # Инкрементальная подсветка
        self.incremental_highlighter = IncrementalHighlighter()
        self.widget_highlighter = TextWidgetHighlighter(self.text_widget)
        
        # Debounce для оптимизации
        self.debounce_handler = DebounceHandler(delay=0.15)
        
        # Привязка событий изменения текста
        self.text_widget.bind('<KeyRelease>', self._on_text_changed)
        self.text_widget.bind('<Button-1>', self._on_text_changed)
        self.text_widget.bind('<<Paste>>', self._on_text_changed)
        
        # Блокировка редактирования
        self.editing_locked = False
        
        # Кэш состояния
        self.last_content_hash = None
        
        # Настройка автоматической подсветки
        self._setup_auto_highlighting()
    
    def _setup_auto_highlighting(self):
        """Настройка автоматической подсветки"""
        # Начальная подсветка
        self._update_highlighting()
        
        # Периодическое обновление
        self._schedule_highlighting_update()
    
    def _on_text_changed(self, event=None):
        """Обработка изменения текста"""
        if self.editing_locked:
            return
        
        # Debounce обновления подсветки
        self.debounce_handler.debounce(
            'highlighting',
            self._update_highlighting
        )
    
    def _update_highlighting(self):
        """Обновление подсветки синтаксиса"""
        try:
            content = self.text_widget.get('1.0', 'end-1c')
            content_hash = hash(content)
            
            # Проверка изменений
            if content_hash == self.last_content_hash:
                return
            
            self.last_content_hash = content_hash
            
            # Разбивка на строки
            lines = content.split('\n')
            
            # Обновление инкрементального подсвечивателя
            self.incremental_highlighter.update_lines(lines, 0, len(lines))
            
            # Получение подсветки
            highlights = self.incremental_highlighter.get_highlights_for_range(0, len(lines))
            
            # Применение подсветки через UI поток
            self.text_widget.after_idle(
                lambda: self.widget_highlighter.apply_highlights(highlights, 0)
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления подсветки: {e}")
    
    def _schedule_highlighting_update(self):
        """Планирование обновления подсветки"""
        self.text_widget.after(500, self._schedule_highlighting_update)
        
        # Периодическая очистка кэша
        if hasattr(self, 'last_cache_clear'):
            if time.time() - self.last_cache_clear > 300:  # 5 минут
                self.incremental_highlighter.clear_cache()
                self.last_cache_clear = time.time()
        else:
            self.last_cache_clear = time.time()
    
    def lock_editing(self):
        """Заблокировать редактирование"""
        self.editing_locked = True
        self.text_widget.config(state='disabled')
    
    def unlock_editing(self):
        """Разблокировать редактирование"""
        self.editing_locked = False
        self.text_widget.config(state='normal')
    
    def get_current_file_path(self) -> str:
        """Получить путь к текущему файлу"""
        return self.current_file or "untitled.amph"


class EnhancedAnamorphIDE:
    """Улучшенная IDE для AnamorphX"""
    
    def __init__(self):
        # Главное окно
        self.root = tk.Tk()
        self.root.title("AnamorphX IDE - Enhanced Edition")
        self.root.geometry("1400x900")
        
        # Менеджеры
        self.session_manager = SessionManager()
        self.debounce_handler = DebounceHandler()
        
        # Состояние
        self.execution_state = ExecutionState.IDLE
        self.current_session_id: Optional[str] = None
        
        # Настройка логирования
        self.logger = logging.getLogger('EnhancedIDE')
        self.logger.setLevel(logging.DEBUG)
        
        # Создание интерфейса
        self._create_interface()
        
        # Интеграция компонентов
        self._setup_integrations()
        
        # Настройка обработчиков
        self._setup_event_handlers()
        
        # Состояние UI
        self._update_ui_state()
    
    def _create_interface(self):
        """Создание пользовательского интерфейса"""
        # Меню
        self._create_menu()
        
        # Toolbar
        self._create_toolbar()
        
        # Главный контейнер
        self.main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        self.main_paned.pack(fill='both', expand=True)
        
        # Левая панель (файлы)
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Файловый проводник
        self.file_explorer = FileExplorer(self.left_frame)
        self.file_explorer.frame.pack(fill='both', expand=True)
        
        # Центральная панель
        self.center_paned = ttk.PanedWindow(self.main_paned, orient='vertical')
        self.main_paned.add(self.center_paned, weight=4)
        
        # Редактор кода
        self.code_editor = EnhancedCodeEditor(self.center_paned)
        self.center_paned.add(self.code_editor.frame, weight=3)
        
        # Нижняя панель (вкладки)
        self.bottom_notebook = ttk.Notebook(self.center_paned)
        self.center_paned.add(self.bottom_notebook, weight=1)
        
        # Терминал
        self.terminal_panel = TerminalPanel(self.bottom_notebook)
        self.bottom_notebook.add(self.terminal_panel.frame, text="Терминал")
        
        # Статусная строка
        self._create_status_bar()
    
    def _create_menu(self):
        """Создание меню"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # Файл
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Новый", command=self._new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Открыть", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Сохранить", command=self._save_file, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self._exit_app)
        
        # Правка
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Правка", menu=edit_menu)
        edit_menu.add_command(label="Отменить", command=self._undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Повторить", command=self._redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Найти", command=self._find, accelerator="Ctrl+F")
        edit_menu.add_command(label="Заменить", command=self._replace, accelerator="Ctrl+H")
        
        # Выполнение
        run_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Выполнение", menu=run_menu)
        run_menu.add_command(label="Запустить", command=self._run_program, accelerator="F5")
        run_menu.add_command(label="Отладка", command=self._debug_program, accelerator="F9")
        run_menu.add_command(label="Профилирование", command=self._profile_program, accelerator="F11")
        run_menu.add_separator()
        run_menu.add_command(label="Остановить", command=self._stop_execution, accelerator="Shift+F5")
        
        # Горячие клавиши
        self.root.bind('<Control-n>', lambda e: self._new_file())
        self.root.bind('<Control-o>', lambda e: self._open_file())
        self.root.bind('<Control-s>', lambda e: self._save_file())
        self.root.bind('<Control-f>', lambda e: self._find())
        self.root.bind('<Control-h>', lambda e: self._replace())
        self.root.bind('<F5>', lambda e: self._run_program())
        self.root.bind('<F9>', lambda e: self._debug_program())
        self.root.bind('<F11>', lambda e: self._profile_program())
        self.root.bind('<Shift-F5>', lambda e: self._stop_execution())
    
    def _create_toolbar(self):
        """Создание панели инструментов"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side='top', fill='x', padx=5, pady=2)
        
        # Кнопки файлов
        self.new_btn = ttk.Button(self.toolbar, text="📄 Новый", command=self._new_file)
        self.new_btn.pack(side='left', padx=2)
        
        self.open_btn = ttk.Button(self.toolbar, text="📁 Открыть", command=self._open_file)
        self.open_btn.pack(side='left', padx=2)
        
        self.save_btn = ttk.Button(self.toolbar, text="💾 Сохранить", command=self._save_file)
        self.save_btn.pack(side='left', padx=2)
        
        ttk.Separator(self.toolbar, orient='vertical').pack(side='left', padx=5, fill='y')
        
        # Кнопки выполнения
        self.run_btn = ttk.Button(self.toolbar, text="▶️ Запуск", command=self._run_program)
        self.run_btn.pack(side='left', padx=2)
        
        self.debug_btn = ttk.Button(self.toolbar, text="🐛 Отладка", command=self._debug_program)
        self.debug_btn.pack(side='left', padx=2)
        
        self.profile_btn = ttk.Button(self.toolbar, text="📊 Профиль", command=self._profile_program)
        self.profile_btn.pack(side='left', padx=2)
        
        self.stop_btn = ttk.Button(self.toolbar, text="⏹️ Стоп", command=self._stop_execution)
        self.stop_btn.pack(side='left', padx=2)
        
        ttk.Separator(self.toolbar, orient='vertical').pack(side='left', padx=5, fill='y')
        
        # Индикатор выполнения
        self.execution_label = ttk.Label(self.toolbar, text="⚫ Готов")
        self.execution_label.pack(side='left', padx=10)
        
        # Прогресс-бар
        self.progress_var = tk.StringVar()
        self.progress_bar = ttk.Progressbar(
            self.toolbar, 
            mode='indeterminate', 
            length=200
        )
        # Скрыт по умолчанию
    
    def _create_status_bar(self):
        """Создание статусной строки"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side='bottom', fill='x')
        
        self.status_label = ttk.Label(self.status_frame, text="Готов")
        self.status_label.pack(side='left', padx=5)
        
        # Информация о позиции курсора
        self.cursor_label = ttk.Label(self.status_frame, text="Строка: 1, Столбец: 1")
        self.cursor_label.pack(side='right', padx=5)
        
        # Информация о файле
        self.file_label = ttk.Label(self.status_frame, text="untitled.amph")
        self.file_label.pack(side='right', padx=20)
    
    def _setup_integrations(self):
        """Настройка интеграций компонентов"""
        # Интеграция с интерпретатором
        self.ide_integration = IDEIntegration(self)
        
        # Визуальный отладчик
        self.visual_debugger = VisualDebugger(self)
        
        # Визуализация профайлера
        self.profiler_visualizer = ProfilerVisualizer(self)
        
        # Подсветка синтаксиса
        self.syntax_highlighter = AnamorphSyntaxHighlighter()
    
    def _setup_event_handlers(self):
        """Настройка обработчиков событий"""
        # Обработчики файлового проводника
        self.file_explorer.on_file_selected = self._on_file_selected
        
        # Обработчики редактора
        self.code_editor.text_widget.bind('<KeyRelease>', self._on_cursor_moved)
        self.code_editor.text_widget.bind('<Button-1>', self._on_cursor_moved)
        
        # Обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _update_ui_state(self):
        """Обновление состояния UI"""
        state = self.execution_state
        
        # Обновление кнопок
        if state == ExecutionState.IDLE:
            self.run_btn.config(state='normal')
            self.debug_btn.config(state='normal')
            self.profile_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.execution_label.config(text="⚫ Готов", foreground='green')
            self.progress_bar.pack_forget()
            self.code_editor.unlock_editing()
            
        elif state in [ExecutionState.RUNNING, ExecutionState.DEBUGGING, ExecutionState.PROFILING]:
            self.run_btn.config(state='disabled')
            self.debug_btn.config(state='disabled')
            self.profile_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            if state == ExecutionState.RUNNING:
                self.execution_label.config(text="▶️ Выполнение", foreground='blue')
            elif state == ExecutionState.DEBUGGING:
                self.execution_label.config(text="🐛 Отладка", foreground='orange')
            elif state == ExecutionState.PROFILING:
                self.execution_label.config(text="📊 Профилирование", foreground='purple')
            
            self.progress_bar.pack(side='right', padx=5)
            self.progress_bar.start()
            self.code_editor.lock_editing()
            
        elif state == ExecutionState.PAUSED:
            self.stop_btn.config(state='normal')
            self.execution_label.config(text="⏸️ Пауза", foreground='orange')
            self.progress_bar.stop()
            
        elif state == ExecutionState.ERROR:
            self.run_btn.config(state='normal')
            self.debug_btn.config(state='normal')
            self.profile_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.execution_label.config(text="❌ Ошибка", foreground='red')
            self.progress_bar.pack_forget()
            self.code_editor.unlock_editing()
        
        # Планирование следующего обновления
        self.root.after(100, self._update_ui_state)
    
    # Обработчики меню и кнопок
    def _new_file(self):
        """Создать новый файл"""
        if self._check_unsaved_changes():
            self.code_editor.set_content("")
            self.code_editor.current_file = None
            self.file_label.config(text="untitled.amph")
            self.status_label.config(text="Новый файл создан")
    
    def _open_file(self):
        """Открыть файл"""
        if not self._check_unsaved_changes():
            return
        
        file_path = filedialog.askopenfilename(
            title="Открыть файл AnamorphX",
            filetypes=[
                ("AnamorphX files", "*.amph"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                self.code_editor.set_content(content)
                self.code_editor.current_file = file_path
                self.file_label.config(text=os.path.basename(file_path))
                self.status_label.config(text=f"Открыт файл: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
    
    def _save_file(self):
        """Сохранить файл"""
        if not self.code_editor.current_file:
            return self._save_file_as()
        
        try:
            with open(self.code_editor.current_file, 'w', encoding='utf-8') as file:
                file.write(self.code_editor.get_content())
            
            self.status_label.config(text=f"Сохранен: {self.code_editor.current_file}")
            return True
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")
            return False
    
    def _save_file_as(self):
        """Сохранить файл как"""
        file_path = filedialog.asksaveasfilename(
            title="Сохранить файл AnamorphX",
            defaultextension=".amph",
            filetypes=[
                ("AnamorphX files", "*.amph"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(self.code_editor.get_content())
                
                self.code_editor.current_file = file_path
                self.file_label.config(text=os.path.basename(file_path))
                self.status_label.config(text=f"Сохранен: {file_path}")
                return True
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")
                return False
        
        return False
    
    def _run_program(self):
        """Запустить программу"""
        if self.execution_state != ExecutionState.IDLE:
            return
        
        code = self.code_editor.get_content()
        if not code.strip():
            messagebox.showwarning("Предупреждение", "Нет кода для выполнения")
            return
        
        file_path = self.code_editor.get_current_file_path()
        
        # Создание и запуск сессии
        session_id = self.session_manager.create_session(
            file_path, code, ExecutionMode.INTERPRET
        )
        
        # Добавление callback
        self.session_manager.add_callback(session_id, self._on_session_event)
        
        if self.session_manager.start_session(session_id):
            self.current_session_id = session_id
            self.execution_state = ExecutionState.RUNNING
            self.status_label.config(text=f"Запуск программы (сессия: {session_id})")
        else:
            messagebox.showerror("Ошибка", "Не удалось запустить программу")
    
    def _debug_program(self):
        """Запустить отладку"""
        if self.execution_state != ExecutionState.IDLE:
            return
        
        code = self.code_editor.get_content()
        if not code.strip():
            messagebox.showwarning("Предупреждение", "Нет кода для отладки")
            return
        
        file_path = self.code_editor.get_current_file_path()
        
        # Запуск визуального отладчика
        self.visual_debugger.start_debugging(code, file_path)
        
        self.execution_state = ExecutionState.DEBUGGING
        self.status_label.config(text="Запущена отладка")
    
    def _profile_program(self):
        """Запустить профилирование"""
        if self.execution_state != ExecutionState.IDLE:
            return
        
        code = self.code_editor.get_content()
        if not code.strip():
            messagebox.showwarning("Предупреждение", "Нет кода для профилирования")
            return
        
        file_path = self.code_editor.get_current_file_path()
        
        # Создание и запуск сессии профилирования
        session_id = self.session_manager.create_session(
            file_path, code, ExecutionMode.PROFILE
        )
        
        self.session_manager.add_callback(session_id, self._on_session_event)
        
        if self.session_manager.start_session(session_id):
            self.current_session_id = session_id
            self.execution_state = ExecutionState.PROFILING
            self.status_label.config(text=f"Профилирование (сессия: {session_id})")
        else:
            messagebox.showerror("Ошибка", "Не удалось запустить профилирование")
    
    def _stop_execution(self):
        """Остановить выполнение"""
        if self.current_session_id:
            self.session_manager.stop_session(self.current_session_id)
        
        if self.execution_state == ExecutionState.DEBUGGING:
            self.visual_debugger.stop_debugging()
        
        self.execution_state = ExecutionState.IDLE
        self.current_session_id = None
        self.status_label.config(text="Выполнение остановлено")
    
    def _on_session_event(self, session_id: str, event: str, data: Any = None):
        """Обработка событий сессии"""
        # Планирование обновления в UI потоке
        self.root.after_idle(lambda: self._handle_session_event(session_id, event, data))
    
    def _handle_session_event(self, session_id: str, event: str, data: Any = None):
        """Обработка событий сессии в UI потоке"""
        if event == 'started':
            self.status_label.config(text=f"Сессия {session_id} запущена")
            
        elif event == 'finished':
            if session_id == self.current_session_id:
                self.execution_state = ExecutionState.IDLE
                self.current_session_id = None
            
            self.status_label.config(text=f"Сессия {session_id} завершена")
            self._display_session_output(session_id)
            
        elif event == 'error':
            if session_id == self.current_session_id:
                self.execution_state = ExecutionState.ERROR
            
            self.status_label.config(text=f"Ошибка в сессии {session_id}")
            messagebox.showerror("Ошибка выполнения", f"Сессия {session_id}:\n{data}")
            
        elif event == 'timeout':
            if session_id == self.current_session_id:
                self.execution_state = ExecutionState.ERROR
            
            self.status_label.config(text=f"Таймаут сессии {session_id}")
            messagebox.showwarning("Таймаут", f"Сессия {session_id} превысила лимит времени")
    
    def _display_session_output(self, session_id: str):
        """Отображение вывода сессии"""
        output = self.session_manager.get_session_output(session_id)
        
        if output:
            # Вывод в терминал
            for line in output:
                self.terminal_panel.append_output(line)
    
    # Дополнительные обработчики
    def _undo(self):
        """Отменить"""
        try:
            self.code_editor.text_widget.edit_undo()
        except tk.TclError:
            pass
    
    def _redo(self):
        """Повторить"""
        try:
            self.code_editor.text_widget.edit_redo()
        except tk.TclError:
            pass
    
    def _find(self):
        """Найти"""
        # TODO: Реализовать диалог поиска
        search_text = simpledialog.askstring("Поиск", "Введите текст для поиска:")
        if search_text:
            self.status_label.config(text=f"Поиск: {search_text}")
    
    def _replace(self):
        """Заменить"""
        # TODO: Реализовать диалог замены
        pass
    
    def _on_file_selected(self, file_path: str):
        """Обработка выбора файла"""
        if self._check_unsaved_changes():
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                self.code_editor.set_content(content)
                self.code_editor.current_file = file_path
                self.file_label.config(text=os.path.basename(file_path))
                self.status_label.config(text=f"Открыт: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")
    
    def _on_cursor_moved(self, event=None):
        """Обработка движения курсора"""
        try:
            cursor_pos = self.code_editor.text_widget.index(tk.INSERT)
            line, col = cursor_pos.split('.')
            self.cursor_label.config(text=f"Строка: {line}, Столбец: {int(col)+1}")
        except:
            pass
    
    def _check_unsaved_changes(self) -> bool:
        """Проверка несохраненных изменений"""
        # TODO: Реализовать проверку изменений
        return True
    
    def _exit_app(self):
        """Выход из приложения"""
        if self._check_unsaved_changes():
            self._on_closing()
    
    def _on_closing(self):
        """Обработка закрытия окна"""
        try:
            # Остановка всех сессий
            if self.current_session_id:
                self.session_manager.stop_session(self.current_session_id)
            
            # Очистка менеджеров
            self.session_manager.cleanup()
            self.debounce_handler.cancel_all()
            
            # Остановка визуального отладчика
            if hasattr(self, 'visual_debugger'):
                self.visual_debugger.stop_debugging()
            
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии: {e}")
            self.root.destroy()
    
    def run(self):
        """Запуск IDE"""
        try:
            self.logger.info("Запуск Enhanced AnamorphX IDE")
            self.status_label.config(text="Enhanced AnamorphX IDE готова к работе")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("IDE остановлена пользователем")
        except Exception as e:
            self.logger.error(f"Критическая ошибка IDE: {e}")
            messagebox.showerror("Критическая ошибка", f"IDE завершается из-за ошибки:\n{e}")
        finally:
            self._on_closing()


def main():
    """Главная функция запуска"""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('anamorphx_ide.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Создание и запуск IDE
        ide = EnhancedAnamorphIDE()
        ide.run()
        
    except Exception as e:
        logging.error(f"Ошибка запуска IDE: {e}")
        messagebox.showerror("Ошибка запуска", f"Не удалось запустить IDE:\n{e}")


if __name__ == "__main__":
    print("🚀 Запуск Enhanced AnamorphX IDE...")
    print("Возможности:")
    print("  ✅ Асинхронное выполнение с управлением сессиями")
    print("  ✅ Инкрементальная подсветка синтаксиса с debounce")
    print("  ✅ Визуальный отладчик с синхронизацией")
    print("  ✅ Визуализация профайлера в реальном времени")
    print("  ✅ Потокобезопасный UI")
    print("  ✅ Управление состояниями и таймауты")
    print("  ✅ Централизованное логирование")
    print()
    
    main() 