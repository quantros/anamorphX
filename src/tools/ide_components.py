"""
IDE Components for AnamorphX

Компоненты интегрированной среды разработки для языка Anamorph:
- Редактор кода с подсветкой синтаксиса
- Файловый менеджер
- Терминал/консоль
- Панель отладки
- Менеджер проектов
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font, Text, Menu
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

from .syntax_highlighter import AnamorphSyntaxHighlighter, THEMES
from ..lexer import AnamorphLexer
from ..parser import AnamorphParser
from ..semantic import SemanticAnalyzer
from ..codegen import PythonCodeGenerator, JavaScriptCodeGenerator


@dataclass
class IDEConfig:
    """Конфигурация IDE"""
    theme: str = "dark"
    font_family: str = "Consolas"
    font_size: int = 12
    tab_size: int = 4
    auto_save: bool = True
    auto_save_interval: int = 5000  # мс
    show_line_numbers: bool = True
    word_wrap: bool = False
    syntax_highlighting: bool = True
    auto_complete: bool = True
    bracket_matching: bool = True
    code_folding: bool = True
    minimap: bool = False
    
    def save_to_file(self, filepath: str):
        """Сохранение конфигурации в файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'IDEConfig':
        """Загрузка конфигурации из файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


@dataclass
class FileTab:
    """Вкладка файла"""
    filepath: Optional[str] = None
    content: str = ""
    modified: bool = False
    language: str = "anamorph"
    cursor_position: tuple = (1, 0)
    
    @property
    def filename(self) -> str:
        """Имя файла"""
        if self.filepath:
            return os.path.basename(self.filepath)
        return "Untitled"
        
    @property
    def title(self) -> str:
        """Заголовок вкладки"""
        title = self.filename
        if self.modified:
            title += " *"
        return title


@dataclass
class FileInfo:
    """Информация о файле"""
    path: str
    name: str
    is_directory: bool
    size: Optional[int] = None
    modified: Optional[float] = None


@dataclass 
class ProjectConfig:
    """Конфигурация проекта"""
    name: str
    root_path: str
    main_file: Optional[str] = None
    build_command: Optional[str] = None
    run_command: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class FileExplorer:
    """Файловый менеджер для IDE"""
    
    def __init__(self, parent, on_file_select: Callable[[str], None]):
        self.parent = parent
        self.on_file_select = on_file_select
        self.current_path = os.getcwd()
        
        # Создание виджета дерева файлов
        self.tree = ttk.Treeview(parent, columns=('size', 'modified'), show='tree headings')
        self.tree.heading('#0', text='Имя файла')
        self.tree.heading('size', text='Размер')
        self.tree.heading('modified', text='Изменен')
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещение
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # События
        self.tree.bind('<Double-1>', self._on_double_click)
        self.tree.bind('<Button-3>', self._on_right_click)
        
        # Контекстное меню
        self.context_menu = Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Открыть", command=self._open_file)
        self.context_menu.add_command(label="Переименовать", command=self._rename_file)
        self.context_menu.add_command(label="Удалить", command=self._delete_file)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Новый файл", command=self._new_file)
        self.context_menu.add_command(label="Новая папка", command=self._new_folder)
        
        self.refresh()
    
    def set_path(self, path: str):
        """Установить текущий путь"""
        self.current_path = path
        self.refresh()
    
    def refresh(self):
        """Обновить содержимое"""
        # Очистка дерева
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Заполнение файлами и папками
        try:
            for item in sorted(os.listdir(self.current_path)):
                full_path = os.path.join(self.current_path, item)
                is_dir = os.path.isdir(full_path)
                
                if is_dir:
                    self.tree.insert('', 'end', text=f"📁 {item}", values=('', ''))
                else:
                    size = os.path.getsize(full_path)
                    modified = os.path.getmtime(full_path)
                    
                    # Определение иконки по расширению
                    icon = self._get_file_icon(item)
                    
                    self.tree.insert('', 'end', text=f"{icon} {item}", 
                                   values=(f"{size} байт", modified))
        except PermissionError:
            messagebox.showerror("Ошибка", "Нет доступа к папке")
    
    def _get_file_icon(self, filename: str) -> str:
        """Получить иконку для файла"""
        ext = os.path.splitext(filename)[1].lower()
        
        icons = {
            '.amph': '🧠',  # Файлы Anamorph
            '.py': '🐍',    # Python
            '.js': '📜',    # JavaScript
            '.html': '🌐',  # HTML
            '.css': '🎨',   # CSS
            '.json': '📋',  # JSON
            '.txt': '📄',   # Текст
            '.md': '📝',    # Markdown
            '.jpg': '🖼️',   # Изображения
            '.png': '🖼️',
            '.gif': '🖼️',
        }
        
        return icons.get(ext, '📄')
    
    def _on_double_click(self, event):
        """Обработка двойного клика"""
        item = self.tree.selection()[0]
        text = self.tree.item(item, 'text')
        filename = text.split(' ', 1)[1]  # Убираем иконку
        full_path = os.path.join(self.current_path, filename)
        
        if os.path.isdir(full_path):
            self.set_path(full_path)
        else:
            self.on_file_select(full_path)
    
    def _on_right_click(self, event):
        """Показать контекстное меню"""
        self.context_menu.post(event.x_root, event.y_root)
    
    def _open_file(self):
        """Открыть выбранный файл"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            text = self.tree.item(item, 'text')
            filename = text.split(' ', 1)[1]
            full_path = os.path.join(self.current_path, filename)
            
            if not os.path.isdir(full_path):
                self.on_file_select(full_path)
    
    def _rename_file(self):
        """Переименовать файл"""
        # TODO: Реализовать переименование
        messagebox.showinfo("Информация", "Функция в разработке")
    
    def _delete_file(self):
        """Удалить файл"""
        # TODO: Реализовать удаление
        messagebox.showinfo("Информация", "Функция в разработке")
    
    def _new_file(self):
        """Создать новый файл"""
        # TODO: Реализовать создание файла
        messagebox.showinfo("Информация", "Функция в разработке")
    
    def _new_folder(self):
        """Создать новую папку"""
        # TODO: Реализовать создание папки
        messagebox.showinfo("Информация", "Функция в разработке")


class CodeEditor:
    """Редактор кода с подсветкой синтаксиса"""
    
    def __init__(self, parent):
        self.parent = parent
        self.current_file = None
        self.is_modified = False
        
        # Главный фрейм
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill='both', expand=True)
        
        # Создание текстового виджета
        self.text_widget = ScrolledText(
            self.main_frame,
            wrap=tk.NONE,
            font=('Consolas', 12),
            undo=True,
            maxundo=50
        )
        self.text_widget.pack(fill='both', expand=True)
        
        # Подсветка синтаксиса
        self.highlighter = AnamorphSyntaxHighlighter(THEMES['dark'])
        self._setup_syntax_highlighting()
        
        # События
        self.text_widget.bind('<KeyRelease>', self._on_text_change)
        self.text_widget.bind('<Control-s>', self._save_file)
        self.text_widget.bind('<Control-o>', self._open_file)
        self.text_widget.bind('<Control-n>', self._new_file)
        
        # Номера строк
        self.line_numbers = Text(
            self.main_frame,
            width=4,
            padx=3,
            takefocus=0,
            border=0,
            background='#2D2D2D',
            foreground='#858585',
            state='disabled',
            wrap='none'
        )
        
        # Привязка скроллинга
        self.text_widget.bind('<MouseWheel>', self._on_mousewheel)
        self.text_widget.bind('<Button-4>', self._on_mousewheel)
        self.text_widget.bind('<Button-5>', self._on_mousewheel)
    
    def _setup_syntax_highlighting(self):
        """Настройка подсветки синтаксиса"""
        # Создание тегов для разных типов токенов
        for token_type, style in self.highlighter.theme.styles.items():
            tag_name = f"token_{token_type.value}"
            
            self.text_widget.tag_configure(
                tag_name,
                foreground=style.color,
                background=style.background,
                font=('Consolas', 12, 
                      ('bold' if style.bold else 'normal') + 
                      (' italic' if style.italic else ''))
            )
    
    def _on_text_change(self, event=None):
        """Обработка изменения текста"""
        self.is_modified = True
        self._update_title()
        self._highlight_syntax()
        self._update_line_numbers()
    
    def _highlight_syntax(self):
        """Применение подсветки синтаксиса"""
        # Получение текста
        content = self.text_widget.get('1.0', 'end-1c')
        
        # Очистка предыдущей подсветки
        for tag in self.text_widget.tag_names():
            if tag.startswith('token_'):
                self.text_widget.tag_delete(tag)
        
        # Токенизация и подсветка
        tokens = self.highlighter.tokenize(content)
        
        for token in tokens:
            if token.type.value in ['whitespace', 'newline']:
                continue
                
            tag_name = f"token_{token.type.value}"
            
            # Вычисление позиции в тексте
            start_pos = f"{token.line}.{token.column - 1}"
            end_pos = f"{token.line}.{token.column - 1 + len(token.value)}"
            
            self.text_widget.tag_add(tag_name, start_pos, end_pos)
    
    def _update_line_numbers(self):
        """Обновление номеров строк"""
        content = self.text_widget.get('1.0', 'end-1c')
        lines = content.split('\n')
        line_count = len(lines)
        
        # Обновление номеров строк
        self.line_numbers.config(state='normal')
        self.line_numbers.delete('1.0', 'end')
        
        for i in range(1, line_count + 1):
            self.line_numbers.insert('end', f"{i:>3}\n")
        
        self.line_numbers.config(state='disabled')
    
    def _update_title(self):
        """Обновление заголовка"""
        title = "Новый файл"
        if self.current_file:
            title = os.path.basename(self.current_file)
        
        if self.is_modified:
            title += " *"
        
        # TODO: Обновить заголовок окна
        pass
    
    def _on_mousewheel(self, event):
        """Синхронизация скроллинга с номерами строк"""
        self.line_numbers.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _save_file(self, event=None):
        """Сохранение файла"""
        if not self.current_file:
            self.current_file = filedialog.asksaveasfilename(
                defaultextension=".amph",
                filetypes=[
                    ("Anamorph files", "*.amph"),
                    ("Python files", "*.py"),
                    ("All files", "*.*")
                ]
            )
        
        if self.current_file:
            try:
                content = self.text_widget.get('1.0', 'end-1c')
                with open(self.current_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.is_modified = False
                self._update_title()
                messagebox.showinfo("Успех", "Файл сохранен")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")
    
    def _open_file(self, event=None):
        """Открытие файла"""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Anamorph files", "*.amph"),
                ("Python files", "*.py"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.open_file(filename)
    
    def _new_file(self, event=None):
        """Создание нового файла"""
        if self.is_modified:
            result = messagebox.askyesnocancel(
                "Сохранить изменения?", 
                "Файл был изменен. Сохранить изменения?"
            )
            
            if result is True:
                self._save_file()
            elif result is None:
                return
        
        self.text_widget.delete('1.0', 'end')
        self.current_file = None
        self.is_modified = False
        self._update_title()
    
    def open_file(self, filename: str):
        """Открыть файл"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.text_widget.delete('1.0', 'end')
            self.text_widget.insert('1.0', content)
            
            self.current_file = filename
            self.is_modified = False
            self._update_title()
            self._highlight_syntax()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл: {e}")
    
    def get_content(self) -> str:
        """Получить содержимое редактора"""
        return self.text_widget.get('1.0', 'end-1c')
    
    def set_content(self, content: str):
        """Установить содержимое редактора"""
        self.text_widget.delete('1.0', 'end')
        self.text_widget.insert('1.0', content)
        self._highlight_syntax()


class TerminalPanel:
    """Терминал/консоль для IDE"""
    
    def __init__(self, parent):
        self.parent = parent
        self.process = None
        
        # Основной фрейм
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill='both', expand=True)
        
        # Текстовое поле для вывода
        self.output_text = ScrolledText(
            self.frame,
            height=10,
            background='black',
            foreground='white',
            font=('Consolas', 10),
            state='disabled'
        )
        self.output_text.pack(fill='both', expand=True)
        
        # Поле ввода команд
        self.input_frame = ttk.Frame(self.frame)
        self.input_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.input_frame, text="$").pack(side='left')
        
        self.input_entry = ttk.Entry(self.input_frame)
        self.input_entry.pack(side='left', fill='x', expand=True, padx=5)
        self.input_entry.bind('<Return>', self._execute_command)
        
        # Кнопка выполнения
        ttk.Button(
            self.input_frame,
            text="Выполнить",
            command=self._execute_command
        ).pack(side='right')
        
        # История команд
        self.command_history = []
        self.history_index = -1
        
        self.input_entry.bind('<Up>', self._history_up)
        self.input_entry.bind('<Down>', self._history_down)
    
    def _execute_command(self, event=None):
        """Выполнение команды"""
        command = self.input_entry.get().strip()
        if not command:
            return
        
        # Добавление в историю
        self.command_history.append(command)
        self.history_index = len(self.command_history)
        
        # Отображение команды
        self._append_output(f"$ {command}\n", 'yellow')
        
        # Очистка поля ввода
        self.input_entry.delete(0, 'end')
        
        # Выполнение в отдельном потоке
        threading.Thread(target=self._run_command, args=(command,)).start()
    
    def _run_command(self, command: str):
        """Запуск команды в отдельном процессе"""
        try:
            # Специальные команды
            if command == 'clear':
                self._clear_output()
                return
            elif command.startswith('cd '):
                path = command[3:].strip()
                try:
                    os.chdir(path)
                    self._append_output(f"Переход в: {os.getcwd()}\n", 'green')
                except Exception as e:
                    self._append_output(f"Ошибка: {e}\n", 'red')
                return
            
            # Выполнение обычной команды
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.stdout:
                self._append_output(result.stdout, 'white')
            
            if result.stderr:
                self._append_output(result.stderr, 'red')
            
            if result.returncode != 0:
                self._append_output(f"Код возврата: {result.returncode}\n", 'red')
                
        except Exception as e:
            self._append_output(f"Ошибка выполнения: {e}\n", 'red')
    
    def _append_output(self, text: str, color: str = 'white'):
        """Добавление текста в вывод"""
        self.output_text.config(state='normal')
        
        # Цветовые теги
        if color not in self.output_text.tag_names():
            colors = {
                'white': '#FFFFFF',
                'green': '#00FF00',
                'red': '#FF0000',
                'yellow': '#FFFF00',
                'blue': '#0000FF'
            }
            self.output_text.tag_configure(color, foreground=colors.get(color, '#FFFFFF'))
        
        self.output_text.insert('end', text, color)
        self.output_text.config(state='disabled')
        self.output_text.see('end')  # Автоскролл
    
    def _clear_output(self):
        """Очистка вывода"""
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', 'end')
        self.output_text.config(state='disabled')
    
    def _history_up(self, event):
        """Навигация по истории команд (вверх)"""
        if self.command_history and self.history_index > 0:
            self.history_index -= 1
            self.input_entry.delete(0, 'end')
            self.input_entry.insert(0, self.command_history[self.history_index])
    
    def _history_down(self, event):
        """Навигация по истории команд (вниз)"""
        if self.command_history and self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self.input_entry.delete(0, 'end')
            self.input_entry.insert(0, self.command_history[self.history_index])
        elif self.history_index == len(self.command_history) - 1:
            self.history_index = len(self.command_history)
            self.input_entry.delete(0, 'end')


class AnamorphIDE:
    """Главное окно IDE для языка Anamorph"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AnamorphX IDE - Интегрированная среда разработки")
        self.root.geometry("1200x800")
        
        # Текущий проект
        self.current_project: Optional[ProjectConfig] = None
        
        # Инициализация компонентов
        self._create_menu()
        self._create_toolbar()
        self._create_layout()
        self._create_status_bar()
        
        # Настройка обработчиков
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_menu(self):
        """Создание главного меню"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # Файл
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Новый файл", command=self._new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Открыть файл", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Сохранить", command=self._save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Сохранить как...", command=self._save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Открыть проект", command=self._open_project)
        file_menu.add_command(label="Создать проект", command=self._new_project)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self._on_closing)
        
        # Правка
        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Правка", menu=edit_menu)
        edit_menu.add_command(label="Отменить", command=self._undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Повторить", command=self._redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Вырезать", command=self._cut, accelerator="Ctrl+X")
        edit_menu.add_command(label="Копировать", command=self._copy, accelerator="Ctrl+C")
        edit_menu.add_command(label="Вставить", command=self._paste, accelerator="Ctrl+V")
        
        # Вид
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Вид", menu=view_menu)
        view_menu.add_command(label="Файловый менеджер", command=self._toggle_file_explorer)
        view_menu.add_command(label="Терминал", command=self._toggle_terminal)
        view_menu.add_command(label="Отладчик", command=self._toggle_debugger)
        
        # Выполнение
        run_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Выполнение", menu=run_menu)
        run_menu.add_command(label="Запустить", command=self._run_program, accelerator="F5")
        run_menu.add_command(label="Отладка", command=self._debug_program, accelerator="F9")
        run_menu.add_command(label="Остановить", command=self._stop_program)
        
        # Помощь
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="Документация", command=self._show_docs)
        help_menu.add_command(label="О программе", command=self._show_about)
    
    def _create_toolbar(self):
        """Создание панели инструментов"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side='top', fill='x', padx=5, pady=2)
        
        # Кнопки
        ttk.Button(self.toolbar, text="📄 Новый", command=self._new_file).pack(side='left', padx=2)
        ttk.Button(self.toolbar, text="📂 Открыть", command=self._open_file).pack(side='left', padx=2)
        ttk.Button(self.toolbar, text="💾 Сохранить", command=self._save_file).pack(side='left', padx=2)
        
        ttk.Separator(self.toolbar, orient='vertical').pack(side='left', fill='y', padx=5)
        
        ttk.Button(self.toolbar, text="▶️ Запуск", command=self._run_program).pack(side='left', padx=2)
        ttk.Button(self.toolbar, text="🐛 Отладка", command=self._debug_program).pack(side='left', padx=2)
        ttk.Button(self.toolbar, text="⏹️ Стоп", command=self._stop_program).pack(side='left', padx=2)
        
        ttk.Separator(self.toolbar, orient='vertical').pack(side='left', fill='y', padx=5)
        
        # Выбор темы
        ttk.Label(self.toolbar, text="Тема:").pack(side='left', padx=5)
        self.theme_var = tk.StringVar(value="dark")
        theme_combo = ttk.Combobox(
            self.toolbar, 
            textvariable=self.theme_var,
            values=list(THEMES.keys()),
            state='readonly',
            width=15
        )
        theme_combo.pack(side='left', padx=2)
        theme_combo.bind('<<ComboboxSelected>>', self._change_theme)
    
    def _create_layout(self):
        """Создание основного макета"""
        # Главный контейнер
        self.main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        self.main_paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Левая панель (файловый менеджер)
        self.left_frame = ttk.Frame(self.main_paned, width=250)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Файловый менеджер
        ttk.Label(self.left_frame, text="Файловы менеджер", font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        self.file_explorer = FileExplorer(self.left_frame, self._on_file_select)
        
        # Центральная область
        self.center_paned = ttk.PanedWindow(self.main_paned, orient='vertical')
        self.main_paned.add(self.center_paned, weight=4)
        
        # Редактор кода
        self.editor_frame = ttk.Frame(self.center_paned)
        self.center_paned.add(self.editor_frame, weight=3)
        
        self.code_editor = CodeEditor(self.editor_frame)
        
        # Нижняя панель (терминал)
        self.bottom_frame = ttk.Frame(self.center_paned, height=200)
        self.center_paned.add(self.bottom_frame, weight=1)
        
        # Notebook для нижней панели
        self.bottom_notebook = ttk.Notebook(self.bottom_frame)
        self.bottom_notebook.pack(fill='both', expand=True)
        
        # Терминал
        self.terminal_frame = ttk.Frame(self.bottom_notebook)
        self.bottom_notebook.add(self.terminal_frame, text="Терминал")
        self.terminal = TerminalPanel(self.terminal_frame)
        
        # Панель отладки (пока пустая)
        self.debug_frame = ttk.Frame(self.bottom_notebook)
        self.bottom_notebook.add(self.debug_frame, text="Отладчик")
        ttk.Label(self.debug_frame, text="Отладчик в разработке...").pack(pady=20)
        
        # Вывод программы
        self.output_frame = ttk.Frame(self.bottom_notebook)
        self.bottom_notebook.add(self.output_frame, text="Вывод")
        
        self.output_text = ScrolledText(
            self.output_frame,
            height=8,
            background='#1E1E1E',
            foreground='#D4D4D4',
            font=('Consolas', 10)
        )
        self.output_text.pack(fill='both', expand=True)
    
    def _create_status_bar(self):
        """Создание строки состояния"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side='bottom', fill='x')
        
        self.status_label = ttk.Label(self.status_bar, text="Готов")
        self.status_label.pack(side='left', padx=5)
        
        # Информация о позиции курсора
        self.cursor_label = ttk.Label(self.status_bar, text="Строка: 1, Столбец: 1")
        self.cursor_label.pack(side='right', padx=5)
    
    def _on_file_select(self, filepath: str):
        """Обработка выбора файла"""
        self.code_editor.open_file(filepath)
        self.status_label.config(text=f"Открыт: {os.path.basename(filepath)}")
    
    def _change_theme(self, event=None):
        """Смена темы подсветки"""
        theme_name = self.theme_var.get()
        if theme_name in THEMES:
            self.code_editor.highlighter = AnamorphSyntaxHighlighter(THEMES[theme_name])
            self.code_editor._setup_syntax_highlighting()
            self.code_editor._highlight_syntax()
    
    # Обработчики меню
    def _new_file(self):
        self.code_editor._new_file()
    
    def _open_file(self):
        self.code_editor._open_file()
    
    def _save_file(self):
        self.code_editor._save_file()
    
    def _save_as_file(self):
        # TODO: Реализовать "Сохранить как"
        pass
    
    def _undo(self):
        try:
            self.code_editor.text_widget.edit_undo()
        except:
            pass
    
    def _redo(self):
        try:
            self.code_editor.text_widget.edit_redo()
        except:
            pass
    
    def _cut(self):
        try:
            self.code_editor.text_widget.event_generate("<<Cut>>")
        except:
            pass
    
    def _copy(self):
        try:
            self.code_editor.text_widget.event_generate("<<Copy>>")
        except:
            pass
    
    def _paste(self):
        try:
            self.code_editor.text_widget.event_generate("<<Paste>>")
        except:
            pass
    
    def _run_program(self):
        """Запуск программы"""
        content = self.code_editor.get_content()
        if not content.strip():
            messagebox.showwarning("Предупреждение", "Нет кода для выполнения")
            return
        
        # Временный файл для выполнения
        temp_file = "temp_program.amph"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Попытка выполнения через интерпретатор AnamorphX
            self.output_text.delete('1.0', 'end')
            self.output_text.insert('end', f"Выполнение программы...\n")
            
            # TODO: Интеграция с интерпретатором
            self.output_text.insert('end', "Интерпретатор в разработке...\n")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выполнить программу: {e}")
        finally:
            # Удаление временного файла
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _debug_program(self):
        """Отладка программы"""
        messagebox.showinfo("Информация", "Отладчик в разработке...")
    
    def _stop_program(self):
        """Остановка программы"""
        messagebox.showinfo("Информация", "Остановка программы...")
    
    def _toggle_file_explorer(self):
        """Переключение файлового менеджера"""
        pass
    
    def _toggle_terminal(self):
        """Переключение терминала"""
        pass
    
    def _toggle_debugger(self):
        """Переключение отладчика"""
        pass
    
    def _new_project(self):
        """Создание нового проекта"""
        messagebox.showinfo("Информация", "Создание проекта в разработке...")
    
    def _open_project(self):
        """Открытие проекта"""
        messagebox.showinfo("Информация", "Открытие проекта в разработке...")
    
    def _show_docs(self):
        """Показать документацию"""
        messagebox.showinfo("Документация", "Документация доступна на GitHub")
    
    def _show_about(self):
        """О программе"""
        messagebox.showinfo(
            "О программе",
            "AnamorphX IDE v1.0.0\n\n"
            "Интегрированная среда разработки\n"
            "для языка нейронного программирования Anamorph\n\n"
            "© 2024 AnamorphX Development Team"
        )
    
    def _on_closing(self):
        """Обработка закрытия окна"""
        if self.code_editor.is_modified:
            result = messagebox.askyesnocancel(
                "Сохранить изменения?",
                "Файл был изменен. Сохранить изменения перед выходом?"
            )
            
            if result is True:
                self.code_editor._save_file()
            elif result is None:
                return
        
        self.root.quit()
    
    def run(self):
        """Запуск IDE"""
        self.root.mainloop()


def launch_ide():
    """Запуск AnamorphX IDE"""
    try:
        ide = AnamorphIDE()
        ide.run()
    except Exception as e:
        print(f"Ошибка запуска IDE: {e}")


if __name__ == "__main__":
    # Запуск IDE
    launch_ide() 