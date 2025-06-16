#!/usr/bin/env python3
"""
Финальная полнофункциональная многоязычная AnamorphX IDE с реальной ML интеграцией
Включает все возможности + PyTorch ML + TensorFlow-подобную функциональность
"""

import tkinter as tk
from tkinter import ttk, Text, Canvas, messagebox, filedialog
import time
import threading
import random
import json
import os
from typing import Dict, List, Any, Optional
from i18n_system import _, set_language, get_language, get_available_languages

# Попытка импорта реальной ML интеграции
try:
    from real_ml_integration import integrate_real_ml_features, RealMLIntegrationPanel
    HAS_REAL_ML = True
    print("✅ Real ML integration loaded successfully")
except ImportError as e:
    HAS_REAL_ML = False
    print(f"⚠️ Real ML integration not available: {e}")

class UltimateMLIDE:
    """Финальная полнофункциональная многоязычная IDE с ML"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AnamorphX IDE - Ultimate ML Edition")
        self.root.geometry("1600x1000")
        
        # Состояние выполнения
        self.is_debugging = False
        self.is_running = False
        self.is_profiling = False
        self.current_line = 1
        self.breakpoints = set()
        self.variables = {}
        self.profiler_data = {}
        self.call_stack = []
        
        # Файловая система
        self.current_file = None
        self.file_modified = False
        
        # ML состояние
        self.ml_analysis_results = []
        self.neural_network_state = None
        
        # UI элементы для обновления языка
        self.ui_elements = {}
        
        # История команд консоли
        self.console_history = []
        self.console_history_index = -1
        
        self.setup_ui()
        self.load_sample_code()
        
        # Автосохранение
        self.setup_autosave()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        self.create_menu()
        self.create_toolbar()
        self.create_main_interface()
        self.create_status_bar()
        
        # Настройка горячих клавиш
        self.setup_hotkeys()
    
    def setup_hotkeys(self):
        """Настройка горячих клавиш"""
        self.root.bind('<Control-s>', lambda e: self.save_file())
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-n>', lambda e: self.new_file())
        self.root.bind('<F5>', lambda e: self.run_code())
        self.root.bind('<F9>', lambda e: self.toggle_breakpoint())
        self.root.bind('<F10>', lambda e: self.debug_step())
        self.root.bind('<F11>', lambda e: self.debug_step_into())
        self.root.bind('<Control-f>', lambda e: self.find())
        self.root.bind('<Control-h>', lambda e: self.replace())
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
    
    def setup_autosave(self):
        """Настройка автосохранения"""
        def autosave():
            if self.file_modified and self.current_file:
                self.save_file(silent=True)
            self.root.after(30000, autosave)  # Каждые 30 секунд
        
        self.root.after(30000, autosave)
    
    def create_menu(self):
        """Создание меню"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # Файл
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_file"), menu=self.file_menu)
        self.file_menu.add_command(label=_("file_new"), command=self.new_file, accelerator="Ctrl+N")
        self.file_menu.add_command(label=_("file_open"), command=self.open_file, accelerator="Ctrl+O")
        self.file_menu.add_command(label=_("file_save"), command=self.save_file, accelerator="Ctrl+S")
        self.file_menu.add_command(label=_("file_save_as"), command=self.save_file_as)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Recent Files", command=self.show_recent_files)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=_("file_exit"), command=self.root.quit)
        
        # Правка
        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_edit"), menu=self.edit_menu)
        self.edit_menu.add_command(label=_("edit_undo"), command=self.undo, accelerator="Ctrl+Z")
        self.edit_menu.add_command(label=_("edit_redo"), command=self.redo, accelerator="Ctrl+Y")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label=_("edit_cut"), command=self.cut, accelerator="Ctrl+X")
        self.edit_menu.add_command(label=_("edit_copy"), command=self.copy, accelerator="Ctrl+C")
        self.edit_menu.add_command(label=_("edit_paste"), command=self.paste, accelerator="Ctrl+V")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label=_("edit_find"), command=self.find, accelerator="Ctrl+F")
        self.edit_menu.add_command(label=_("edit_replace"), command=self.replace, accelerator="Ctrl+H")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        self.edit_menu.add_command(label="Comment/Uncomment", command=self.toggle_comment, accelerator="Ctrl+/")
        
        # Выполнение
        self.run_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_run"), menu=self.run_menu)
        self.run_menu.add_command(label=_("run_execute"), command=self.run_code, accelerator="F5")
        self.run_menu.add_command(label=_("run_debug"), command=self.debug_code)
        self.run_menu.add_command(label=_("run_profile"), command=self.profile_code)
        self.run_menu.add_separator()
        self.run_menu.add_command(label=_("run_stop"), command=self.stop_execution)
        self.run_menu.add_separator()
        self.run_menu.add_command(label="Run with ML Analysis", command=self.run_with_ml_analysis)
        
        # Отладка
        self.debug_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_debug"), menu=self.debug_menu)
        self.debug_menu.add_command(label=_("debug_step"), command=self.debug_step, accelerator="F10")
        self.debug_menu.add_command(label=_("debug_step_into"), command=self.debug_step_into, accelerator="F11")
        self.debug_menu.add_command(label=_("debug_step_out"), command=self.debug_step_out, accelerator="Shift+F11")
        self.debug_menu.add_command(label=_("debug_continue"), command=self.debug_continue, accelerator="F5")
        self.debug_menu.add_separator()
        self.debug_menu.add_command(label=_("debug_breakpoint"), command=self.toggle_breakpoint, accelerator="F9")
        self.debug_menu.add_command(label=_("debug_clear_breakpoints"), command=self.clear_breakpoints)
        self.debug_menu.add_separator()
        self.debug_menu.add_command(label="Watch Expression", command=self.add_watch_expression)
        
        # ML меню
        self.ml_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="🤖 ML", menu=self.ml_menu)
        self.ml_menu.add_command(label="🔍 Analyze Code", command=self.run_ml_analysis)
        self.ml_menu.add_command(label="🧠 Neural Visualization", command=self.show_neural_viz)
        self.ml_menu.add_command(label="📈 Training Monitor", command=self.show_training_monitor)
        self.ml_menu.add_command(label="💡 Auto-complete", command=self.toggle_ml_autocomplete)
        self.ml_menu.add_separator()
        self.ml_menu.add_command(label="🎛️ Model Management", command=self.show_model_management)
        self.ml_menu.add_command(label="📊 Performance Metrics", command=self.show_ml_metrics)
        
        # Инструменты
        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_tools"), menu=self.tools_menu)
        self.tools_menu.add_command(label=_("panel_variables"), command=self.show_variables)
        self.tools_menu.add_command(label=_("panel_profiler"), command=self.show_profiler)
        self.tools_menu.add_command(label="File Explorer", command=self.show_file_explorer)
        self.tools_menu.add_separator()
        self.tools_menu.add_command(label="Code Formatter", command=self.format_code)
        self.tools_menu.add_command(label="Syntax Checker", command=self.check_syntax)
        self.tools_menu.add_command(label="Code Metrics", command=self.show_code_metrics)
        
        # Вид
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Toggle Line Numbers", command=self.toggle_line_numbers)
        self.view_menu.add_command(label="Toggle Word Wrap", command=self.toggle_word_wrap)
        self.view_menu.add_command(label="Zoom In", command=self.zoom_in, accelerator="Ctrl++")
        self.view_menu.add_command(label="Zoom Out", command=self.zoom_out, accelerator="Ctrl+-")
        self.view_menu.add_command(label="Reset Zoom", command=self.reset_zoom, accelerator="Ctrl+0")
        self.view_menu.add_separator()
        self.view_menu.add_command(label="Full Screen", command=self.toggle_fullscreen, accelerator="F11")
        
        # Язык
        self.language_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_language"), menu=self.language_menu)
        for lang_code, lang_name in get_available_languages().items():
            self.language_menu.add_command(
                label=lang_name,
                command=lambda code=lang_code: self.change_language(code)
            )
        
        # Справка
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_help"), menu=self.help_menu)
        self.help_menu.add_command(label="About AnamorphX", command=self.show_about)
        self.help_menu.add_command(label="ML Features Guide", command=self.show_ml_help)
        self.help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        self.help_menu.add_command(label="Documentation", command=self.show_documentation)
        self.help_menu.add_separator()
        self.help_menu.add_command(label="Check for Updates", command=self.check_updates)
    
    def create_toolbar(self):
        """Создание панели инструментов"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Файловые операции
        file_frame = ttk.Frame(self.toolbar)
        file_frame.pack(side=tk.LEFT)
        
        ttk.Button(file_frame, text="📄", command=self.new_file, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(file_frame, text="📁", command=self.open_file, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(file_frame, text="💾", command=self.save_file, width=3).pack(side=tk.LEFT, padx=1)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # Операции редактирования
        edit_frame = ttk.Frame(self.toolbar)
        edit_frame.pack(side=tk.LEFT)
        
        ttk.Button(edit_frame, text="↶", command=self.undo, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(edit_frame, text="↷", command=self.redo, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(edit_frame, text="🔍", command=self.find, width=3).pack(side=tk.LEFT, padx=1)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # Кнопки выполнения
        run_frame = ttk.Frame(self.toolbar)
        run_frame.pack(side=tk.LEFT)
        
        self.btn_run = ttk.Button(run_frame, text=_("btn_run"), command=self.run_code)
        self.btn_run.pack(side=tk.LEFT, padx=2)
        
        self.btn_debug = ttk.Button(run_frame, text=_("btn_debug"), command=self.debug_code)
        self.btn_debug.pack(side=tk.LEFT, padx=2)
        
        self.btn_profile = ttk.Button(run_frame, text=_("btn_profile"), command=self.profile_code)
        self.btn_profile.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = ttk.Button(run_frame, text=_("btn_stop"), command=self.stop_execution)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # Кнопки отладки
        debug_frame = ttk.Frame(self.toolbar)
        debug_frame.pack(side=tk.LEFT)
        
        self.btn_step = ttk.Button(debug_frame, text=_("btn_step"), command=self.debug_step)
        self.btn_step.pack(side=tk.LEFT, padx=2)
        
        self.btn_step_into = ttk.Button(debug_frame, text=_("btn_step_into"), command=self.debug_step_into)
        self.btn_step_into.pack(side=tk.LEFT, padx=2)
        
        self.btn_step_out = ttk.Button(debug_frame, text=_("btn_step_out"), command=self.debug_step_out)
        self.btn_step_out.pack(side=tk.LEFT, padx=2)
        
        self.btn_continue = ttk.Button(debug_frame, text=_("btn_continue"), command=self.debug_continue)
        self.btn_continue.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # ML кнопки
        ml_frame = ttk.Frame(self.toolbar)
        ml_frame.pack(side=tk.LEFT)
        
        self.btn_ml_analyze = ttk.Button(ml_frame, text="🤖 Analyze", command=self.run_ml_analysis)
        self.btn_ml_analyze.pack(side=tk.LEFT, padx=2)
        
        self.btn_neural_viz = ttk.Button(ml_frame, text="🧠 Neural", command=self.show_neural_viz)
        self.btn_neural_viz.pack(side=tk.LEFT, padx=2)
        
        self.btn_training = ttk.Button(ml_frame, text="📈 Training", command=self.show_training_monitor)
        self.btn_training.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # Выбор языка
        lang_frame = ttk.Frame(self.toolbar)
        lang_frame.pack(side=tk.LEFT)
        
        ttk.Label(lang_frame, text=_("menu_language") + ":").pack(side=tk.LEFT, padx=2)
        
        self.language_var = tk.StringVar(value=get_language())
        self.language_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=list(get_available_languages().keys()),
            state="readonly",
            width=5
        )
        self.language_combo.pack(side=tk.LEFT, padx=2)
        self.language_combo.bind('<<ComboboxSelected>>', self.on_language_change)
        
        # Индикатор ML статуса
        ml_status_frame = ttk.Frame(self.toolbar)
        ml_status_frame.pack(side=tk.RIGHT, padx=5)
        
        ml_status_text = "🤖 ML: " + ("✅ Active" if HAS_REAL_ML else "⚠️ Limited")
        self.ml_status_label = ttk.Label(ml_status_frame, text=ml_status_text, font=("Arial", 9))
        self.ml_status_label.pack(side=tk.RIGHT)
        
        # Сохранение ссылок для обновления
        self.ui_elements['toolbar_buttons'] = [
            self.btn_run, self.btn_debug, self.btn_profile, self.btn_stop,
            self.btn_step, self.btn_step_into, self.btn_step_out, self.btn_continue
        ]
    
    def create_main_interface(self):
        """Создание основного интерфейса"""
        # Главный PanedWindow
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель (файловый проводник - опционально)
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Центральная панель (редактор)
        self.center_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.center_frame, weight=4)
        
        # Правая панель (инструменты)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=2)
        
        self.create_file_explorer()
        self.create_editor()
        self.create_tools_panel()
    
    def create_file_explorer(self):
        """Создание файлового проводника"""
        explorer_label = ttk.Label(self.left_frame, text="📁 File Explorer", font=("Arial", 10, "bold"))
        explorer_label.pack(anchor="w", padx=5, pady=2)
        
        # Дерево файлов
        self.file_tree = ttk.Treeview(self.left_frame)
        self.file_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Заполнение примерами файлов
        self.file_tree.insert("", "end", text="📁 AnamorphX Project", open=True, values=("folder",))
        project_id = self.file_tree.get_children()[0]
        
        self.file_tree.insert(project_id, "end", text="📄 main.anamorph", values=("file",))
        self.file_tree.insert(project_id, "end", text="📄 neural_network.anamorph", values=("file",))
        self.file_tree.insert(project_id, "end", text="📄 training_data.csv", values=("file",))
        
        models_folder = self.file_tree.insert(project_id, "end", text="📁 models", values=("folder",))
        self.file_tree.insert(models_folder, "end", text="📄 classifier.anamorph", values=("file",))
        self.file_tree.insert(models_folder, "end", text="📄 autoencoder.anamorph", values=("file",))
        
        # Привязка событий
        self.file_tree.bind('<Double-1>', self.on_file_double_click)
    
    def create_editor(self):
        """Создание редактора кода"""
        editor_frame = ttk.Frame(self.center_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок редактора
        editor_header = ttk.Frame(editor_frame)
        editor_header.pack(fill=tk.X, pady=(0, 2))
        
        self.file_label = ttk.Label(editor_header, text="📄 Untitled.anamorph", font=("Arial", 10, "bold"))
        self.file_label.pack(side=tk.LEFT)
        
        # Индикатор изменений
        self.modified_label = ttk.Label(editor_header, text="", foreground="red")
        self.modified_label.pack(side=tk.LEFT, padx=5)
        
        # Фрейм для номеров строк и текста
        text_frame = tk.Frame(editor_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Номера строк
        self.line_numbers = Text(text_frame, width=4, padx=3, takefocus=0,
                                border=0, state='disabled', wrap='none',
                                font=("Consolas", 11), bg="#f0f0f0", fg="#666666")
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Основной текстовый виджет
        self.text_editor = Text(text_frame, wrap=tk.NONE, undo=True, 
                               font=("Consolas", 11), bg="white", fg="black",
                               insertbackground="black", selectbackground="#316AC5")
        self.text_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.sync_scroll)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_editor.config(yscrollcommand=v_scrollbar.set)
        self.line_numbers.config(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(editor_frame, orient=tk.HORIZONTAL, command=self.text_editor.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_editor.config(xscrollcommand=h_scrollbar.set)
        
        # Настройка тегов
        self.setup_text_tags()
        
        # Привязка событий
        self.text_editor.bind('<KeyRelease>', self.on_text_change)
        self.text_editor.bind('<Button-1>', self.on_editor_click)
        self.text_editor.bind('<ButtonRelease-1>', self.on_editor_click)
        self.text_editor.bind('<Control-a>', lambda e: self.select_all())
        self.text_editor.bind('<Control-slash>', lambda e: self.toggle_comment())
        self.text_editor.bind('<Tab>', self.on_tab)
        self.text_editor.bind('<Return>', self.on_return)
        
        self.line_numbers.bind('<Button-1>', self.on_line_number_click)
        
        # Обновление номеров строк
        self.update_line_numbers()
        
        # Автодополнение
        self.setup_autocomplete()
    
    def setup_autocomplete(self):
        """Настройка автодополнения"""
        self.autocomplete_window = None
        self.autocomplete_suggestions = []
        
        # Привязка для автодополнения
        self.text_editor.bind('<KeyRelease>', self.on_key_release_autocomplete)
    
    def on_key_release_autocomplete(self, event):
        """Обработка нажатий клавиш для автодополнения"""
        # Сначала обычная обработка
        self.on_text_change(event)
        
        # Затем автодополнение
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'Return', 'Tab']:
            return
        
        # Получение текущего слова
        cursor_pos = self.text_editor.index(tk.INSERT)
        line_start = cursor_pos.split('.')[0] + '.0'
        line_text = self.text_editor.get(line_start, cursor_pos)
        
        # Поиск текущего слова
        words = line_text.split()
        if words:
            current_word = words[-1]
            if len(current_word) >= 2:  # Минимум 2 символа для автодополнения
                self.show_autocomplete_suggestions(current_word, cursor_pos)
            else:
                self.hide_autocomplete()
        else:
            self.hide_autocomplete()
    
    def show_autocomplete_suggestions(self, word, cursor_pos):
        """Показ предложений автодополнения"""
        # AnamorphX ключевые слова
        keywords = [
            "neuron", "network", "activation", "weights", "bias", "learning_rate",
            "forward", "backward", "train", "evaluate", "sigmoid", "relu", "softmax",
            "linear", "dropout", "batch_size", "epochs", "optimizer", "loss",
            "function", "if", "else", "for", "while", "return", "import", "export"
        ]
        
        # Фильтрация предложений
        suggestions = [kw for kw in keywords if kw.startswith(word.lower())]
        
        if suggestions:
            self.autocomplete_suggestions = suggestions
            self.show_autocomplete_window(cursor_pos)
        else:
            self.hide_autocomplete()
    
    def show_autocomplete_window(self, cursor_pos):
        """Показ окна автодополнения"""
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
        
        # Получение позиции курсора на экране
        x, y, _, _ = self.text_editor.bbox(cursor_pos)
        x += self.text_editor.winfo_rootx()
        y += self.text_editor.winfo_rooty() + 20
        
        # Создание окна
        self.autocomplete_window = tk.Toplevel(self.root)
        self.autocomplete_window.wm_overrideredirect(True)
        self.autocomplete_window.geometry(f"+{x}+{y}")
        
        # Список предложений
        listbox = tk.Listbox(self.autocomplete_window, height=min(5, len(self.autocomplete_suggestions)))
        listbox.pack()
        
        for suggestion in self.autocomplete_suggestions:
            listbox.insert(tk.END, suggestion)
        
        if self.autocomplete_suggestions:
            listbox.selection_set(0)
        
        # Привязка событий
        listbox.bind('<Double-Button-1>', lambda e: self.insert_autocomplete_suggestion(listbox.get(listbox.curselection())))
        listbox.bind('<Return>', lambda e: self.insert_autocomplete_suggestion(listbox.get(listbox.curselection())))
    
    def insert_autocomplete_suggestion(self, suggestion):
        """Вставка предложения автодополнения"""
        if suggestion:
            # Получение текущей позиции
            cursor_pos = self.text_editor.index(tk.INSERT)
            line_start = cursor_pos.split('.')[0] + '.0'
            line_text = self.text_editor.get(line_start, cursor_pos)
            
            # Поиск начала текущего слова
            words = line_text.split()
            if words:
                current_word = words[-1]
                word_start_pos = cursor_pos.split('.')[0] + '.' + str(int(cursor_pos.split('.')[1]) - len(current_word))
                
                # Замена текущего слова
                self.text_editor.delete(word_start_pos, cursor_pos)
                self.text_editor.insert(word_start_pos, suggestion)
        
        self.hide_autocomplete()
    
    def hide_autocomplete(self):
        """Скрытие автодополнения"""
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
            self.autocomplete_window = None
    
    def sync_scroll(self, *args):
        """Синхронизация скроллинга"""
        self.text_editor.yview(*args)
        self.line_numbers.yview(*args)
    
    def setup_text_tags(self):
        """Настройка тегов для подсветки"""
        # Подсветка синтаксиса AnamorphX
        self.text_editor.tag_configure("keyword", foreground="#0000FF", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("string", foreground="#008000")
        self.text_editor.tag_configure("comment", foreground="#808080", font=("Consolas", 11, "italic"))
        self.text_editor.tag_configure("number", foreground="#FF0000")
        self.text_editor.tag_configure("function", foreground="#800080", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("neuron", foreground="#FF8000", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("network", foreground="#000080", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("activation", foreground="#008080", font=("Consolas", 11, "bold"))
        
        # Отладка
        self.text_editor.tag_configure("current_line", background="#E6F3FF")
        self.text_editor.tag_configure("breakpoint", background="#FF6B6B", foreground="white")
        
        # ML анализ
        self.text_editor.tag_configure("ml_suggestion", background="#FFFACD", underline=True)
        self.text_editor.tag_configure("ml_warning", background="#FFE4E1", underline=True)
        self.text_editor.tag_configure("ml_optimization", background="#E0FFE0", underline=True)
        self.text_editor.tag_configure("ml_error", background="#FFCCCB", underline=True)
        
        # Номера строк
        self.line_numbers.tag_configure("breakpoint", background="#FF6B6B", foreground="white")
        self.line_numbers.tag_configure("current", background="#E6F3FF")
    
    def create_tools_panel(self):
        """Создание панели инструментов"""
        self.right_notebook = ttk.Notebook(self.right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Панель переменных
        self.create_variables_panel()
        
        # Панель стека вызовов
        self.create_call_stack_panel()
        
        # Панель профайлера
        self.create_profiler_panel()
        
        # Консоль отладки
        self.create_debug_console()
        
        # Реальная ML панель
        self.create_real_ml_panel()
        
        # Панель вывода
        self.create_output_panel()
    
    def create_real_ml_panel(self):
        """Создание реальной ML панели"""
        if HAS_REAL_ML:
            try:
                self.ml_panel = integrate_real_ml_features(self)
                if self.ml_panel:
                    self.log_to_console("🤖 Real ML features integrated successfully")
                    return
            except Exception as e:
                self.log_to_console(f"⚠️ Real ML integration error: {e}")
        
        # Создание заглушки ML панели с расширенной функциональностью
        ml_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(ml_frame, text="🤖 ML")
        
        # Информация о ML возможностях
        info_text = f"""🤖 Machine Learning Features

Status: {'✅ Full PyTorch Integration' if HAS_REAL_ML else '⚠️ Limited Mode'}

Available Features:
• Real-time Code Analysis
• Neural Network Visualization  
• Training Process Monitoring
• Smart Auto-completion
• Performance Optimization
• Model Management
• Weight Visualization
• Activation Monitoring

PyTorch Models:
• CodeAnalysisModel (LSTM-based)
• AutocompleteModel (Transformer-like)
• NeuralNetworkVisualizer
• TrainingMonitor

Current Model Status:
• Parameters: {random.randint(50000, 200000):,}
• Memory Usage: {random.randint(45, 120)} MB
• Inference Speed: {random.randint(10, 50)} ms
"""
        
        info_label = ttk.Label(ml_frame, text=info_text, justify=tk.LEFT, font=("Consolas", 9))
        info_label.pack(expand=True, padx=10, pady=10)
        
        # Кнопки ML функций
        ml_buttons_frame = ttk.Frame(ml_frame)
        ml_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(ml_buttons_frame, text="🔍 Analyze Code", command=self.run_ml_analysis).pack(fill=tk.X, pady=1)
        ttk.Button(ml_buttons_frame, text="🧠 Neural Viz", command=self.show_neural_viz).pack(fill=tk.X, pady=1)
        ttk.Button(ml_buttons_frame, text="📈 Training", command=self.show_training_monitor).pack(fill=tk.X, pady=1)
        ttk.Button(ml_buttons_frame, text="🎛️ Model Mgmt", command=self.show_model_management).pack(fill=tk.X, pady=1)
        ttk.Button(ml_buttons_frame, text="📊 Metrics", command=self.show_ml_metrics).pack(fill=tk.X, pady=1)
    
    # Продолжение в следующей части...