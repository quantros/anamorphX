#!/usr/bin/env python3
"""
Упрощенная многоязычная демонстрация AnamorphX IDE
Поддержка русского и английского языков
"""

import tkinter as tk
from tkinter import ttk, Text, messagebox
import threading
import time
from i18n_system import _, set_language, get_language, get_available_languages

class MultilingualIDE:
    """Многоязычная IDE"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AnamorphX IDE - Multilingual")
        self.root.geometry("900x600")
        
        # Состояние
        self.is_running = False
        self.breakpoints = set()
        
        self.setup_ui()
        self.load_sample_code()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        self.create_menu()
        self.create_toolbar()
        self.create_main_area()
        self.create_status_bar()
    
    def create_menu(self):
        """Создание меню"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # Файл
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_file"), menu=self.file_menu)
        self.file_menu.add_command(label=_("file_new"), command=self.new_file)
        self.file_menu.add_command(label=_("file_open"), command=self.open_file)
        self.file_menu.add_command(label=_("file_save"), command=self.save_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=_("file_exit"), command=self.root.quit)
        
        # Выполнение
        self.run_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_run"), menu=self.run_menu)
        self.run_menu.add_command(label=_("run_execute"), command=self.run_code)
        self.run_menu.add_command(label=_("run_debug"), command=self.debug_code)
        self.run_menu.add_command(label=_("run_stop"), command=self.stop_execution)
        
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
        self.help_menu.add_command(label="About", command=self.show_about)
    
    def create_toolbar(self):
        """Создание панели инструментов"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Кнопки
        self.btn_run = ttk.Button(self.toolbar, text=_("btn_run"), command=self.run_code)
        self.btn_run.pack(side=tk.LEFT, padx=2)
        
        self.btn_debug = ttk.Button(self.toolbar, text=_("btn_debug"), command=self.debug_code)
        self.btn_debug.pack(side=tk.LEFT, padx=2)
        
        self.btn_stop = ttk.Button(self.toolbar, text=_("btn_stop"), command=self.stop_execution)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Выбор языка
        ttk.Label(self.toolbar, text=_("menu_language") + ":").pack(side=tk.LEFT, padx=2)
        
        self.language_var = tk.StringVar(value=get_language())
        self.language_combo = ttk.Combobox(
            self.toolbar,
            textvariable=self.language_var,
            values=list(get_available_languages().keys()),
            state="readonly",
            width=5
        )
        self.language_combo.pack(side=tk.LEFT, padx=2)
        self.language_combo.bind('<<ComboboxSelected>>', self.on_language_change)
    
    def create_main_area(self):
        """Создание основной области"""
        # Панель с вкладками
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Редактор кода
        self.editor_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.editor_frame, weight=3)
        
        self.text_editor = Text(self.editor_frame, wrap=tk.NONE, undo=True, font=("Consolas", 11))
        self.text_editor.pack(fill=tk.BOTH, expand=True)
        
        # Панель инструментов
        self.tools_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.tools_frame, weight=1)
        
        self.notebook = ttk.Notebook(self.tools_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Консоль
        self.create_console()
        
        # Переменные
        self.create_variables_panel()
    
    def create_console(self):
        """Создание консоли"""
        self.console_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.console_frame, text=_("panel_console"))
        
        self.console_output = Text(self.console_frame, height=15, state='disabled', font=("Consolas", 10))
        self.console_output.pack(fill=tk.BOTH, expand=True)
        
        # Ввод команд
        input_frame = ttk.Frame(self.console_frame)
        input_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(input_frame, text=">>> ").pack(side=tk.LEFT)
        self.console_input = ttk.Entry(input_frame)
        self.console_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.console_input.bind('<Return>', self.execute_console_command)
        
        self.btn_execute = ttk.Button(input_frame, text=_("btn_execute"), command=self.execute_console_command)
        self.btn_execute.pack(side=tk.RIGHT)
    
    def create_variables_panel(self):
        """Создание панели переменных"""
        self.var_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.var_frame, text=_("panel_variables"))
        
        self.var_tree = ttk.Treeview(self.var_frame, columns=("value", "type"), show="tree headings")
        self.var_tree.heading("#0", text=_("col_name"))
        self.var_tree.heading("value", text=_("col_value"))
        self.var_tree.heading("type", text=_("col_type"))
        self.var_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_status_bar(self):
        """Создание статусной строки"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text=_("status_ready"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.lang_label = ttk.Label(self.status_bar, text=f"Language: {get_available_languages()[get_language()]}")
        self.lang_label.pack(side=tk.RIGHT, padx=5)
    
    def change_language(self, language_code):
        """Смена языка"""
        if set_language(language_code):
            self.language_var.set(language_code)
            self.update_ui_language()
            self.log_to_console(f"Language changed to: {get_available_languages()[language_code]}")
    
    def on_language_change(self, event):
        """Обработчик смены языка"""
        selected_lang = self.language_var.get()
        self.change_language(selected_lang)
    
    def update_ui_language(self):
        """Обновление языка интерфейса"""
        # Меню
        self.menubar.entryconfig(0, label=_("menu_file"))
        self.menubar.entryconfig(1, label=_("menu_run"))
        self.menubar.entryconfig(2, label=_("menu_language"))
        self.menubar.entryconfig(3, label=_("menu_help"))
        
        # Подменю файл
        self.file_menu.entryconfig(0, label=_("file_new"))
        self.file_menu.entryconfig(1, label=_("file_open"))
        self.file_menu.entryconfig(2, label=_("file_save"))
        self.file_menu.entryconfig(4, label=_("file_exit"))
        
        # Подменю выполнение
        self.run_menu.entryconfig(0, label=_("run_execute"))
        self.run_menu.entryconfig(1, label=_("run_debug"))
        self.run_menu.entryconfig(2, label=_("run_stop"))
        
        # Кнопки
        self.btn_run.config(text=_("btn_run"))
        self.btn_debug.config(text=_("btn_debug"))
        self.btn_stop.config(text=_("btn_stop"))
        self.btn_execute.config(text=_("btn_execute"))
        
        # Панели
        self.notebook.tab(0, text=_("panel_console"))
        self.notebook.tab(1, text=_("panel_variables"))
        
        # Заголовки колонок
        self.var_tree.heading("#0", text=_("col_name"))
        self.var_tree.heading("value", text=_("col_value"))
        self.var_tree.heading("type", text=_("col_type"))
        
        # Статус
        self.status_label.config(text=_("status_ready"))
        self.lang_label.config(text=f"Language: {get_available_languages()[get_language()]}")
        
        # Обновление примера кода
        self.load_sample_code()
    
    def load_sample_code(self):
        """Загрузка примера кода"""
        if get_language() == "en":
            sample_code = """// AnamorphX Neural Network Example
neuron InputLayer {
    size: 784
    activation: linear
}

neuron HiddenLayer {
    size: 128
    activation: relu
    dropout: 0.2
}

neuron OutputLayer {
    size: 10
    activation: softmax
}

network MNISTClassifier {
    layers: [InputLayer, HiddenLayer, OutputLayer]
    optimizer: adam
    learning_rate: 0.001
}

function train_model(data, labels) {
    model = new MNISTClassifier()
    
    for epoch in range(10) {
        loss = model.train_epoch(data, labels)
        accuracy = model.evaluate(data, labels)
        
        print("Epoch:", epoch + 1)
        print("Loss:", loss)
        print("Accuracy:", accuracy)
    }
    
    return model
}

function main() {
    // Load MNIST dataset
    train_data, train_labels = load_mnist_train()
    test_data, test_labels = load_mnist_test()
    
    // Train model
    model = train_model(train_data, train_labels)
    
    // Final evaluation
    final_accuracy = model.evaluate(test_data, test_labels)
    print("Final test accuracy:", final_accuracy)
}"""
        else:
            sample_code = """// Пример нейронной сети AnamorphX
neuron InputLayer {
    size: 784
    activation: linear
}

neuron HiddenLayer {
    size: 128
    activation: relu
    dropout: 0.2
}

neuron OutputLayer {
    size: 10
    activation: softmax
}

network MNISTClassifier {
    layers: [InputLayer, HiddenLayer, OutputLayer]
    optimizer: adam
    learning_rate: 0.001
}

function train_model(data, labels) {
    model = new MNISTClassifier()
    
    for epoch in range(10) {
        loss = model.train_epoch(data, labels)
        accuracy = model.evaluate(data, labels)
        
        print("Эпоха:", epoch + 1)
        print("Потери:", loss)
        print("Точность:", accuracy)
    }
    
    return model
}

function main() {
    // Загрузка датасета MNIST
    train_data, train_labels = load_mnist_train()
    test_data, test_labels = load_mnist_test()
    
    // Обучение модели
    model = train_model(train_data, train_labels)
    
    // Финальная оценка
    final_accuracy = model.evaluate(test_data, test_labels)
    print("Финальная точность на тесте:", final_accuracy)
}"""
        
        self.text_editor.delete("1.0", tk.END)
        self.text_editor.insert("1.0", sample_code)
    
    def run_code(self):
        """Запуск кода"""
        if self.is_running:
            return
        
        self.is_running = True
        self.log_to_console(_("msg_execution_started"))
        self.status_label.config(text=_("status_running"))
        
        threading.Thread(target=self.simulate_execution, daemon=True).start()
    
    def debug_code(self):
        """Отладка кода"""
        self.log_to_console(_("msg_debug_started"))
        self.status_label.config(text=_("status_debugging"))
        
        # Добавление переменных для демонстрации
        variables = [
            ("epoch", "1", "int"),
            ("loss", "0.856", "float"),
            ("accuracy", "0.123", "float"),
            ("learning_rate", "0.001", "float"),
            ("batch_size", "32", "int")
        ]
        
        # Очистка и заполнение дерева переменных
        for item in self.var_tree.get_children():
            self.var_tree.delete(item)
        
        for name, value, var_type in variables:
            self.var_tree.insert("", "end", text=name, values=(value, var_type))
    
    def stop_execution(self):
        """Остановка выполнения"""
        self.is_running = False
        self.status_label.config(text=_("status_ready"))
        self.log_to_console("Execution stopped.")
    
    def simulate_execution(self):
        """Симуляция выполнения"""
        if get_language() == "en":
            steps = [
                "Initializing MNIST classifier...",
                "Loading training data...",
                "Starting training process...",
                "Epoch 1/10 - Loss: 2.301, Accuracy: 0.112",
                "Epoch 2/10 - Loss: 1.847, Accuracy: 0.334",
                "Epoch 5/10 - Loss: 0.523, Accuracy: 0.847",
                "Epoch 10/10 - Loss: 0.123, Accuracy: 0.967",
                "Training completed!",
                "Evaluating on test set...",
                "Final test accuracy: 96.8%",
                "✅ Execution completed successfully!"
            ]
        else:
            steps = [
                "Инициализация MNIST классификатора...",
                "Загрузка обучающих данных...",
                "Начало процесса обучения...",
                "Эпоха 1/10 - Потери: 2.301, Точность: 0.112",
                "Эпоха 2/10 - Потери: 1.847, Точность: 0.334",
                "Эпоха 5/10 - Потери: 0.523, Точность: 0.847",
                "Эпоха 10/10 - Потери: 0.123, Точность: 0.967",
                "Обучение завершено!",
                "Оценка на тестовом наборе...",
                "Финальная точность на тесте: 96.8%",
                "✅ Выполнение завершено успешно!"
            ]
        
        for step in steps:
            if not self.is_running:
                break
            time.sleep(1)
            self.root.after(0, lambda s=step: self.log_to_console(s))
        
        self.is_running = False
        self.root.after(0, lambda: self.status_label.config(text=_("status_ready")))
    
    def execute_console_command(self, event=None):
        """Выполнение команды консоли"""
        command = self.console_input.get().strip()
        if not command:
            return
        
        self.log_to_console(f">>> {command}")
        
        if command == "help":
            self.log_to_console(_("console_help"))
            self.log_to_console(f"  help - {_('console_help_cmd')}")
            self.log_to_console(f"  vars - {_('console_vars')}")
            self.log_to_console(f"  lang - show current language")
            self.log_to_console(f"  clear - clear console")
        elif command == "vars":
            self.log_to_console("Variables:")
            for child in self.var_tree.get_children():
                name = self.var_tree.item(child, "text")
                values = self.var_tree.item(child, "values")
                if values:
                    self.log_to_console(f"  {name} = {values[0]} ({values[1]})")
        elif command == "lang":
            current_lang = get_available_languages()[get_language()]
            self.log_to_console(f"Current language: {current_lang}")
        elif command == "clear":
            self.console_output.config(state='normal')
            self.console_output.delete("1.0", tk.END)
            self.console_output.config(state='disabled')
        else:
            self.log_to_console(f"Unknown command: {command}")
        
        self.console_input.delete(0, tk.END)
    
    def log_to_console(self, message):
        """Вывод в консоль"""
        self.console_output.config(state='normal')
        self.console_output.insert(tk.END, f"{message}\n")
        self.console_output.see(tk.END)
        self.console_output.config(state='disabled')
    
    # Заглушки для методов меню
    def new_file(self):
        self.text_editor.delete("1.0", tk.END)
        self.log_to_console("New file created")
    
    def open_file(self):
        self.log_to_console("Open file dialog (not implemented)")
    
    def save_file(self):
        self.log_to_console("Save file dialog (not implemented)")
    
    def show_about(self):
        about_text = "AnamorphX IDE - Multilingual Demo\n\nSupports:\n• Russian (Русский)\n• English\n\nVersion: 1.0"
        messagebox.showinfo("About AnamorphX IDE", about_text)
    
    def run(self):
        """Запуск IDE"""
        self.root.mainloop()

if __name__ == "__main__":
    print("🚀 Запуск многоязычной AnamorphX IDE...")
    ide = MultilingualIDE()
    ide.run() 