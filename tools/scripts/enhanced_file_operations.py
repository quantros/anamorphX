#!/usr/bin/env python3
"""
Enhanced File Operations for AnamorphX IDE
Улучшенные файловые операции с поддержкой .anamorph файлов
"""

import os
import json
from pathlib import Path
from tkinter import filedialog, messagebox

class AnamorphXFileManager:
    """Менеджер файлов для AnamorphX IDE"""
    
    def __init__(self, ide_instance):
        self.ide = ide_instance
        self.current_project = None
        self.recent_files = []
        self.file_types = [
            ("AnamorphX files", "*.anamorph"),
            ("All files", "*.*")
        ]
    
    def new_file(self):
        """Создание нового файла"""
        if self.ide.file_modified:
            if not self.ask_save_changes():
                return
        
        # Очистка редактора
        self.ide.text_editor.delete("1.0", tk.END)
        
        # Загрузка шаблона
        template = self.get_file_template()
        self.ide.text_editor.insert("1.0", template)
        
        # Обновление состояния
        self.ide.current_file = None
        self.ide.file_modified = False
        self.ide.file_label.config(text="📄 Untitled.anamorph")
        self.ide.modified_label.config(text="")
        
        # Применение подсветки синтаксиса
        self.apply_syntax_highlighting()
        
        self.ide.log_to_console("📄 New AnamorphX file created")
    
    def open_file(self):
        """Открытие файла"""
        if self.ide.file_modified:
            if not self.ask_save_changes():
                return
        
        file_path = filedialog.askopenfilename(
            title="Open AnamorphX File",
            filetypes=self.file_types,
            defaultextension=".anamorph"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, file_path):
        """Загрузка файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Очистка и загрузка содержимого
            self.ide.text_editor.delete("1.0", tk.END)
            self.ide.text_editor.insert("1.0", content)
            
            # Обновление состояния
            self.ide.current_file = file_path
            self.ide.file_modified = False
            filename = os.path.basename(file_path)
            self.ide.file_label.config(text=f"📄 {filename}")
            self.ide.modified_label.config(text="")
            
            # Добавление в недавние файлы
            self.add_to_recent_files(file_path)
            
            # Применение подсветки синтаксиса
            self.apply_syntax_highlighting()
            
            # ML анализ нового файла
            self.ide.root.after(1000, self.ide.perform_realtime_ml_analysis)
            
            self.ide.log_to_console(f"📁 Opened: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
            self.ide.log_to_console(f"❌ Error opening file: {e}")
    
    def save_file(self):
        """Сохранение файла"""
        if self.ide.current_file:
            self.save_to_file(self.ide.current_file)
        else:
            self.save_file_as()
    
    def save_file_as(self):
        """Сохранение файла как"""
        file_path = filedialog.asksaveasfilename(
            title="Save AnamorphX File",
            filetypes=self.file_types,
            defaultextension=".anamorph"
        )
        
        if file_path:
            self.save_to_file(file_path)
            self.ide.current_file = file_path
            filename = os.path.basename(file_path)
            self.ide.file_label.config(text=f"📄 {filename}")
    
    def save_to_file(self, file_path):
        """Сохранение в файл"""
        try:
            content = self.ide.text_editor.get("1.0", tk.END)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            # Обновление состояния
            self.ide.file_modified = False
            self.ide.modified_label.config(text="")
            
            # Добавление в недавние файлы
            self.add_to_recent_files(file_path)
            
            filename = os.path.basename(file_path)
            self.ide.log_to_console(f"💾 Saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
            self.ide.log_to_console(f"❌ Error saving file: {e}")
    
    def ask_save_changes(self):
        """Запрос сохранения изменений"""
        result = messagebox.askyesnocancel(
            "Save Changes",
            "The file has been modified. Do you want to save changes?"
        )
        
        if result is True:  # Yes
            self.save_file()
            return True
        elif result is False:  # No
            return True
        else:  # Cancel
            return False
    
    def add_to_recent_files(self, file_path):
        """Добавление в список недавних файлов"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        
        self.recent_files.insert(0, file_path)
        
        # Ограничение списка
        if len(self.recent_files) > 10:
            self.recent_files = self.recent_files[:10]
    
    def get_file_template(self):
        """Получение шаблона нового файла"""
        return '''// AnamorphX Neural Network Example
network DeepClassifier {
    // Input layer
    neuron InputLayer {
        activation: linear
        weights: [0.5, 0.3, 0.2, 0.8]
        bias: 0.1
    }
    
    // Hidden layers
    neuron HiddenLayer1 {
        activation: relu
        weights: random_normal(0, 0.1)
        dropout: 0.2
    }
    
    neuron HiddenLayer2 {
        activation: relu
        weights: random_normal(0, 0.1)
        dropout: 0.3
    }
    
    // Output layer
    neuron OutputLayer {
        activation: softmax
        weights: random_normal(0, 0.05)
    }
    
    // Training configuration
    optimizer: adam
    learning_rate: 0.001
    loss: crossentropy
    batch_size: 32
    epochs: 100
}

// Training function
function train_model() {
    // Load training data
    data = load_dataset("training_data.csv")
    
    // Train the network
    for epoch in range(epochs) {
        loss = network.train(data)
        accuracy = network.evaluate(data)
        
        if epoch % 10 == 0 {
            print("Epoch: " + epoch + ", Loss: " + loss + ", Accuracy: " + accuracy)
        }
    }
    
    // Save the trained model
    network.save("trained_model.anamorph")
}

// Main execution
function main() {
    print("Starting AnamorphX ML training...")
    train_model()
    print("Training completed!")
}
'''
    
    def apply_syntax_highlighting(self):
        """Применение подсветки синтаксиса"""
        if hasattr(self.ide, 'syntax_highlighter'):
            self.ide.syntax_highlighter.highlight_syntax()
        else:
            # Создание подсветки синтаксиса
            self.ide.syntax_highlighter = AnamorphXSyntaxHighlighter(self.ide.text_editor)
            self.ide.syntax_highlighter.highlight_syntax()

class AnamorphXSyntaxHighlighter:
    """Улучшенная подсветка синтаксиса AnamorphX"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.setup_tags()
        
        # AnamorphX ключевые слова
        self.keywords = {
            'network', 'neuron', 'layer', 'activation', 'weights', 'bias',
            'optimizer', 'learning_rate', 'batch_size', 'epochs', 'loss',
            'function', 'if', 'else', 'for', 'while', 'return', 'import',
            'class', 'def', 'try', 'except', 'finally', 'with', 'as',
            'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'dropout',
            'adam', 'sgd', 'rmsprop', 'crossentropy', 'mse', 'mae',
            'print', 'range', 'len', 'load_dataset', 'save', 'train', 'evaluate'
        }
        
        # Паттерны для подсветки
        self.patterns = [
            (r'\b(' + '|'.join(self.keywords) + r')\b', 'keyword'),
            (r'"[^"]*"', 'string'),
            (r"'[^']*'", 'string'),
            (r'//.*$', 'comment'),
            (r'/\*.*?\*/', 'comment'),
            (r'\b\d+\.?\d*\b', 'number'),
            (r'\b[A-Z][a-zA-Z0-9_]*\b', 'class_name'),
            (r'\b[a-z_][a-zA-Z0-9_]*(?=\s*\()', 'function_call'),
            (r'\{|\}', 'brace'),
            (r'\[|\]', 'bracket'),
            (r'\(|\)', 'paren'),
            (r'[+\-*/=<>!&|]', 'operator'),
        ]
    
    def setup_tags(self):
        """Настройка тегов для подсветки"""
        # Основные теги
        self.text_widget.tag_configure("keyword", foreground="#0066CC", font=("Consolas", 11, "bold"))
        self.text_widget.tag_configure("string", foreground="#009900")
        self.text_widget.tag_configure("comment", foreground="#808080", font=("Consolas", 11, "italic"))
        self.text_widget.tag_configure("number", foreground="#FF6600")
        self.text_widget.tag_configure("class_name", foreground="#CC0066", font=("Consolas", 11, "bold"))
        self.text_widget.tag_configure("function_call", foreground="#9900CC")
        self.text_widget.tag_configure("brace", foreground="#FF0000", font=("Consolas", 11, "bold"))
        self.text_widget.tag_configure("bracket", foreground="#0066FF", font=("Consolas", 11, "bold"))
        self.text_widget.tag_configure("paren", foreground="#666666")
        self.text_widget.tag_configure("operator", foreground="#CC6600")
        
        # ML специфичные теги
        self.text_widget.tag_configure("ml_error", background="#FFCCCB", underline=True)
        self.text_widget.tag_configure("ml_warning", background="#FFE4B5", underline=True)
        self.text_widget.tag_configure("ml_optimization", background="#E0FFE0", underline=True)
        self.text_widget.tag_configure("ml_suggestion", background="#E6E6FA", underline=True)
        self.text_widget.tag_configure("ml_neural", background="#F0F8FF", underline=True)
    
    def highlight_syntax(self):
        """Применение подсветки синтаксиса"""
        # Очистка предыдущих тегов
        for tag in ["keyword", "string", "comment", "number", "class_name", 
                   "function_call", "brace", "bracket", "paren", "operator"]:
            self.text_widget.tag_remove(tag, "1.0", tk.END)
        
        content = self.text_widget.get("1.0", tk.END)
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, tag in self.patterns:
                for match in re.finditer(pattern, line, re.MULTILINE):
                    start = f"{line_num}.{match.start()}"
                    end = f"{line_num}.{match.end()}"
                    self.text_widget.tag_add(tag, start, end)

class ProjectManager:
    """Менеджер проектов AnamorphX"""
    
    def __init__(self, ide_instance):
        self.ide = ide_instance
        self.current_project = None
        self.project_config = {}
    
    def new_project(self):
        """Создание нового проекта"""
        project_dir = filedialog.askdirectory(title="Select Project Directory")
        
        if project_dir:
            project_name = os.path.basename(project_dir)
            
            # Создание структуры проекта
            self.create_project_structure(project_dir, project_name)
            
            # Загрузка проекта
            self.load_project(project_dir)
    
    def create_project_structure(self, project_dir, project_name):
        """Создание структуры проекта"""
        try:
            # Основные папки
            os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(project_dir, "tests"), exist_ok=True)
            
            # Конфигурация проекта
            config = {
                "name": project_name,
                "version": "1.0.0",
                "description": f"AnamorphX ML Project: {project_name}",
                "main_file": "main.anamorph",
                "dependencies": [],
                "ml_config": {
                    "default_optimizer": "adam",
                    "default_loss": "crossentropy",
                    "default_activation": "relu"
                }
            }
            
            config_path = os.path.join(project_dir, "project.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            # Главный файл
            main_file_path = os.path.join(project_dir, "main.anamorph")
            with open(main_file_path, 'w', encoding='utf-8') as f:
                f.write(self.get_project_template(project_name))
            
            self.ide.log_to_console(f"📁 Created project: {project_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create project: {e}")
    
    def load_project(self, project_dir):
        """Загрузка проекта"""
        try:
            config_path = os.path.join(project_dir, "project.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.project_config = json.load(f)
            
            self.current_project = project_dir
            self.ide.project_root = project_dir
            
            # Обновление файлового дерева
            self.update_file_tree()
            
            # Загрузка главного файла
            main_file = os.path.join(project_dir, self.project_config.get("main_file", "main.anamorph"))
            if os.path.exists(main_file):
                self.ide.file_manager.load_file(main_file)
            
            project_name = self.project_config.get("name", os.path.basename(project_dir))
            self.ide.log_to_console(f"📂 Loaded project: {project_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project: {e}")
    
    def update_file_tree(self):
        """Обновление дерева файлов"""
        if not hasattr(self.ide, 'file_tree'):
            return
        
        # Очистка дерева
        for item in self.ide.file_tree.get_children():
            self.ide.file_tree.delete(item)
        
        if self.current_project:
            self.populate_project_tree(self.current_project)
    
    def populate_project_tree(self, project_dir):
        """Заполнение дерева проекта"""
        project_name = os.path.basename(project_dir)
        root_item = self.ide.file_tree.insert("", "end", text=f"📁 {project_name}", 
                                             open=True, values=("folder", project_dir))
        
        self.add_directory_to_tree(project_dir, root_item)
    
    def add_directory_to_tree(self, dir_path, parent_item):
        """Добавление директории в дерево"""
        try:
            for item in sorted(os.listdir(dir_path)):
                item_path = os.path.join(dir_path, item)
                
                if os.path.isdir(item_path):
                    # Папка
                    folder_item = self.ide.file_tree.insert(parent_item, "end", 
                                                           text=f"📁 {item}", 
                                                           values=("folder", item_path))
                    self.add_directory_to_tree(item_path, folder_item)
                else:
                    # Файл
                    icon = "📄" if item.endswith(".anamorph") else "📋"
                    self.ide.file_tree.insert(parent_item, "end", 
                                             text=f"{icon} {item}", 
                                             values=("file", item_path))
        except PermissionError:
            pass
    
    def get_project_template(self, project_name):
        """Получение шаблона проекта"""
        return f'''// {project_name} - AnamorphX ML Project
// Generated by AnamorphX IDE

network {project_name}Model {{
    // Define your neural network architecture here
    
    neuron InputLayer {{
        activation: linear
        weights: random_normal(0, 0.1)
    }}
    
    neuron HiddenLayer {{
        activation: relu
        weights: random_normal(0, 0.1)
        dropout: 0.2
    }}
    
    neuron OutputLayer {{
        activation: softmax
        weights: random_normal(0, 0.05)
    }}
    
    // Training configuration
    optimizer: adam
    learning_rate: 0.001
    loss: crossentropy
    batch_size: 32
    epochs: 100
}}

function main() {{
    print("Starting {project_name} training...")
    
    // Your training code here
    
    print("Training completed!")
}}
''' 