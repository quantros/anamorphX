#!/usr/bin/env python3
"""
Интеграция улучшенных файловых операций в UnifiedMLIDE
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import re

# Добавляем улучшенные методы в класс UnifiedMLIDE
def enhance_unified_ml_ide():
    """Улучшение UnifiedMLIDE с файловыми операциями"""
    
    def enhanced_new_file(self):
        """Улучшенное создание нового файла"""
        if hasattr(self, 'file_modified') and self.file_modified:
            if not self.ask_save_changes():
                return
        
        # Очистка редактора
        if hasattr(self, 'text_editor'):
            self.text_editor.delete("1.0", tk.END)
            
            # Загрузка шаблона
            template = self.get_file_template()
            self.text_editor.insert("1.0", template)
            
            # Обновление состояния
            self.current_file = None
            self.file_modified = False
            if hasattr(self, 'file_label'):
                self.file_label.config(text="📄 Untitled.anamorph")
            if hasattr(self, 'modified_label'):
                self.modified_label.config(text="")
            
            # Применение подсветки синтаксиса
            self.apply_enhanced_syntax_highlighting()
            
            self.log_to_console("📄 New AnamorphX file created")
    
    def enhanced_open_file(self):
        """Улучшенное открытие файла"""
        if hasattr(self, 'file_modified') and self.file_modified:
            if not self.ask_save_changes():
                return
        
        file_types = [
            ("AnamorphX files", "*.anamorph"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Open AnamorphX File",
            filetypes=file_types,
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
            if hasattr(self, 'text_editor'):
                self.text_editor.delete("1.0", tk.END)
                self.text_editor.insert("1.0", content)
                
                # Обновление состояния
                self.current_file = file_path
                self.file_modified = False
                filename = os.path.basename(file_path)
                
                if hasattr(self, 'file_label'):
                    self.file_label.config(text=f"📄 {filename}")
                if hasattr(self, 'modified_label'):
                    self.modified_label.config(text="")
                
                # Применение подсветки синтаксиса
                self.apply_enhanced_syntax_highlighting()
                
                # ML анализ нового файла
                self.root.after(1000, self.perform_realtime_ml_analysis)
                
                self.log_to_console(f"📁 Opened: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
            self.log_to_console(f"❌ Error opening file: {e}")
    
    def enhanced_save_file(self):
        """Улучшенное сохранение файла"""
        if hasattr(self, 'current_file') and self.current_file:
            self.save_to_file(self.current_file)
        else:
            self.enhanced_save_file_as()
    
    def enhanced_save_file_as(self):
        """Улучшенное сохранение файла как"""
        file_types = [
            ("AnamorphX files", "*.anamorph"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.asksaveasfilename(
            title="Save AnamorphX File",
            filetypes=file_types,
            defaultextension=".anamorph"
        )
        
        if file_path:
            self.save_to_file(file_path)
            self.current_file = file_path
            filename = os.path.basename(file_path)
            if hasattr(self, 'file_label'):
                self.file_label.config(text=f"📄 {filename}")
    
    def save_to_file(self, file_path):
        """Сохранение в файл"""
        try:
            if hasattr(self, 'text_editor'):
                content = self.text_editor.get("1.0", tk.END)
                
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                
                # Обновление состояния
                self.file_modified = False
                if hasattr(self, 'modified_label'):
                    self.modified_label.config(text="")
                
                filename = os.path.basename(file_path)
                self.log_to_console(f"💾 Saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
            self.log_to_console(f"❌ Error saving file: {e}")
    
    def ask_save_changes(self):
        """Запрос сохранения изменений"""
        result = messagebox.askyesnocancel(
            "Save Changes",
            "The file has been modified. Do you want to save changes?"
        )
        
        if result is True:  # Yes
            self.enhanced_save_file()
            return True
        elif result is False:  # No
            return True
        else:  # Cancel
            return False
    
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
    
    def apply_enhanced_syntax_highlighting(self):
        """Применение улучшенной подсветки синтаксиса"""
        if not hasattr(self, 'text_editor'):
            return
        
        # AnamorphX ключевые слова
        keywords = {
            'network', 'neuron', 'layer', 'activation', 'weights', 'bias',
            'optimizer', 'learning_rate', 'batch_size', 'epochs', 'loss',
            'function', 'if', 'else', 'for', 'while', 'return', 'import',
            'class', 'def', 'try', 'except', 'finally', 'with', 'as',
            'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'dropout',
            'adam', 'sgd', 'rmsprop', 'crossentropy', 'mse', 'mae',
            'print', 'range', 'len', 'load_dataset', 'save', 'train', 'evaluate'
        }
        
        # Паттерны для подсветки
        patterns = [
            (r'\b(' + '|'.join(keywords) + r')\b', 'keyword'),
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
        
        # Очистка предыдущих тегов
        for tag in ["keyword", "string", "comment", "number", "class_name", 
                   "function_call", "brace", "bracket", "paren", "operator"]:
            self.text_editor.tag_remove(tag, "1.0", tk.END)
        
        content = self.text_editor.get("1.0", tk.END)
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, tag in patterns:
                for match in re.finditer(pattern, line, re.MULTILINE):
                    start = f"{line_num}.{match.start()}"
                    end = f"{line_num}.{match.end()}"
                    self.text_editor.tag_add(tag, start, end)
    
    def enhanced_on_file_double_click(self, event):
        """Улучшенная обработка двойного клика по файлу"""
        if not hasattr(self, 'file_tree'):
            return
        
        selection = self.file_tree.selection()
        if selection:
            item = selection[0]
            values = self.file_tree.item(item, 'values')
            
            if len(values) >= 2 and values[0] == 'file':
                file_path = values[1]
                if file_path.endswith('.anamorph'):
                    self.load_file(file_path)
                    self.log_to_console(f"📄 Opened from tree: {os.path.basename(file_path)}")
    
    def enhanced_populate_file_tree(self):
        """Улучшенное заполнение дерева файлов"""
        if not hasattr(self, 'file_tree'):
            return
        
        # Очистка дерева
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Корневая папка проекта
        project_root = self.file_tree.insert("", "end", text="📁 AnamorphX ML Project", 
                                            open=True, values=("folder", ""))
        
        # Основные файлы
        sample_files = [
            ("📄 main.anamorph", "main.anamorph"),
            ("📄 neural_classifier.anamorph", "neural_classifier.anamorph"),
            ("📄 deep_network.anamorph", "deep_network.anamorph"),
        ]
        
        for file_text, file_name in sample_files:
            self.file_tree.insert(project_root, "end", text=file_text, 
                                 values=("file", file_name))
        
        # Папка моделей
        models_folder = self.file_tree.insert(project_root, "end", text="📁 models", 
                                             values=("folder", "models"))
        
        model_files = [
            ("📄 cnn_model.anamorph", "models/cnn_model.anamorph"),
            ("📄 rnn_model.anamorph", "models/rnn_model.anamorph"),
            ("📄 transformer.anamorph", "models/transformer.anamorph"),
        ]
        
        for file_text, file_path in model_files:
            self.file_tree.insert(models_folder, "end", text=file_text, 
                                 values=("file", file_path))
    
    # Добавляем методы в класс UnifiedMLIDE
    from unified_ml_ide import UnifiedMLIDE
    
    UnifiedMLIDE.enhanced_new_file = enhanced_new_file
    UnifiedMLIDE.enhanced_open_file = enhanced_open_file
    UnifiedMLIDE.enhanced_save_file = enhanced_save_file
    UnifiedMLIDE.enhanced_save_file_as = enhanced_save_file_as
    UnifiedMLIDE.load_file = load_file
    UnifiedMLIDE.save_to_file = save_to_file
    UnifiedMLIDE.ask_save_changes = ask_save_changes
    UnifiedMLIDE.get_file_template = get_file_template
    UnifiedMLIDE.apply_enhanced_syntax_highlighting = apply_enhanced_syntax_highlighting
    UnifiedMLIDE.enhanced_on_file_double_click = enhanced_on_file_double_click
    UnifiedMLIDE.enhanced_populate_file_tree = enhanced_populate_file_tree
    
    # Переопределяем базовые методы
    UnifiedMLIDE.new_file = enhanced_new_file
    UnifiedMLIDE.open_file = enhanced_open_file
    UnifiedMLIDE.save_file = enhanced_save_file
    UnifiedMLIDE.save_file_as = enhanced_save_file_as
    UnifiedMLIDE.on_file_double_click = enhanced_on_file_double_click
    UnifiedMLIDE.populate_file_tree = enhanced_populate_file_tree

# Вызываем улучшение
if __name__ == "__main__":
    enhance_unified_ml_ide()
    print("✅ Enhanced file operations integrated into UnifiedMLIDE") 