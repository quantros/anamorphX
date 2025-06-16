#!/usr/bin/env python3
"""
Полный тест UI инструментов разработки AnamorphX с реальными Tkinter компонентами
"""

import unittest
import tkinter as tk
from tkinter import ttk, Text, Canvas
import threading
import time
import tempfile
import os
from unittest.mock import Mock, patch

class MockAnamorphLexer:
    """Заглушка для лексера"""
    def __init__(self):
        self.tokens = []
    
    def tokenize(self, text):
        # Простая токенизация для демонстрации
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            if word in ['neuron', 'network', 'activation']:
                tokens.append(('KEYWORD', word, i*10, i*10+len(word)))
            elif word in ['{', '}']:
                tokens.append(('DELIMITER', word, i*10, i*10+len(word)))
            else:
                tokens.append(('IDENTIFIER', word, i*10, i*10+len(word)))
        return tokens

class CodeEditor:
    """Простой редактор кода с подсветкой синтаксиса"""
    
    def __init__(self, parent):
        self.parent = parent
        self.text_widget = Text(parent, wrap=tk.NONE)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Настройка тегов для подсветки
        self.text_widget.tag_configure("keyword", foreground="blue", font=("Courier", 10, "bold"))
        self.text_widget.tag_configure("identifier", foreground="black")
        self.text_widget.tag_configure("delimiter", foreground="red")
        self.text_widget.tag_configure("current_line", background="yellow")
        self.text_widget.tag_configure("breakpoint", background="red", foreground="white")
        
        self.lexer = MockAnamorphLexer()
        self.breakpoints = set()
        
        # Привязка событий
        self.text_widget.bind('<KeyRelease>', self.on_text_change)
        self.text_widget.bind('<Button-1>', self.on_click)
    
    def on_text_change(self, event):
        """Обработчик изменения текста"""
        self.update_highlighting()
    
    def on_click(self, event):
        """Обработчик клика мыши"""
        line_num = int(self.text_widget.index(tk.INSERT).split('.')[0])
        return line_num
    
    def update_highlighting(self):
        """Обновление подсветки синтаксиса"""
        # Очистка предыдущей подсветки
        for tag in ["keyword", "identifier", "delimiter"]:
            self.text_widget.tag_remove(tag, "1.0", tk.END)
        
        # Получение текста
        text = self.text_widget.get("1.0", tk.END)
        
        # Токенизация и подсветка
        tokens = self.lexer.tokenize(text)
        for token_type, token_value, start, end in tokens:
            if token_type == 'KEYWORD':
                tag = "keyword"
            elif token_type == 'DELIMITER':
                tag = "delimiter"
            else:
                tag = "identifier"
            
            # Приблизительное позиционирование (упрощенное)
            try:
                self.text_widget.tag_add(tag, f"1.{start}", f"1.{end}")
            except:
                pass
    
    def toggle_breakpoint(self, line_num):
        """Переключение точки останова"""
        if line_num in self.breakpoints:
            self.breakpoints.remove(line_num)
            self.text_widget.tag_remove("breakpoint", f"{line_num}.0", f"{line_num}.end")
        else:
            self.breakpoints.add(line_num)
            self.text_widget.tag_add("breakpoint", f"{line_num}.0", f"{line_num}.end")
    
    def highlight_current_line(self, line_num):
        """Подсветка текущей строки"""
        self.text_widget.tag_remove("current_line", "1.0", tk.END)
        self.text_widget.tag_add("current_line", f"{line_num}.0", f"{line_num}.end")
    
    def get_text(self):
        """Получение текста"""
        return self.text_widget.get("1.0", tk.END)
    
    def set_text(self, text):
        """Установка текста"""
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert("1.0", text)
        self.update_highlighting()

class SimpleChart:
    """Простая диаграмма для визуализации данных"""
    
    def __init__(self, parent):
        self.parent = parent
        self.canvas = Canvas(parent, width=400, height=300, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def draw_bar_chart(self, data):
        """Рисование столбчатой диаграммы"""
        self.canvas.delete("all")
        
        if not data:
            return
        
        width = self.canvas.winfo_width() or 400
        height = self.canvas.winfo_height() or 300
        
        max_val = max(data.values()) if data else 1
        bar_width = width // len(data)
        
        for i, (label, value) in enumerate(data.items()):
            x1 = i * bar_width
            x2 = x1 + bar_width - 5
            y1 = height - 20
            y2 = height - 20 - (value / max_val) * (height - 40)
            
            # Рисование столбца
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue", outline="black")
            
            # Подпись
            self.canvas.create_text(x1 + bar_width//2, height - 10, text=label, font=("Arial", 8))
    
    def draw_pie_chart(self, data):
        """Рисование круговой диаграммы"""
        self.canvas.delete("all")
        
        if not data:
            return
        
        width = self.canvas.winfo_width() or 400
        height = self.canvas.winfo_height() or 300
        
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3
        
        total = sum(data.values())
        start_angle = 0
        colors = ["red", "green", "blue", "yellow", "orange", "purple"]
        
        for i, (label, value) in enumerate(data.items()):
            extent = (value / total) * 360
            color = colors[i % len(colors)]
            
            self.canvas.create_arc(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                start=start_angle, extent=extent,
                fill=color, outline="black"
            )
            
            start_angle += extent
    
    def clear(self):
        """Очистка диаграммы"""
        self.canvas.delete("all")

class VariablesPanel:
    """Панель переменных для отладки"""
    
    def __init__(self, parent):
        self.parent = parent
        self.tree = ttk.Treeview(parent, columns=("name", "value", "type"), show="headings")
        self.tree.heading("name", text="Имя")
        self.tree.heading("value", text="Значение")
        self.tree.heading("type", text="Тип")
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.variables = {}
    
    def update_variables(self, variables):
        """Обновление переменных"""
        # Очистка дерева
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Добавление переменных
        for name, value in variables.items():
            var_type = type(value).__name__
            self.tree.insert("", tk.END, values=(name, str(value), var_type))
        
        self.variables = variables.copy()

class TestUIComponents(unittest.TestCase):
    """Тесты UI компонентов"""
    
    def setUp(self):
        """Настройка тестового окружения"""
        self.root = tk.Tk()
        self.root.withdraw()  # Скрываем главное окно
        
        # Создание тестового фрейма
        self.test_frame = tk.Frame(self.root)
        self.test_frame.pack(fill=tk.BOTH, expand=True)
    
    def tearDown(self):
        """Очистка после тестов"""
        try:
            self.root.destroy()
        except:
            pass
    
    def test_code_editor_creation(self):
        """Тест создания редактора кода"""
        editor = CodeEditor(self.test_frame)
        
        self.assertIsNotNone(editor.text_widget)
        self.assertIsInstance(editor.lexer, MockAnamorphLexer)
        self.assertEqual(len(editor.breakpoints), 0)
    
    def test_code_editor_text_operations(self):
        """Тест операций с текстом в редакторе"""
        editor = CodeEditor(self.test_frame)
        
        test_text = "neuron test { activation: sigmoid }"
        editor.set_text(test_text)
        
        retrieved_text = editor.get_text().strip()
        self.assertEqual(retrieved_text, test_text)
    
    def test_breakpoint_operations(self):
        """Тест операций с точками останова"""
        editor = CodeEditor(self.test_frame)
        
        # Добавление точки останова
        editor.toggle_breakpoint(1)
        self.assertIn(1, editor.breakpoints)
        
        # Удаление точки останова
        editor.toggle_breakpoint(1)
        self.assertNotIn(1, editor.breakpoints)
    
    def test_current_line_highlighting(self):
        """Тест подсветки текущей строки"""
        editor = CodeEditor(self.test_frame)
        editor.set_text("line 1\nline 2\nline 3")
        
        # Подсветка строки
        editor.highlight_current_line(2)
        
        # Проверка применения тега
        tags = editor.text_widget.tag_names("2.0")
        self.assertIn("current_line", tags)
    
    def test_simple_chart_bar(self):
        """Тест столбчатой диаграммы"""
        chart = SimpleChart(self.test_frame)
        
        test_data = {"A": 10, "B": 20, "C": 15}
        chart.draw_bar_chart(test_data)
        
        # Проверка, что что-то нарисовано
        items = chart.canvas.find_all()
        self.assertGreater(len(items), 0)
    
    def test_simple_chart_pie(self):
        """Тест круговой диаграммы"""
        chart = SimpleChart(self.test_frame)
        
        test_data = {"Red": 30, "Green": 40, "Blue": 30}
        chart.draw_pie_chart(test_data)
        
        # Проверка, что что-то нарисовано
        items = chart.canvas.find_all()
        self.assertGreater(len(items), 0)
    
    def test_chart_clear(self):
        """Тест очистки диаграммы"""
        chart = SimpleChart(self.test_frame)
        
        # Рисование данных
        chart.draw_bar_chart({"A": 10, "B": 20})
        self.assertGreater(len(chart.canvas.find_all()), 0)
        
        # Очистка
        chart.clear()
        self.assertEqual(len(chart.canvas.find_all()), 0)
    
    def test_variables_panel(self):
        """Тест панели переменных"""
        panel = VariablesPanel(self.test_frame)
        
        test_vars = {
            "x": 42,
            "name": "test",
            "flag": True
        }
        
        panel.update_variables(test_vars)
        
        # Проверка обновления
        self.assertEqual(panel.variables, test_vars)
        
        # Проверка элементов в дереве
        items = panel.tree.get_children()
        self.assertEqual(len(items), len(test_vars))

class TestIntegratedWorkflow(unittest.TestCase):
    """Тест интегрированного рабочего процесса"""
    
    def setUp(self):
        """Настройка интегрированного тестового окружения"""
        self.root = tk.Tk()
        self.root.withdraw()
        
        # Создание главного окна приложения
        self.main_window = tk.Toplevel(self.root)
        self.main_window.title("AnamorphX IDE Test")
        self.main_window.geometry("800x600")
        self.main_window.withdraw()  # Сначала скрываем
        
        # Создание компонентов
        self.setup_components()
    
    def setup_components(self):
        """Настройка компонентов IDE"""
        # Создание главного paned window
        self.main_paned = ttk.PanedWindow(self.main_window, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель (редактор)
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=3)
        
        # Правая панель (инструменты отладки)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=1)
        
        # Редактор кода
        self.editor = CodeEditor(self.left_frame)
        
        # Notebook для правой панели
        self.right_notebook = ttk.Notebook(self.right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Панель переменных
        self.vars_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.vars_frame, text="Переменные")
        self.variables_panel = VariablesPanel(self.vars_frame)
        
        # Панель профайлера
        self.profiler_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.profiler_frame, text="Профайлер")
        self.profiler_chart = SimpleChart(self.profiler_frame)
    
    def tearDown(self):
        """Очистка после тестов"""
        try:
            self.main_window.destroy()
            self.root.destroy()
        except:
            pass
    
    def test_integrated_setup(self):
        """Тест интегрированной настройки"""
        self.assertIsNotNone(self.editor)
        self.assertIsNotNone(self.variables_panel)
        self.assertIsNotNone(self.profiler_chart)
    
    def test_complete_workflow(self):
        """Тест полного рабочего процесса"""
        # 1. Загрузка кода
        test_code = """neuron classifier {
    activation: sigmoid
    layers: [10, 5, 1]
}

network main {
    neurons: [classifier]
    training: supervised
}"""
        
        self.editor.set_text(test_code)
        
        # 2. Установка точки останова
        self.editor.toggle_breakpoint(2)
        self.assertIn(2, self.editor.breakpoints)
        
        # 3. Обновление переменных отладки
        debug_vars = {
            "activation": "sigmoid",
            "layers": [10, 5, 1],
            "epoch": 42
        }
        self.variables_panel.update_variables(debug_vars)
        
        # 4. Обновление данных профайлера
        profiler_data = {
            "forward_pass": 0.12,
            "backward_pass": 0.08,
            "optimization": 0.04
        }
        self.profiler_chart.draw_bar_chart(profiler_data)
        
        # 5. Подсветка текущей строки выполнения
        self.editor.highlight_current_line(2)
        
        # Проверки
        self.assertEqual(len(self.variables_panel.variables), 3)
        self.assertGreater(len(self.profiler_chart.canvas.find_all()), 0)
    
    def test_ui_responsiveness(self):
        """Тест отзывчивости UI"""
        # Симуляция множественных обновлений
        for i in range(10):
            self.editor.set_text(f"neuron test_{i} {{ activation: relu }}")
            self.variables_panel.update_variables({f"var_{i}": i})
            
            # Обновление UI
            self.root.update_idletasks()
        
        # Проверка финального состояния
        final_text = self.editor.get_text().strip()
        self.assertIn("test_9", final_text)

def run_performance_test():
    """Тест производительности UI"""
    print("\n🚀 Тест производительности UI компонентов...")
    
    root = tk.Tk()
    root.withdraw()
    
    # Тест редактора
    editor = CodeEditor(root)
    
    start_time = time.time()
    for i in range(1000):
        editor.set_text(f"neuron test_{i} {{ activation: sigmoid }}")
        root.update_idletasks()
    end_time = time.time()
    
    print(f"📝 Редактор: 1000 операций за {end_time - start_time:.3f}s")
    
    # Тест диаграммы
    chart = SimpleChart(root)
    
    start_time = time.time()
    for i in range(100):
        data = {f"item_{j}": j for j in range(10)}
        chart.draw_bar_chart(data)
        root.update_idletasks()
    end_time = time.time()
    
    print(f"📊 Диаграммы: 100 операций за {end_time - start_time:.3f}s")
    
    root.destroy()

if __name__ == "__main__":
    print("🧪 Запуск полных UI тестов AnamorphX...")
    print("=" * 60)
    
    # Запуск unit тестов
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Запуск тестов производительности
    run_performance_test()
    
    print("\n✅ Все тесты завершены!") 