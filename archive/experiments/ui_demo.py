#!/usr/bin/env python3
"""
Интерактивная демонстрация UI инструментов разработки AnamorphX
"""

import tkinter as tk
from tkinter import ttk, Text, Canvas, messagebox, filedialog
import time
import threading
import random

class AnamorphXDemo:
    """Демонстрация IDE AnamorphX"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AnamorphX IDE - Демонстрация инструментов разработки")
        self.root.geometry("1200x800")
        
        # Переменные состояния
        self.is_debugging = False
        self.current_line = 1
        self.breakpoints = set()
        self.variables = {}
        self.profiler_data = {}
        
        self.setup_ui()
        self.load_sample_code()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Меню
        self.create_menu()
        
        # Панель инструментов
        self.create_toolbar()
        
        # Основной интерфейс
        self.create_main_interface()
        
        # Статусная строка
        self.create_status_bar()
    
    def create_menu(self):
        """Создание меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Новый", command=self.new_file)
        file_menu.add_command(label="Открыть", command=self.open_file)
        file_menu.add_command(label="Сохранить", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Выполнение
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Выполнение", menu=run_menu)
        run_menu.add_command(label="Запустить", command=self.run_code)
        run_menu.add_command(label="Отладка", command=self.debug_code)
        run_menu.add_command(label="Профилировать", command=self.profile_code)
        run_menu.add_separator()
        run_menu.add_command(label="Остановить", command=self.stop_execution)
        
        # Инструменты
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Инструменты", menu=tools_menu)
        tools_menu.add_command(label="Переключить точку останова", command=self.toggle_breakpoint)
        tools_menu.add_command(label="Очистить все точки останова", command=self.clear_breakpoints)
        tools_menu.add_separator()
        tools_menu.add_command(label="Показать переменные", command=self.show_variables)
        tools_menu.add_command(label="Показать профайлер", command=self.show_profiler)
    
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Кнопки выполнения
        ttk.Button(toolbar, text="▶ Запустить", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🐛 Отладка", command=self.debug_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Профилировать", command=self.profile_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹ Стоп", command=self.stop_execution).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Кнопки отладки
        ttk.Button(toolbar, text="▶ Шаг", command=self.debug_step).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="↳ Шаг в", command=self.debug_step_into).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="↰ Шаг из", command=self.debug_step_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏭ Продолжить", command=self.debug_continue).pack(side=tk.LEFT, padx=2)
    
    def create_main_interface(self):
        """Создание основного интерфейса"""
        # Главный PanedWindow
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель (редактор + номера строк)
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=3)
        
        # Правая панель (инструменты)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=1)
        
        self.create_editor()
        self.create_tools_panel()
    
    def create_editor(self):
        """Создание редактора кода"""
        editor_frame = ttk.Frame(self.left_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Фрейм для номеров строк и текста
        text_frame = tk.Frame(editor_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Номера строк
        self.line_numbers = Text(text_frame, width=4, padx=3, takefocus=0,
                                border=0, state='disabled', wrap='none')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Основной текстовый виджет
        self.text_editor = Text(text_frame, wrap=tk.NONE, undo=True)
        self.text_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_editor.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_editor.config(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(editor_frame, orient=tk.HORIZONTAL, command=self.text_editor.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_editor.config(xscrollcommand=h_scrollbar.set)
        
        # Настройка тегов
        self.setup_text_tags()
        
        # Привязка событий
        self.text_editor.bind('<KeyRelease>', self.on_text_change)
        self.text_editor.bind('<Button-1>', self.on_editor_click)
        self.line_numbers.bind('<Button-1>', self.on_line_number_click)
        
        # Синхронизация скроллинга
        self.text_editor.bind('<MouseWheel>', self.on_mousewheel)
        self.line_numbers.bind('<MouseWheel>', self.on_mousewheel)
    
    def setup_text_tags(self):
        """Настройка тегов для подсветки"""
        self.text_editor.tag_configure("keyword", foreground="blue", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("string", foreground="green")
        self.text_editor.tag_configure("comment", foreground="gray", font=("Consolas", 11, "italic"))
        self.text_editor.tag_configure("number", foreground="red")
        self.text_editor.tag_configure("current_line", background="lightblue")
        self.text_editor.tag_configure("breakpoint", background="red", foreground="white")
        
        self.line_numbers.tag_configure("breakpoint", background="red", foreground="white")
        self.line_numbers.tag_configure("current", background="lightblue")
    
    def create_tools_panel(self):
        """Создание панели инструментов"""
        notebook = ttk.Notebook(self.right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Панель переменных
        self.create_variables_panel(notebook)
        
        # Панель стека вызовов
        self.create_call_stack_panel(notebook)
        
        # Панель профайлера
        self.create_profiler_panel(notebook)
        
        # Консоль отладки
        self.create_debug_console(notebook)
    
    def create_variables_panel(self, parent):
        """Создание панели переменных"""
        var_frame = ttk.Frame(parent)
        parent.add(var_frame, text="Переменные")
        
        # Дерево переменных
        self.var_tree = ttk.Treeview(var_frame, columns=("value", "type"), show="tree headings")
        self.var_tree.heading("#0", text="Имя")
        self.var_tree.heading("value", text="Значение")
        self.var_tree.heading("type", text="Тип")
        self.var_tree.pack(fill=tk.BOTH, expand=True)
        
        # Кнопки управления
        var_buttons = ttk.Frame(var_frame)
        var_buttons.pack(fill=tk.X, pady=2)
        ttk.Button(var_buttons, text="Обновить", command=self.refresh_variables).pack(side=tk.LEFT, padx=2)
        ttk.Button(var_buttons, text="Добавить", command=self.add_watch).pack(side=tk.LEFT, padx=2)
    
    def create_call_stack_panel(self, parent):
        """Создание панели стека вызовов"""
        stack_frame = ttk.Frame(parent)
        parent.add(stack_frame, text="Стек вызовов")
        
        self.stack_listbox = tk.Listbox(stack_frame)
        self.stack_listbox.pack(fill=tk.BOTH, expand=True)
    
    def create_profiler_panel(self, parent):
        """Создание панели профайлера"""
        profiler_frame = ttk.Frame(parent)
        parent.add(profiler_frame, text="Профайлер")
        
        # Canvas для диаграмм
        self.profiler_canvas = Canvas(profiler_frame, bg="white", height=200)
        self.profiler_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Список функций
        prof_list_frame = ttk.Frame(profiler_frame)
        prof_list_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(prof_list_frame, text="Функции:").pack(anchor=tk.W)
        self.prof_listbox = tk.Listbox(prof_list_frame, height=6)
        self.prof_listbox.pack(fill=tk.X)
    
    def create_debug_console(self, parent):
        """Создание консоли отладки"""
        console_frame = ttk.Frame(parent)
        parent.add(console_frame, text="Консоль")
        
        # Вывод
        self.console_output = Text(console_frame, height=15, state='disabled')
        self.console_output.pack(fill=tk.BOTH, expand=True)
        
        # Ввод команд
        input_frame = ttk.Frame(console_frame)
        input_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(input_frame, text=">>> ").pack(side=tk.LEFT)
        self.console_input = ttk.Entry(input_frame)
        self.console_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.console_input.bind('<Return>', self.execute_console_command)
        
        ttk.Button(input_frame, text="Выполнить", command=self.execute_console_command).pack(side=tk.RIGHT)
    
    def create_status_bar(self):
        """Создание статусной строки"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Готов")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.line_col_label = ttk.Label(self.status_bar, text="Строка: 1, Столбец: 1")
        self.line_col_label.pack(side=tk.RIGHT, padx=5)
    
    def load_sample_code(self):
        """Загрузка примера кода"""
        sample_code = """// Пример нейронной сети на AnamorphX
neuron InputNeuron {
    activation: linear
    weights: [0.5, 0.3, 0.2]
    bias: 0.1
}

neuron HiddenNeuron {
    activation: sigmoid
    weights: [0.8, 0.6, 0.4]
    bias: 0.05
}

neuron OutputNeuron {
    activation: softmax
    weights: [0.9, 0.7, 0.1]
    bias: 0.0
}

network ClassificationNetwork {
    neurons: [InputNeuron, HiddenNeuron, OutputNeuron]
    connections: {
        InputNeuron -> HiddenNeuron,
        HiddenNeuron -> OutputNeuron
    }
    
    training: {
        algorithm: backpropagation
        learning_rate: 0.01
        epochs: 1000
        batch_size: 32
    }
}

function train_network(data, labels) {
    network = new ClassificationNetwork()
    
    for epoch in range(1000) {
        loss = 0.0
        
        for batch in data.batches(32) {
            predictions = network.forward(batch)
            loss += network.loss(predictions, labels)
            network.backward()
            network.update_weights()
        }
        
        if epoch % 100 == 0 {
            print("Epoch:", epoch, "Loss:", loss)
        }
    }
    
    return network
}

// Основная функция
function main() {
    // Загрузка данных
    data = load_dataset("iris.csv")
    X, y = data.split()
    
    // Обучение сети
    model = train_network(X, y)
    
    // Тестирование
    test_data = load_dataset("iris_test.csv")
    accuracy = model.evaluate(test_data)
    
    print("Точность модели:", accuracy)
}"""
        
        self.text_editor.insert("1.0", sample_code)
        self.update_line_numbers()
        self.highlight_syntax()
    
    def update_line_numbers(self):
        """Обновление номеров строк"""
        self.line_numbers.config(state='normal')
        self.line_numbers.delete("1.0", tk.END)
        
        line_count = int(self.text_editor.index('end-1c').split('.')[0])
        line_numbers_text = "\n".join(str(i) for i in range(1, line_count + 1))
        
        self.line_numbers.insert("1.0", line_numbers_text)
        self.line_numbers.config(state='disabled')
    
    def highlight_syntax(self):
        """Подсветка синтаксиса"""
        # Очистка предыдущей подсветки
        for tag in ["keyword", "string", "comment", "number"]:
            self.text_editor.tag_remove(tag, "1.0", tk.END)
        
        text = self.text_editor.get("1.0", tk.END)
        
        # Ключевые слова
        keywords = ["neuron", "network", "function", "activation", "weights", "bias", 
                   "training", "connections", "for", "in", "if", "return", "print", "new"]
        
        for keyword in keywords:
            start = "1.0"
            while True:
                start = self.text_editor.search(keyword, start, tk.END)
                if not start:
                    break
                end = f"{start}+{len(keyword)}c"
                self.text_editor.tag_add("keyword", start, end)
                start = end
    
    def on_text_change(self, event):
        """Обработчик изменения текста"""
        self.update_line_numbers()
        self.highlight_syntax()
        self.update_cursor_position()
    
    def on_editor_click(self, event):
        """Обработчик клика в редакторе"""
        self.update_cursor_position()
    
    def on_line_number_click(self, event):
        """Обработчик клика на номер строки"""
        line_index = self.line_numbers.index(f"@{event.x},{event.y}")
        line_num = int(line_index.split('.')[0])
        self.toggle_breakpoint_at_line(line_num)
    
    def update_cursor_position(self):
        """Обновление позиции курсора"""
        cursor_pos = self.text_editor.index(tk.INSERT)
        line, col = cursor_pos.split('.')
        self.line_col_label.config(text=f"Строка: {line}, Столбец: {int(col)+1}")
    
    def toggle_breakpoint_at_line(self, line_num):
        """Переключение точки останова на строке"""
        if line_num in self.breakpoints:
            self.breakpoints.remove(line_num)
            self.text_editor.tag_remove("breakpoint", f"{line_num}.0", f"{line_num}.end")
            self.line_numbers.tag_remove("breakpoint", f"{line_num}.0", f"{line_num}.end")
        else:
            self.breakpoints.add(line_num)
            self.text_editor.tag_add("breakpoint", f"{line_num}.0", f"{line_num}.end")
            self.line_numbers.tag_add("breakpoint", f"{line_num}.0", f"{line_num}.end")
        
        self.log_to_console(f"Точка останова {'установлена' if line_num in self.breakpoints else 'удалена'} на строке {line_num}")
    
    def run_code(self):
        """Запуск кода"""
        self.log_to_console("🚀 Запуск программы...")
        self.status_label.config(text="Выполняется...")
        
        # Симуляция выполнения
        threading.Thread(target=self.simulate_execution, daemon=True).start()
    
    def debug_code(self):
        """Отладка кода"""
        self.is_debugging = True
        self.current_line = 1
        self.log_to_console("🐛 Начало отладки...")
        self.status_label.config(text="Отладка...")
        
        # Генерация тестовых переменных
        self.variables = {
            "epoch": 0,
            "loss": 0.0,
            "learning_rate": 0.01,
            "batch_size": 32,
            "accuracy": 0.0
        }
        
        self.refresh_variables()
        self.highlight_current_line()
    
    def profile_code(self):
        """Профилирование кода"""
        self.log_to_console("📊 Начало профилирования...")
        self.status_label.config(text="Профилирование...")
        
        # Генерация данных профайлера
        self.profiler_data = {
            "forward_pass": random.uniform(0.1, 0.5),
            "backward_pass": random.uniform(0.05, 0.3),
            "weight_update": random.uniform(0.02, 0.1),
            "loss_calculation": random.uniform(0.01, 0.05)
        }
        
        self.update_profiler_display()
        
        threading.Thread(target=self.simulate_profiling, daemon=True).start()
    
    def simulate_execution(self):
        """Симуляция выполнения программы"""
        import time
        
        steps = [
            "Инициализация нейронной сети...",
            "Загрузка данных...",
            "Начало обучения...",
            "Эпоха 1/1000 - потери: 0.856",
            "Эпоха 100/1000 - потери: 0.432",
            "Эпоха 200/1000 - потери: 0.291",
            "Эпоха 500/1000 - потери: 0.123",
            "Эпоха 1000/1000 - потери: 0.045",
            "Тестирование модели...",
            "Точность: 94.2%",
            "✅ Выполнение завершено успешно!"
        ]
        
        for step in steps:
            time.sleep(1)
            self.log_to_console(step)
        
        self.status_label.config(text="Готов")
    
    def simulate_profiling(self):
        """Симуляция профилирования"""
        import time
        
        for i in range(10):
            time.sleep(0.5)
            
            # Обновление данных профайлера
            for func in self.profiler_data:
                self.profiler_data[func] = random.uniform(0.01, 0.5)
            
            # Обновление отображения в главном потоке
            self.root.after(0, self.update_profiler_display)
        
        self.root.after(0, lambda: self.status_label.config(text="Готов"))
        self.root.after(0, lambda: self.log_to_console("📊 Профилирование завершено"))
    
    def debug_step(self):
        """Шаг отладки"""
        if self.is_debugging:
            self.current_line += 1
            self.variables["epoch"] = self.current_line
            self.variables["loss"] = round(random.uniform(0.01, 1.0), 3)
            
            self.highlight_current_line()
            self.refresh_variables()
            self.log_to_console(f"Шаг: строка {self.current_line}")
    
    def debug_continue(self):
        """Продолжить выполнение до следующей точки останова"""
        if self.is_debugging:
            # Найти следующую точку останова
            next_breakpoint = None
            for bp in sorted(self.breakpoints):
                if bp > self.current_line:
                    next_breakpoint = bp
                    break
            
            if next_breakpoint:
                self.current_line = next_breakpoint
                self.highlight_current_line()
                self.log_to_console(f"Остановлено на точке останова: строка {self.current_line}")
            else:
                self.log_to_console("Точек останова не найдено, выполнение продолжается...")
    
    def highlight_current_line(self):
        """Подсветка текущей строки выполнения"""
        self.text_editor.tag_remove("current_line", "1.0", tk.END)
        self.line_numbers.tag_remove("current", "1.0", tk.END)
        
        if self.is_debugging:
            self.text_editor.tag_add("current_line", f"{self.current_line}.0", f"{self.current_line}.end")
            self.line_numbers.tag_add("current", f"{self.current_line}.0", f"{self.current_line}.end")
    
    def refresh_variables(self):
        """Обновление панели переменных"""
        # Очистка дерева
        for item in self.var_tree.get_children():
            self.var_tree.delete(item)
        
        # Добавление переменных
        for name, value in self.variables.items():
            var_type = type(value).__name__
            self.var_tree.insert("", tk.END, text=name, values=(str(value), var_type))
    
    def update_profiler_display(self):
        """Обновление отображения профайлера"""
        # Очистка canvas
        self.profiler_canvas.delete("all")
        
        if not self.profiler_data:
            return
        
        # Получение размеров
        width = self.profiler_canvas.winfo_width() or 300
        height = self.profiler_canvas.winfo_height() or 200
        
        # Рисование столбчатой диаграммы
        max_val = max(self.profiler_data.values()) if self.profiler_data else 1
        bar_width = width // len(self.profiler_data)
        
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        
        for i, (func, time_val) in enumerate(self.profiler_data.items()):
            x1 = i * bar_width + 10
            x2 = x1 + bar_width - 20
            y1 = height - 30
            y2 = height - 30 - (time_val / max_val) * (height - 60)
            
            # Столбец
            color = colors[i % len(colors)]
            self.profiler_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
            
            # Подпись функции
            self.profiler_canvas.create_text(x1 + (x2-x1)//2, height - 15, text=func[:8], font=("Arial", 8))
            
            # Значение времени
            self.profiler_canvas.create_text(x1 + (x2-x1)//2, y2 - 10, text=f"{time_val:.3f}s", font=("Arial", 8))
        
        # Обновление списка функций
        self.prof_listbox.delete(0, tk.END)
        for func, time_val in sorted(self.profiler_data.items(), key=lambda x: x[1], reverse=True):
            self.prof_listbox.insert(tk.END, f"{func}: {time_val:.3f}s")
    
    def log_to_console(self, message):
        """Вывод сообщения в консоль"""
        self.console_output.config(state='normal')
        self.console_output.insert(tk.END, f"{message}\n")
        self.console_output.see(tk.END)
        self.console_output.config(state='disabled')
    
    def execute_console_command(self, event=None):
        """Выполнение команды в консоли"""
        command = self.console_input.get().strip()
        if not command:
            return
        
        self.log_to_console(f">>> {command}")
        
        # Простые команды отладки
        if command == "vars":
            for name, value in self.variables.items():
                self.log_to_console(f"  {name} = {value}")
        elif command == "break":
            self.log_to_console(f"Точки останова: {sorted(self.breakpoints)}")
        elif command.startswith("print "):
            var_name = command[6:]
            if var_name in self.variables:
                self.log_to_console(f"  {var_name} = {self.variables[var_name]}")
            else:
                self.log_to_console(f"  Переменная '{var_name}' не найдена")
        elif command == "help":
            self.log_to_console("Доступные команды:")
            self.log_to_console("  vars - показать все переменные")
            self.log_to_console("  break - показать точки останова")
            self.log_to_console("  print <var> - показать значение переменной")
            self.log_to_console("  help - показать эту справку")
        else:
            self.log_to_console(f"  Неизвестная команда: {command}")
        
        self.console_input.delete(0, tk.END)
    
    # Заглушки для остальных методов
    def new_file(self): pass
    def open_file(self): pass
    def save_file(self): pass
    def stop_execution(self): self.is_debugging = False
    def debug_step_into(self): self.debug_step()
    def debug_step_out(self): self.debug_step()
    def toggle_breakpoint(self): 
        line = int(self.text_editor.index(tk.INSERT).split('.')[0])
        self.toggle_breakpoint_at_line(line)
    def clear_breakpoints(self): 
        self.breakpoints.clear()
        self.text_editor.tag_remove("breakpoint", "1.0", tk.END)
        self.line_numbers.tag_remove("breakpoint", "1.0", tk.END)
    def show_variables(self): pass
    def show_profiler(self): pass
    def add_watch(self): pass
    def on_mousewheel(self, event): pass
    
    def run(self):
        """Запуск демонстрации"""
        self.root.mainloop()

if __name__ == "__main__":
    print("🚀 Запуск демонстрации AnamorphX IDE...")
    demo = AnamorphXDemo()
    demo.run() 