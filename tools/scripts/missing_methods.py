    # ML методы - основная функциональность
    
    def on_ml_text_change(self, event=None):
        """Обработка изменения текста с ML анализом"""
        self.file_modified = True
        if hasattr(self, 'modified_label'):
            self.modified_label.config(text="●")
        self.update_line_numbers()
        self.update_cursor_position()
        
        # Обновление ML статуса
        if hasattr(self, 'ml_file_status'):
            self.ml_file_status.config(text="🤖 ML: Analyzing...", foreground="orange")
        
        # Планирование ML анализа
        if hasattr(self, 'ml_analysis_timer'):
            self.root.after_cancel(self.ml_analysis_timer)
        
        self.ml_analysis_timer = self.root.after(1500, self.perform_realtime_ml_analysis)
    
    def perform_realtime_ml_analysis(self):
        """Выполнение ML анализа в реальном времени"""
        if not self.ml_engine.auto_analysis_enabled:
            return
        
        start_time = time.time()
        
        try:
            code = self.text_editor.get("1.0", tk.END)
            self.ml_analysis_results = self.ml_engine.analyze_code_realtime(code)
            
            # Обновление подсветки
            self.update_ml_highlights()
            
            # Обновление дерева анализа
            self.update_ml_analysis_tree()
            
            # Обновление статистики
            self.update_analysis_statistics()
            
            # Обновление статуса
            analysis_time = (time.time() - start_time) * 1000
            if hasattr(self, 'ml_file_status'):
                self.ml_file_status.config(text=f"🤖 ML: Ready ({len(self.ml_analysis_results)} issues)", foreground="green")
            if hasattr(self, 'ml_perf_label'):
                self.ml_perf_label.config(text=f"⚡ ML: {analysis_time:.1f}ms")
            
        except Exception as e:
            self.log_to_console(f"ML analysis error: {e}")
            if hasattr(self, 'ml_file_status'):
                self.ml_file_status.config(text="🤖 ML: Error", foreground="red")
    
    def update_ml_highlights(self):
        """Обновление ML подсветки в редакторе"""
        if not hasattr(self, 'text_editor'):
            return
            
        # Очистка предыдущих ML тегов
        for tag in ["ml_error", "ml_warning", "ml_optimization", "ml_suggestion", "ml_neural"]:
            self.text_editor.tag_remove(tag, "1.0", tk.END)
            if hasattr(self, 'line_numbers'):
                self.line_numbers.tag_remove("ml_issue", "1.0", tk.END)
        
        # Применение новых тегов
        for result in self.ml_analysis_results:
            line_start = f"{result.line_number}.0"
            line_end = f"{result.line_number}.end"
            
            # Выбор тега по типу проблемы
            if result.issue_type == "error":
                tag = "ml_error"
            elif result.issue_type == "warning":
                tag = "ml_warning"
            elif result.issue_type == "optimization":
                tag = "ml_optimization"
            elif result.issue_type == "neural":
                tag = "ml_neural"
            else:
                tag = "ml_suggestion"
            
            # Применение тега к строке
            self.text_editor.tag_add(tag, line_start, line_end)
            
            # Отметка в номерах строк
            if hasattr(self, 'line_numbers'):
                line_num_start = f"{result.line_number}.0"
                line_num_end = f"{result.line_number}.end"
                self.line_numbers.tag_add("ml_issue", line_num_start, line_num_end)
    
    def update_ml_analysis_tree(self):
        """Обновление дерева ML анализа"""
        if not hasattr(self, 'ml_analysis_tree'):
            return
            
        # Очистка дерева
        for item in self.ml_analysis_tree.get_children():
            self.ml_analysis_tree.delete(item)
        
        # Добавление результатов
        for result in self.ml_analysis_results:
            # Иконка по типу
            if result.issue_type == "error":
                icon = "❌"
            elif result.issue_type == "warning":
                icon = "⚠️"
            elif result.issue_type == "optimization":
                icon = "⚡"
            elif result.issue_type == "neural":
                icon = "🧠"
            else:
                icon = "💡"
            
            self.ml_analysis_tree.insert("", "end",
                text=f"{icon} Line {result.line_number}",
                values=(result.issue_type, result.severity, f"{result.confidence:.2f}", result.suggestion)
            )
    
    def update_analysis_statistics(self):
        """Обновление статистики анализа"""
        if not hasattr(self, 'analysis_stats_label'):
            return
            
        if not self.ml_analysis_results:
            self.analysis_stats_label.config(text="No issues found ✅")
            return
        
        # Подсчет по типам
        error_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "error")
        warning_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "warning")
        optimization_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "optimization")
        neural_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "neural")
        
        stats_text = f"❌ {error_count} errors, ⚠️ {warning_count} warnings, ⚡ {optimization_count} optimizations, 🧠 {neural_count} neural"
        self.analysis_stats_label.config(text=stats_text)
    
    def trigger_ml_autocomplete(self, event):
        """Запуск ML автодополнения"""
        if not hasattr(self, 'text_editor'):
            return "break"
            
        cursor_pos = self.text_editor.index(tk.INSERT)
        context = self.text_editor.get("1.0", cursor_pos)
        
        # Получение ML предложений
        suggestions = self.ml_engine.get_autocomplete_suggestions(context, len(context))
        
        if suggestions:
            self.show_ml_autocomplete_window(suggestions, cursor_pos)
        
        return "break"
    
    def show_ml_autocomplete_window(self, suggestions, cursor_pos):
        """Показ окна ML автодополнения"""
        if hasattr(self, 'autocomplete_window') and self.autocomplete_window:
            self.autocomplete_window.destroy()
        
        # Получение позиции курсора на экране
        try:
            x, y, _, _ = self.text_editor.bbox(cursor_pos)
            x += self.text_editor.winfo_rootx()
            y += self.text_editor.winfo_rooty() + 20
        except:
            return
        
        # Создание окна
        self.autocomplete_window = tk.Toplevel(self.root)
        self.autocomplete_window.wm_overrideredirect(True)
        self.autocomplete_window.geometry(f"+{x}+{y}")
        
        # Заголовок
        header = tk.Label(self.autocomplete_window, text="🤖 ML Suggestions", 
                         font=("Arial", 9, "bold"), bg="lightblue")
        header.pack(fill=tk.X)
        
        # Список предложений
        listbox = tk.Listbox(self.autocomplete_window, height=min(8, len(suggestions)), 
                           font=("Consolas", 9))
        listbox.pack()
        
        for suggestion in suggestions:
            listbox.insert(tk.END, suggestion)
        
        if suggestions:
            listbox.selection_set(0)
        
        # Привязка событий
        listbox.bind('<Double-Button-1>', lambda e: self.insert_ml_suggestion(listbox.get(listbox.curselection())))
        listbox.bind('<Return>', lambda e: self.insert_ml_suggestion(listbox.get(listbox.curselection())))
        listbox.bind('<Escape>', lambda e: self.hide_ml_autocomplete())
        
        # Фокус на список
        listbox.focus_set()
    
    def insert_ml_suggestion(self, suggestion):
        """Вставка ML предложения"""
        if suggestion and hasattr(self, 'text_editor'):
            cursor_pos = self.text_editor.index(tk.INSERT)
            
            # Получение текущего слова
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
        
        self.hide_ml_autocomplete()
    
    def hide_ml_autocomplete(self):
        """Скрытие ML автодополнения"""
        if hasattr(self, 'autocomplete_window') and self.autocomplete_window:
            self.autocomplete_window.destroy()
            self.autocomplete_window = None
    
    def log_to_console(self, message):
        """Логирование в консоль"""
        if hasattr(self, 'console_output'):
            self.console_output.config(state='normal')
            timestamp = time.strftime("%H:%M:%S")
            self.console_output.insert(tk.END, f"[{timestamp}] {message}\n")
            self.console_output.see(tk.END)
            self.console_output.config(state='disabled')
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def update_line_numbers(self):
        """Обновление номеров строк"""
        if not hasattr(self, 'line_numbers') or not hasattr(self, 'text_editor'):
            return
        
        self.line_numbers.config(state='normal')
        self.line_numbers.delete("1.0", tk.END)
        
        content = self.text_editor.get("1.0", tk.END)
        lines = content.split('\n')
        
        for i in range(len(lines)):
            line_num = i + 1
            # Добавление иконок для точек останова и ML проблем
            icon = ""
            if hasattr(self, 'breakpoints') and line_num in self.breakpoints:
                icon = "🔴"
            elif hasattr(self, 'ml_analysis_results') and any(r.line_number == line_num for r in self.ml_analysis_results):
                icon = "⚠️"
            
            self.line_numbers.insert(tk.END, f"{icon}{line_num:4d}\n")
        
        self.line_numbers.config(state='disabled')
    
    def update_cursor_position(self):
        """Обновление позиции курсора"""
        if hasattr(self, 'cursor_label') and hasattr(self, 'text_editor'):
            cursor_pos = self.text_editor.index(tk.INSERT)
            line, col = cursor_pos.split('.')
            self.cursor_label.config(text=f"Line: {line}, Col: {int(col)+1}")
    
    def sync_scroll(self, *args):
        """Синхронизация скроллинга"""
        if hasattr(self, 'text_editor') and hasattr(self, 'line_numbers'):
            self.text_editor.yview(*args)
            self.line_numbers.yview(*args)
    
    # Основные методы IDE
    
    def run_full_ml_analysis(self):
        """Запуск полного ML анализа"""
        self.log_to_console("🤖 Starting full ML analysis...")
        
        if not hasattr(self, 'text_editor'):
            self.log_to_console("❌ Text editor not initialized")
            return
            
        code = self.text_editor.get("1.0", tk.END)
        
        # Запуск в отдельном потоке
        def analyze():
            try:
                results = self.ml_engine.analyze_code_realtime(code)
                self.root.after(0, lambda: self.display_full_analysis_results(results))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_console(f"Analysis error: {e}"))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def display_full_analysis_results(self, results):
        """Отображение результатов полного анализа"""
        self.ml_analysis_results = results
        self.update_ml_highlights()
        self.update_ml_analysis_tree()
        self.update_analysis_statistics()
        
        self.log_to_console(f"🎯 Analysis complete: {len(results)} issues found")
        
        # Переключение на вкладку анализа
        if hasattr(self, 'right_notebook'):
            self.right_notebook.select(0)
    
    def run_with_ml_analysis(self):
        """Запуск кода с ML анализом"""
        self.log_to_console("🤖 Running code with ML analysis...")
        
        # Сначала анализ
        self.run_full_ml_analysis()
        
        # Затем выполнение
        self.root.after(1000, self.run_code)
    
    def show_neural_visualization(self):
        """Показ нейронной визуализации"""
        self.neural_viz_active = True
        if hasattr(self, 'right_notebook'):
            self.right_notebook.select(1)  # Переключение на вкладку нейронной визуализации
        
        # Обновление визуализации
        if hasattr(self, 'neural_canvas'):
            self.ml_engine.create_neural_network_visualization(self.neural_canvas)
        
        self.log_to_console("🧠 Neural visualization activated")
    
    def show_training_monitor(self):
        """Показ монитора обучения"""
        self.training_active = True
        if hasattr(self, 'right_notebook'):
            self.right_notebook.select(2)  # Переключение на вкладку обучения
        
        # Запуск визуализации обучения
        if hasattr(self, 'training_canvas'):
            self.ml_engine.start_training_visualization(self.training_canvas)
        
        self.log_to_console("📈 Training monitor activated")
    
    def start_ml_training(self):
        """Запуск ML обучения"""
        if hasattr(self, 'training_active') and self.training_active:
            self.log_to_console("⚠️ Training already in progress")
            return
        
        self.training_active = True
        if hasattr(self, 'training_status_label'):
            self.training_status_label.config(text="Training Status: Running", foreground="green")
        
        # Получение параметров
        lr = float(self.lr_var.get()) if hasattr(self, 'lr_var') else 0.001
        batch_size = int(self.batch_var.get()) if hasattr(self, 'batch_var') else 32
        epochs = int(self.epochs_var.get()) if hasattr(self, 'epochs_var') else 100
        
        self.log_to_console(f"🚀 Starting training: LR={lr}, Batch={batch_size}, Epochs={epochs}")
        
        # Запуск симуляции обучения
        if hasattr(self, 'training_canvas'):
            self.ml_engine.start_training_visualization(self.training_canvas)
    
    def stop_ml_training(self):
        """Остановка ML обучения"""
        self.training_active = False
        if hasattr(self, 'training_status_label'):
            self.training_status_label.config(text="Training Status: Stopped", foreground="red")
        self.log_to_console("⏹️ Training stopped")
    
    def toggle_realtime_analysis(self):
        """Переключение анализа в реальном времени"""
        if hasattr(self, 'realtime_var'):
            self.ml_engine.auto_analysis_enabled = self.realtime_var.get()
        else:
            self.ml_engine.auto_analysis_enabled = not self.ml_engine.auto_analysis_enabled
        status = "enabled" if self.ml_engine.auto_analysis_enabled else "disabled"
        self.log_to_console(f"🔄 Real-time analysis {status}")
    
    # Заглушки для основных методов
    def new_file(self): 
        self.log_to_console("📄 New file created")
    def open_file(self): 
        self.log_to_console("📁 Opening file...")
    def save_file(self): 
        self.log_to_console("💾 File saved")
    def save_file_as(self): 
        self.log_to_console("💾 Save file as...")
    def undo(self): pass
    def redo(self): pass
    def cut(self): pass
    def copy(self): pass
    def paste(self): pass
    def run_code(self): 
        self.log_to_console("🚀 Running AnamorphX code...")
    def debug_code(self): 
        self.log_to_console("🐛 Starting debug session...")
    def debug_with_ml(self): 
        self.log_to_console("🧠 Starting ML-enhanced debugging...")
    def stop_execution(self): 
        self.log_to_console("⏹️ Execution stopped")
    def clear_console(self):
        if hasattr(self, 'console_output'):
            self.console_output.config(state='normal')
            self.console_output.delete("1.0", tk.END)
            self.console_output.config(state='disabled')
    def on_file_double_click(self, event): 
        self.log_to_console("📄 File double-clicked")
    def on_line_number_click(self, event): pass
    def on_line_number_right_click(self, event): pass
    def on_ml_editor_click(self, event): 
        self.update_cursor_position()
        self.hide_ml_autocomplete()
    def on_language_change(self, event): pass
    def execute_ml_console_command(self, event=None): 
        if hasattr(self, 'console_input'):
            command = self.console_input.get()
            if command:
                self.log_to_console(f"🤖 Executing: {command}")
                self.console_input.delete(0, tk.END)
    
    # Заглушки для ML методов
    def toggle_auto_analysis(self): 
        self.toggle_realtime_analysis()
    def apply_ml_fixes(self): 
        self.log_to_console("✨ Applying ML fixes...")
    def show_network_architecture(self): 
        self.log_to_console("🎯 Showing network architecture...")
    def refresh_neural_viz(self): 
        self.log_to_console("🔄 Refreshing neural visualization...")
    def initialize_neural_visualization(self): 
        if hasattr(self, 'neural_canvas'):
            self.ml_engine.create_neural_network_visualization(self.neural_canvas)
    def show_about(self): 
        self.log_to_console("ℹ️ About AnamorphX ML IDE")
    def export_ml_analysis(self): 
        self.log_to_console("📊 Exporting ML analysis report...")
    def toggle_ml_autocomplete(self): 
        self.log_to_console("🤖 ML autocomplete toggled")
    def apply_ml_optimizations(self): 
        self.log_to_console("⚡ Applying ML optimizations...")
    def show_ml_suggestions(self): 
        self.log_to_console("💡 Showing ML suggestions...")
    def show_variables(self): 
        self.log_to_console("🔢 Showing variables...")
    def show_ml_variables(self): 
        self.log_to_console("🤖 Showing ML variables...")
    def show_profiler(self): 
        self.log_to_console("⚡ Showing profiler...")
    def show_neural_profiler(self): 
        self.log_to_console("🧠 Showing neural profiler...")
    def show_ml_settings(self): 
        self.log_to_console("🎛️ Showing ML settings...")
    def show_ml_performance(self): 
        self.log_to_console("📊 Showing ML performance...")
    def train_custom_model(self): 
        self.log_to_console("🔧 Training custom model...") 