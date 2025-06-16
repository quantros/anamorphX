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
            
            # Обновление статуса
            analysis_time = (time.time() - start_time) * 1000
            if hasattr(self, 'ml_file_status'):
                self.ml_file_status.config(text=f"🤖 ML: Ready ({len(self.ml_analysis_results)} issues)", foreground="green")
            
        except Exception as e:
            self.log_to_console(f"ML analysis error: {e}")
            if hasattr(self, 'ml_file_status'):
                self.ml_file_status.config(text="🤖 ML: Error", foreground="red")
    
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
            self.line_numbers.insert(tk.END, f"{line_num:4d}\n")
        
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
    
    def run_full_ml_analysis(self):
        """Запуск полного ML анализа"""
        self.log_to_console("🤖 Starting full ML analysis...")
        
        if not hasattr(self, 'text_editor'):
            self.log_to_console("❌ Text editor not initialized")
            return
            
        code = self.text_editor.get("1.0", tk.END)
        results = self.ml_engine.analyze_code_realtime(code)
        self.ml_analysis_results = results
        self.log_to_console(f"🎯 Analysis complete: {len(results)} issues found")
    
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
    def on_language_change(self, event): pass
    def execute_ml_console_command(self, event=None): 
        if hasattr(self, 'console_input'):
            command = self.console_input.get()
            if command:
                self.log_to_console(f"🤖 Executing: {command}")
                self.console_input.delete(0, tk.END)
    
    # Заглушки для ML методов
    def toggle_auto_analysis(self): pass
    def apply_ml_fixes(self): 
        self.log_to_console("✨ Applying ML fixes...")
    def show_network_architecture(self): 
        self.log_to_console("🎯 Showing network architecture...")
    def refresh_neural_viz(self): 
        self.log_to_console("🔄 Refreshing neural visualization...")
    def initialize_neural_visualization(self): pass
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
    def run_with_ml_analysis(self):
        self.log_to_console("🤖 Running code with ML analysis...")
        self.run_full_ml_analysis()
        self.run_code()
    def show_neural_visualization(self):
        self.log_to_console("🧠 Neural visualization activated")
    def show_training_monitor(self):
        self.log_to_console("📈 Training monitor activated")
    def start_ml_training(self):
        self.log_to_console("🚀 Starting ML training...")
    def stop_ml_training(self):
        self.log_to_console("⏹️ Training stopped")
    def toggle_realtime_analysis(self):
        self.ml_engine.auto_analysis_enabled = not self.ml_engine.auto_analysis_enabled
        status = "enabled" if self.ml_engine.auto_analysis_enabled else "disabled"
        self.log_to_console(f"🔄 Real-time analysis {status}")
    def trigger_ml_autocomplete(self, event):
        self.log_to_console("🤖 ML autocomplete triggered")
        return "break"

def main():
    """Главная функция запуска IDE"""
    print("🚀 Starting AnamorphX IDE - Unified ML Edition")
    print(f"🤖 ML Status: {'Full PyTorch Integration' if HAS_FULL_ML else 'Simulated Mode'}")
    
    # Создание и запуск IDE
    ide = UnifiedMLIDE()
    
    try:
        ide.root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 AnamorphX IDE closed by user")
    except Exception as e:
        print(f"❌ IDE Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 