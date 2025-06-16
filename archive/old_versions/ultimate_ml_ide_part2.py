def create_variables_panel(self):
    """Создание панели переменных"""
    variables_frame = ttk.Frame(self.right_notebook)
    self.right_notebook.add(variables_frame, text=_("panel_variables"))
    
    # Дерево переменных
    self.variables_tree = ttk.Treeview(variables_frame, columns=("value", "type"), show="tree headings")
    self.variables_tree.heading("#0", text=_("var_name"))
    self.variables_tree.heading("value", text=_("var_value"))
    self.variables_tree.heading("type", text=_("var_type"))
    
    self.variables_tree.column("#0", width=100)
    self.variables_tree.column("value", width=100)
    self.variables_tree.column("type", width=80)
    
    self.variables_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Кнопки управления переменными
    var_buttons = ttk.Frame(variables_frame)
    var_buttons.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Button(var_buttons, text=_("btn_refresh"), command=self.refresh_variables).pack(side=tk.LEFT, padx=2)
    ttk.Button(var_buttons, text="Watch", command=self.add_watch_expression).pack(side=tk.LEFT, padx=2)
    ttk.Button(var_buttons, text="Clear", command=self.clear_variables).pack(side=tk.LEFT, padx=2)
    
    self.ui_elements['variables_tree'] = self.variables_tree
    
def create_call_stack_panel(self):
    """Создание панели стека вызовов"""
    stack_frame = ttk.Frame(self.right_notebook)
    self.right_notebook.add(stack_frame, text=_("panel_call_stack"))
    
    # Список стека вызовов
    self.call_stack_listbox = tk.Listbox(stack_frame, font=("Consolas", 9))
    self.call_stack_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Кнопки управления стеком
    stack_buttons = ttk.Frame(stack_frame)
    stack_buttons.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Button(stack_buttons, text=_("btn_refresh"), command=self.refresh_call_stack).pack(side=tk.LEFT, padx=2)
    ttk.Button(stack_buttons, text="Go to", command=self.goto_stack_frame).pack(side=tk.LEFT, padx=2)
    
    self.ui_elements['call_stack'] = self.call_stack_listbox
    
def create_profiler_panel(self):
    """Создание панели профайлера"""
    profiler_frame = ttk.Frame(self.right_notebook)
    self.right_notebook.add(profiler_frame, text=_("panel_profiler"))
    
    # Дерево профайлера
    self.profiler_tree = ttk.Treeview(profiler_frame, columns=("time", "calls", "avg"), show="tree headings")
    self.profiler_tree.heading("#0", text=_("prof_function"))
    self.profiler_tree.heading("time", text=_("prof_time"))
    self.profiler_tree.heading("calls", text=_("prof_calls"))
    self.profiler_tree.heading("avg", text=_("prof_avg"))
    
    self.profiler_tree.column("#0", width=120)
    self.profiler_tree.column("time", width=80)
    self.profiler_tree.column("calls", width=60)
    self.profiler_tree.column("avg", width=80)
    
    self.profiler_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Кнопки профайлера
    prof_buttons = ttk.Frame(profiler_frame)
    prof_buttons.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Button(prof_buttons, text=_("btn_start_profiling"), command=self.start_profiling).pack(side=tk.LEFT, padx=2)
    ttk.Button(prof_buttons, text=_("btn_stop_profiling"), command=self.stop_profiling).pack(side=tk.LEFT, padx=2)
    ttk.Button(prof_buttons, text=_("btn_clear_profile"), command=self.clear_profiler).pack(side=tk.LEFT, padx=2)
    
    self.ui_elements['profiler_tree'] = self.profiler_tree
    
def create_debug_console(self):
    """Создание консоли отладки"""
    console_frame = ttk.Frame(self.right_notebook)
    self.right_notebook.add(console_frame, text=_("panel_console"))
    
    # Область вывода консоли
    self.console_output = Text(console_frame, height=15, state='disabled', 
                              font=("Consolas", 9), bg="black", fg="white")
    self.console_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))
    
    # Поле ввода команд
    input_frame = ttk.Frame(console_frame)
    input_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
    
    ttk.Label(input_frame, text=">>>").pack(side=tk.LEFT)
    
    self.console_input = ttk.Entry(input_frame, font=("Consolas", 9))
    self.console_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
    self.console_input.bind('<Return>', self.execute_console_command)
    self.console_input.bind('<Up>', self.console_history_up)
    self.console_input.bind('<Down>', self.console_history_down)
    
    ttk.Button(input_frame, text="Execute", command=self.execute_console_command).pack(side=tk.LEFT, padx=2)
    ttk.Button(input_frame, text="Clear", command=self.clear_console).pack(side=tk.LEFT, padx=2)
    
    self.ui_elements['console'] = self.console_output
    
    # Приветственное сообщение
    self.log_to_console("🚀 AnamorphX IDE Ultimate ML Edition")
    self.log_to_console("💡 Type 'help' for available commands")
    if HAS_REAL_ML:
        self.log_to_console("🤖 Real ML integration: ACTIVE")
    else:
        self.log_to_console("⚠️ Real ML integration: LIMITED MODE")
    
def create_output_panel(self):
    """Создание панели вывода"""
    output_frame = ttk.Frame(self.right_notebook)
    self.right_notebook.add(output_frame, text="Output")
    
    # Область вывода программы
    self.program_output = Text(output_frame, state='disabled', 
                              font=("Consolas", 10), bg="#f8f8f8")
    self.program_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Кнопки управления выводом
    output_buttons = ttk.Frame(output_frame)
    output_buttons.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Button(output_buttons, text="Clear Output", command=self.clear_output).pack(side=tk.LEFT, padx=2)
    ttk.Button(output_buttons, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=2)
    ttk.Button(output_buttons, text="Copy Output", command=self.copy_output).pack(side=tk.LEFT, padx=2)
    
def create_status_bar(self):
    """Создание строки состояния"""
    self.status_bar = ttk.Frame(self.root)
    self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
    
    # Статус выполнения
    self.status_label = ttk.Label(self.status_bar, text=_("status_ready"))
    self.status_label.pack(side=tk.LEFT)
    
    # Позиция курсора
    self.cursor_label = ttk.Label(self.status_bar, text="Line: 1, Col: 1")
    self.cursor_label.pack(side=tk.RIGHT, padx=10)
    
    # Язык файла
    self.language_label = ttk.Label(self.status_bar, text="AnamorphX")
    self.language_label.pack(side=tk.RIGHT, padx=10)
    
    # Кодировка
    self.encoding_label = ttk.Label(self.status_bar, text="UTF-8")
    self.encoding_label.pack(side=tk.RIGHT, padx=10)
    
    # ML статус
    ml_status = "🤖 ML: Active" if HAS_REAL_ML else "🤖 ML: Limited"
    self.ml_status_bar = ttk.Label(self.status_bar, text=ml_status)
    self.ml_status_bar.pack(side=tk.RIGHT, padx=10)
    
    self.ui_elements['status_labels'] = [
        self.status_label, self.cursor_label, self.language_label, 
        self.encoding_label, self.ml_status_bar
    ]
    
# Методы обработки событий
    
def on_text_change(self, event=None):
    """Обработка изменения текста"""
    self.file_modified = True
    self.modified_label.config(text="●")
    self.update_line_numbers()
    self.update_cursor_position()
    self.highlight_syntax()
    
    # ML анализ в реальном времени (если включен)
    if HAS_REAL_ML and hasattr(self, 'ml_panel'):
        self.schedule_ml_analysis()
    
def schedule_ml_analysis(self):
    """Планирование ML анализа"""
    # Отменяем предыдущий анализ
    if hasattr(self, 'ml_analysis_timer'):
        self.root.after_cancel(self.ml_analysis_timer)
    
    # Планируем новый анализ через 2 секунды
    self.ml_analysis_timer = self.root.after(2000, self.run_ml_analysis_background)
    
def run_ml_analysis_background(self):
    """Фоновый ML анализ"""
    try:
        if hasattr(self, 'ml_panel') and self.ml_panel:
            # Запуск анализа в фоне
            threading.Thread(target=self.ml_panel.analyze_current_code, daemon=True).start()
    except Exception as e:
        self.log_to_console(f"ML analysis error: {e}")
    
def on_editor_click(self, event=None):
    """Обработка клика в редакторе"""
    self.update_cursor_position()
    self.hide_autocomplete()
    
def on_line_number_click(self, event):
    """Обработка клика по номеру строки"""
    # Получение номера строки
    index = self.line_numbers.index(f"@{event.x},{event.y}")
    line_num = int(index.split('.')[0])
    
    # Переключение точки останова
    self.toggle_breakpoint_at_line(line_num)
    
def on_file_double_click(self, event):
    """Обработка двойного клика по файлу"""
    selection = self.file_tree.selection()
    if selection:
        item = self.file_tree.item(selection[0])
        if item['values'] and item['values'][0] == 'file':
            filename = item['text']
            self.load_file_content(filename)
    
def on_tab(self, event):
    """Обработка нажатия Tab"""
    # Вставка 4 пробелов вместо табуляции
    self.text_editor.insert(tk.INSERT, "    ")
    return "break"
    
def on_return(self, event):
    """Обработка нажатия Enter"""
    # Автоматический отступ
    current_line = self.text_editor.get("insert linestart", "insert")
    indent = ""
    for char in current_line:
        if char in [' ', '\t']:
            indent += char
        else:
            break
    
    # Дополнительный отступ для блоков
    if current_line.strip().endswith('{') or current_line.strip().endswith(':'):
        indent += "    "
    
    self.text_editor.insert(tk.INSERT, "\n" + indent)
    return "break"
    
def on_language_change(self, event=None):
    """Обработка смены языка"""
    new_language = self.language_var.get()
    self.change_language(new_language)
    
# Файловые операции
    
def new_file(self):
    """Создание нового файла"""
    if self.file_modified:
        if not self.ask_save_changes():
            return
    
    self.text_editor.delete("1.0", tk.END)
    self.current_file = None
    self.file_modified = False
    self.modified_label.config(text="")
    self.file_label.config(text="📄 Untitled.anamorph")
    self.update_line_numbers()
    self.log_to_console("📄 New file created")
    
def open_file(self):
    """Открытие файла"""
    if self.file_modified:
        if not self.ask_save_changes():
            return
    
    filename = filedialog.askopenfilename(
        title="Open File",
        filetypes=[
            ("AnamorphX files", "*.anamorph"),
            ("Python files", "*.py"),
            ("All files", "*.*")
        ]
    )
    
    if filename:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
            
            self.text_editor.delete("1.0", tk.END)
            self.text_editor.insert("1.0", content)
            
            self.current_file = filename
            self.file_modified = False
            self.modified_label.config(text="")
            self.file_label.config(text=f"📄 {os.path.basename(filename)}")
            self.update_line_numbers()
            self.highlight_syntax()
            
            self.log_to_console(f"📁 Opened: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
    
def save_file(self, silent=False):
    """Сохранение файла"""
    if not self.current_file:
        return self.save_file_as()
    
    try:
        content = self.text_editor.get("1.0", tk.END + "-1c")
        with open(self.current_file, 'w', encoding='utf-8') as file:
            file.write(content)
        
        self.file_modified = False
        self.modified_label.config(text="")
        
        if not silent:
            self.log_to_console(f"💾 Saved: {self.current_file}")
        
        return True
        
    except Exception as e:
        if not silent:
            messagebox.showerror("Error", f"Failed to save file: {e}")
        return False
    
def save_file_as(self):
    """Сохранение файла как"""
    filename = filedialog.asksaveasfilename(
        title="Save File As",
        defaultextension=".anamorph",
        filetypes=[
            ("AnamorphX files", "*.anamorph"),
            ("Python files", "*.py"),
            ("All files", "*.*")
        ]
    )
    
    if filename:
        self.current_file = filename
        self.file_label.config(text=f"📄 {os.path.basename(filename)}")
        return self.save_file()
    
    return False
    
def ask_save_changes(self):
    """Запрос сохранения изменений"""
    result = messagebox.askyesnocancel(
        "Save Changes",
        "Do you want to save changes to the current file?"
    )
    
    if result is True:
        return self.save_file()
    elif result is False:
        return True
    else:
        return False
    
# Операции редактирования
    
def undo(self):
    """Отмена"""
    try:
        self.text_editor.edit_undo()
        self.update_line_numbers()
    except tk.TclError:
        pass
    
def redo(self):
    """Повтор"""
    try:
        self.text_editor.edit_redo()
        self.update_line_numbers()
    except tk.TclError:
        pass
    
def cut(self):
    """Вырезать"""
    try:
        self.text_editor.event_generate("<<Cut>>")
    except tk.TclError:
        pass
    
def copy(self):
    """Копировать"""
    try:
        self.text_editor.event_generate("<<Copy>>")
    except tk.TclError:
        pass
    
def paste(self):
    """Вставить"""
    try:
        self.text_editor.event_generate("<<Paste>>")
        self.update_line_numbers()
    except tk.TclError:
        pass
    
def select_all(self):
    """Выделить все"""
    self.text_editor.tag_add(tk.SEL, "1.0", tk.END)
    self.text_editor.mark_set(tk.INSERT, "1.0")
    self.text_editor.see(tk.INSERT)
    return "break"
    
def find(self):
    """Поиск"""
    search_window = tk.Toplevel(self.root)
    search_window.title("Find")
    search_window.geometry("400x100")
    search_window.transient(self.root)
    
    ttk.Label(search_window, text="Find:").pack(pady=5)
    
    search_entry = ttk.Entry(search_window, width=40)
    search_entry.pack(pady=5)
    search_entry.focus()
    
    def do_search():
        query = search_entry.get()
        if query:
            self.search_text(query)
            search_window.destroy()
    
    ttk.Button(search_window, text="Find", command=do_search).pack(pady=5)
    search_entry.bind('<Return>', lambda e: do_search())
    
def replace(self):
    """Замена"""
    replace_window = tk.Toplevel(self.root)
    replace_window.title("Replace")
    replace_window.geometry("400x150")
    replace_window.transient(self.root)
    
    ttk.Label(replace_window, text="Find:").pack(pady=2)
    find_entry = ttk.Entry(replace_window, width=40)
    find_entry.pack(pady=2)
    
    ttk.Label(replace_window, text="Replace with:").pack(pady=2)
    replace_entry = ttk.Entry(replace_window, width=40)
    replace_entry.pack(pady=2)
    
    buttons_frame = ttk.Frame(replace_window)
    buttons_frame.pack(pady=10)
    
    def do_replace():
        find_text = find_entry.get()
        replace_text = replace_entry.get()
        if find_text:
            self.replace_text(find_text, replace_text)
    
    def do_replace_all():
        find_text = find_entry.get()
        replace_text = replace_entry.get()
        if find_text:
            self.replace_all_text(find_text, replace_text)
            replace_window.destroy()
    
    ttk.Button(buttons_frame, text="Replace", command=do_replace).pack(side=tk.LEFT, padx=5)
    ttk.Button(buttons_frame, text="Replace All", command=do_replace_all).pack(side=tk.LEFT, padx=5)
    
    find_entry.focus()
    
def toggle_comment(self):
    """Переключение комментария"""
    try:
        # Получение выделенного текста или текущей строки
        if self.text_editor.tag_ranges(tk.SEL):
            start = self.text_editor.index(tk.SEL_FIRST)
            end = self.text_editor.index(tk.SEL_LAST)
        else:
            start = self.text_editor.index("insert linestart")
            end = self.text_editor.index("insert lineend")
        
        # Получение текста
        text = self.text_editor.get(start, end)
        lines = text.split('\n')
        
        # Проверка, закомментированы ли строки
        all_commented = all(line.strip().startswith('//') or line.strip() == '' for line in lines)
        
        new_lines = []
        for line in lines:
            if all_commented:
                # Раскомментирование
                if line.strip().startswith('//'):
                    new_lines.append(line.replace('//', '', 1))
                else:
                    new_lines.append(line)
            else:
                # Комментирование
                if line.strip():
                    new_lines.append('//' + line)
                else:
                    new_lines.append(line)
        
        # Замена текста
        self.text_editor.delete(start, end)
        self.text_editor.insert(start, '\n'.join(new_lines))
        
    except Exception as e:
        self.log_to_console(f"Comment toggle error: {e}")
    
    return "break"
    
# Продолжение следует...