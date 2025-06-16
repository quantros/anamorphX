    def create_main_interface(self):
        """Создание основного интерфейса с интегрированным ML"""
        # Главный PanedWindow
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель (файловый проводник)
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Центральная панель (редактор с ML)
        self.center_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.center_frame, weight=4)
        
        # Правая панель (инструменты + ML)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=2)
        
        self.create_file_explorer()
        self.create_ml_enhanced_editor()
        self.create_integrated_tools_panel()
    
    def create_file_explorer(self):
        """Создание файлового проводника"""
        explorer_label = ttk.Label(self.left_frame, text="📁 Project Explorer", font=("Arial", 10, "bold"))
        explorer_label.pack(anchor="w", padx=5, pady=2)
        
        # Дерево файлов
        self.file_tree = ttk.Treeview(self.left_frame)
        self.file_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Заполнение примерами файлов
        self.populate_file_tree()
        
        # Привязка событий
        self.file_tree.bind("<Double-1>", self.on_file_double_click)
    
    def populate_file_tree(self):
        """Заполнение дерева файлов"""
        # Корневая папка проекта
        project_root = self.file_tree.insert("", "end", text="📁 AnamorphX ML Project", open=True, values=("folder",))
        
        # Основные файлы
        self.file_tree.insert(project_root, "end", text="📄 main.anamorph", values=("file",))
        self.file_tree.insert(project_root, "end", text="📄 neural_classifier.anamorph", values=("file",))
        self.file_tree.insert(project_root, "end", text="📄 deep_network.anamorph", values=("file",))
    
    def create_ml_enhanced_editor(self):
        """Создание редактора с ML улучшениями"""
        editor_frame = ttk.Frame(self.center_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок редактора с ML статусом
        editor_header = ttk.Frame(editor_frame)
        editor_header.pack(fill=tk.X, pady=(0, 2))
        
        self.file_label = ttk.Label(editor_header, text="📄 Untitled.anamorph", font=("Arial", 10, "bold"))
        self.file_label.pack(side=tk.LEFT)
        
        # Индикатор изменений
        self.modified_label = ttk.Label(editor_header, text="", foreground="red")
        self.modified_label.pack(side=tk.LEFT, padx=5)
        
        # ML статус для файла
        self.ml_file_status = ttk.Label(editor_header, text="🤖 ML: Ready", font=("Arial", 9), foreground="green")
        self.ml_file_status.pack(side=tk.RIGHT, padx=5)
        
        # Фрейм для номеров строк и текста
        text_frame = tk.Frame(editor_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Номера строк с ML индикаторами
        self.line_numbers = Text(text_frame, width=6, padx=3, takefocus=0,
                                border=0, state="disabled", wrap="none",
                                font=("Consolas", 11), bg="#f0f0f0", fg="#666666")
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Основной текстовый виджет с ML
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
        
        # Настройка тегов с ML
        self.setup_ml_text_tags()
        
        # Привязка событий с ML
        self.setup_ml_editor_events()
        
        # Обновление номеров строк
        self.update_line_numbers()
    
    def setup_ml_text_tags(self):
        """Настройка тегов для подсветки с ML"""
        # Базовая подсветка синтаксиса AnamorphX
        self.text_editor.tag_configure("keyword", foreground="#0000FF", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("string", foreground="#008000")
        self.text_editor.tag_configure("comment", foreground="#808080", font=("Consolas", 11, "italic"))
        self.text_editor.tag_configure("number", foreground="#FF0000")
        self.text_editor.tag_configure("function", foreground="#800080", font=("Consolas", 11, "bold"))
        
        # AnamorphX специфичные теги
        self.text_editor.tag_configure("neuron", foreground="#FF8000", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("network", foreground="#000080", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("activation", foreground="#008080", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("layer", foreground="#4B0082", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("optimizer", foreground="#DC143C", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("loss", foreground="#B22222", font=("Consolas", 11, "bold"))
        
        # ML анализ теги
        self.text_editor.tag_configure("ml_error", background="#FFCCCB", underline=True)
        self.text_editor.tag_configure("ml_warning", background="#FFE4B5", underline=True)
        self.text_editor.tag_configure("ml_optimization", background="#E0FFE0", underline=True)
        self.text_editor.tag_configure("ml_suggestion", background="#E6E6FA", underline=True)
        self.text_editor.tag_configure("ml_neural", background="#F0F8FF", underline=True)
        
        # Отладка
        self.text_editor.tag_configure("current_line", background="#E6F3FF")
        self.text_editor.tag_configure("breakpoint", background="#FF6B6B", foreground="white")
        
        # Номера строк с ML индикаторами
        self.line_numbers.tag_configure("breakpoint", background="#FF6B6B", foreground="white")
        self.line_numbers.tag_configure("current", background="#E6F3FF")
        self.line_numbers.tag_configure("ml_issue", background="#FFE4B5")
        self.line_numbers.tag_configure("ml_suggestion", background="#E6E6FA")
    
    def setup_ml_editor_events(self):
        """Настройка событий редактора с ML"""
        # Основные события
        self.text_editor.bind("<KeyRelease>", self.on_ml_text_change)
        self.text_editor.bind("<Button-1>", self.on_ml_editor_click)
        self.text_editor.bind("<ButtonRelease-1>", self.on_ml_editor_click)
        
        # ML специфичные события
        self.text_editor.bind("<Control-space>", self.trigger_ml_autocomplete)
        self.text_editor.bind("<Control-m>", lambda e: self.run_full_ml_analysis())
        
        # События номеров строк
        self.line_numbers.bind("<Button-1>", self.on_line_number_click)
        self.line_numbers.bind("<Button-3>", self.on_line_number_right_click)
    
    def create_integrated_tools_panel(self):
        """Создание интегрированной панели инструментов с ML"""
        self.right_notebook = ttk.Notebook(self.right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # ML анализ (основная вкладка)
        self.create_ml_analysis_panel()
        
        # Нейронная визуализация
        self.create_neural_visualization_panel()
        
        # Мониторинг обучения
        self.create_training_monitoring_panel()
        
        # Консоль с ML
        self.create_ml_console_panel()
    
    def create_ml_analysis_panel(self):
        """Создание панели ML анализа"""
        analysis_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(analysis_frame, text="🤖 ML Analysis")
        
        # Заголовок с настройками
        header_frame = ttk.Frame(analysis_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(header_frame, text="Real-time Code Analysis", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        # Переключатель автоанализа
        self.auto_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(header_frame, text="Auto", variable=self.auto_analysis_var, 
                       command=self.toggle_auto_analysis).pack(side=tk.RIGHT)
        
        # Кнопки управления
        control_frame = ttk.Frame(analysis_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(control_frame, text="🔍 Analyze Now", command=self.run_full_ml_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="✨ Apply Fixes", command=self.apply_ml_fixes).pack(side=tk.LEFT, padx=2)
        
        # Дерево результатов анализа
        self.ml_analysis_tree = ttk.Treeview(analysis_frame, 
                                           columns=("type", "severity", "confidence", "suggestion"), 
                                           show="tree headings")
        self.ml_analysis_tree.heading("#0", text="Line")
        self.ml_analysis_tree.heading("type", text="Type")
        self.ml_analysis_tree.heading("severity", text="Severity")
        self.ml_analysis_tree.heading("confidence", text="Confidence")
        self.ml_analysis_tree.heading("suggestion", text="Suggestion")
        
        # Настройка колонок
        self.ml_analysis_tree.column("#0", width=50)
        self.ml_analysis_tree.column("type", width=80)
        self.ml_analysis_tree.column("severity", width=60)
        self.ml_analysis_tree.column("confidence", width=70)
        self.ml_analysis_tree.column("suggestion", width=200)
        
        self.ml_analysis_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Статистика анализа
        stats_frame = ttk.LabelFrame(analysis_frame, text="Analysis Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.analysis_stats_label = ttk.Label(stats_frame, text="No analysis performed yet", font=("Arial", 9))
        self.analysis_stats_label.pack(padx=5, pady=2)
    
    def create_neural_visualization_panel(self):
        """Создание панели нейронной визуализации"""
        neural_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(neural_frame, text="🧠 Neural Viz")
        
        # Кнопки управления визуализацией
        viz_control_frame = ttk.Frame(neural_frame)
        viz_control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(viz_control_frame, text="🎯 Show Architecture", command=self.show_network_architecture).pack(side=tk.LEFT, padx=2)
        ttk.Button(viz_control_frame, text="🔄 Refresh", command=self.refresh_neural_viz).pack(side=tk.LEFT, padx=2)
        
        # Canvas для нейронной визуализации
        self.neural_canvas = Canvas(neural_frame, bg="white", height=300)
        self.neural_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Инициализация визуализации
        self.neural_viz_active = True
        self.root.after(1000, self.initialize_neural_visualization)
    
    def create_training_monitoring_panel(self):
        """Создание панели мониторинга обучения"""
        training_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(training_frame, text="📈 Training")
        
        # Кнопки управления обучением
        training_control_frame = ttk.Frame(training_frame)
        training_control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(training_control_frame, text="▶️ Start Training", command=self.start_ml_training).pack(side=tk.LEFT, padx=2)
        ttk.Button(training_control_frame, text="⏹️ Stop", command=self.stop_ml_training).pack(side=tk.LEFT, padx=2)
        
        # Canvas для графиков обучения
        self.training_canvas = Canvas(training_frame, bg="white", height=250)
        self.training_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Параметры обучения
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Learning Rate
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=10).pack(side=tk.RIGHT)
        
        # Batch Size
        batch_frame = ttk.Frame(params_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_var = tk.StringVar(value="32")
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=10).pack(side=tk.RIGHT)
        
        # Epochs
        epochs_frame = ttk.Frame(params_frame)
        epochs_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(epochs_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=10).pack(side=tk.RIGHT)
        
        # Статус обучения
        self.training_status_label = ttk.Label(training_frame, text="Training Status: Ready", font=("Arial", 9))
        self.training_status_label.pack(pady=2)
        
        self.training_active = False
    
    def create_ml_console_panel(self):
        """Создание консоли с ML командами"""
        console_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(console_frame, text="💻 ML Console")
        
        # Область вывода консоли
        self.console_output = Text(console_frame, height=15, state="disabled", 
                                  font=("Consolas", 9), bg="black", fg="white")
        self.console_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))
        
        # Поле ввода команд
        input_frame = ttk.Frame(console_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(input_frame, text="ML>>>").pack(side=tk.LEFT)
        
        self.console_input = ttk.Entry(input_frame, font=("Consolas", 9))
        self.console_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.console_input.bind("<Return>", self.execute_ml_console_command)
        
        ttk.Button(input_frame, text="Execute", command=self.execute_ml_console_command).pack(side=tk.LEFT, padx=2)
        ttk.Button(input_frame, text="Clear", command=self.clear_console).pack(side=tk.LEFT, padx=2)
        
        # Приветственное сообщение
        self.log_to_console("🤖 AnamorphX ML IDE - Unified Edition")
        self.log_to_console("💡 ML integration is fully active")
        self.log_to_console("🔍 Real-time analysis enabled")
        self.log_to_console("Type \"help\" for ML commands")
    
    def create_status_bar(self):
        """Создание строки состояния с ML информацией"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # Основной статус
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # ML статус
        self.ml_status_detail = ttk.Label(self.status_bar, text="🤖 ML: Analyzing...", foreground="blue")
        self.ml_status_detail.pack(side=tk.LEFT, padx=10)
        
        # Позиция курсора
        self.cursor_label = ttk.Label(self.status_bar, text="Line: 1, Col: 1")
        self.cursor_label.pack(side=tk.RIGHT, padx=10)
        
        # ML производительность
        self.ml_perf_label = ttk.Label(self.status_bar, text="⚡ ML: 0ms", foreground="green")
        self.ml_perf_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_hotkeys(self):
        """Настройка горячих клавиш с ML"""
        # Стандартные горячие клавиши
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<F5>", lambda e: self.run_code())
        
        # ML специфичные горячие клавиши
        self.root.bind("<Control-m>", lambda e: self.run_full_ml_analysis())
        self.root.bind("<Shift-F5>", lambda e: self.run_with_ml_analysis())
        self.root.bind("<Control-space>", lambda e: self.trigger_ml_autocomplete(e)) 