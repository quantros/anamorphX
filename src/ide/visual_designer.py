"""
Visual Network Designer для AnamorphX IDE

Визуальный редактор для создания нейронных сетей с помощью drag & drop интерфейса.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class NodeType(Enum):
    """Типы узлов в визуальном дизайнере"""
    DENSE = "dense"
    CONV2D = "conv2d"
    MAXPOOL = "maxpool"
    AVGPOOL = "avgpool"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    ACTIVATION = "activation"
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class NodeConfig:
    """Конфигурация узла"""
    id: str
    type: NodeType
    name: str
    x: float
    y: float
    width: float = 120
    height: float = 60
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class Connection:
    """Соединение между узлами"""
    id: str
    source_id: str
    target_id: str
    source_port: str = "output"
    target_port: str = "input"


class NetworkCanvas(tk.Canvas):
    """Канвас для рисования нейронной сети"""
    
    def __init__(self, parent, designer):
        super().__init__(parent, bg='white', width=800, height=600)
        self.designer = designer
        self.nodes: Dict[str, NodeConfig] = {}
        self.connections: Dict[str, Connection] = {}
        self.selected_node = None
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.connection_mode = False
        self.connection_start = None
        
        # Привязываем события
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Double-Button-1>", self.on_double_click)
        self.bind("<Button-3>", self.on_right_click)
        
        # Цвета для разных типов узлов
        self.node_colors = {
            NodeType.INPUT: "#4CAF50",
            NodeType.OUTPUT: "#F44336",
            NodeType.DENSE: "#2196F3",
            NodeType.CONV2D: "#FF9800",
            NodeType.MAXPOOL: "#9C27B0",
            NodeType.AVGPOOL: "#9C27B0",
            NodeType.LSTM: "#607D8B",
            NodeType.GRU: "#607D8B",
            NodeType.TRANSFORMER: "#E91E63",
            NodeType.ATTENTION: "#E91E63",
            NodeType.EMBEDDING: "#795548",
            NodeType.DROPOUT: "#FFC107",
            NodeType.BATCH_NORM: "#00BCD4",
            NodeType.LAYER_NORM: "#00BCD4",
            NodeType.ACTIVATION: "#8BC34A"
        }
    
    def add_node(self, node_type: NodeType, x: float, y: float) -> str:
        """Добавить узел на канвас"""
        node_id = f"node_{len(self.nodes)}"
        node_name = f"{node_type.value}_{len(self.nodes)}"
        
        node = NodeConfig(
            id=node_id,
            type=node_type,
            name=node_name,
            x=x,
            y=y,
            parameters=self._get_default_parameters(node_type)
        )
        
        self.nodes[node_id] = node
        self.draw_node(node)
        self.designer.on_network_changed()
        
        return node_id
    
    def _get_default_parameters(self, node_type: NodeType) -> Dict[str, Any]:
        """Получить параметры по умолчанию для типа узла"""
        defaults = {
            NodeType.DENSE: {"units": 128, "activation": "relu"},
            NodeType.CONV2D: {"filters": 32, "kernel_size": 3, "activation": "relu"},
            NodeType.MAXPOOL: {"pool_size": 2, "stride": 2},
            NodeType.AVGPOOL: {"pool_size": 2, "stride": 2},
            NodeType.LSTM: {"units": 128, "return_sequences": True},
            NodeType.GRU: {"units": 128, "return_sequences": True},
            NodeType.TRANSFORMER: {"embed_dim": 512, "num_heads": 8, "ff_dim": 2048},
            NodeType.ATTENTION: {"embed_dim": 512, "num_heads": 8},
            NodeType.EMBEDDING: {"vocab_size": 10000, "embed_dim": 512},
            NodeType.DROPOUT: {"rate": 0.2},
            NodeType.BATCH_NORM: {"momentum": 0.99},
            NodeType.LAYER_NORM: {"epsilon": 1e-6},
            NodeType.ACTIVATION: {"function": "relu"},
            NodeType.INPUT: {"shape": [None, 784]},
            NodeType.OUTPUT: {"units": 10, "activation": "softmax"}
        }
        return defaults.get(node_type, {})
    
    def draw_node(self, node: NodeConfig):
        """Нарисовать узел на канвасе"""
        color = self.node_colors.get(node.type, "#CCCCCC")
        
        # Рисуем прямоугольник узла
        rect_id = self.create_rectangle(
            node.x, node.y, 
            node.x + node.width, node.y + node.height,
            fill=color, outline="black", width=2,
            tags=(node.id, "node")
        )
        
        # Добавляем текст
        text_id = self.create_text(
            node.x + node.width/2, node.y + node.height/2,
            text=node.name, fill="white", font=("Arial", 10, "bold"),
            tags=(node.id, "node_text")
        )
        
        # Добавляем порты подключения
        self._draw_ports(node)
    
    def _draw_ports(self, node: NodeConfig):
        """Нарисовать порты подключения для узла"""
        port_size = 8
        
        # Входной порт (слева)
        if node.type != NodeType.INPUT:
            self.create_oval(
                node.x - port_size/2, node.y + node.height/2 - port_size/2,
                node.x + port_size/2, node.y + node.height/2 + port_size/2,
                fill="blue", outline="darkblue",
                tags=(node.id, "input_port")
            )
        
        # Выходной порт (справа)
        if node.type != NodeType.OUTPUT:
            self.create_oval(
                node.x + node.width - port_size/2, node.y + node.height/2 - port_size/2,
                node.x + node.width + port_size/2, node.y + node.height/2 + port_size/2,
                fill="red", outline="darkred",
                tags=(node.id, "output_port")
            )
    
    def connect_nodes(self, source_id: str, target_id: str) -> str:
        """Соединить два узла"""
        connection_id = f"conn_{len(self.connections)}"
        
        connection = Connection(
            id=connection_id,
            source_id=source_id,
            target_id=target_id
        )
        
        self.connections[connection_id] = connection
        self.draw_connection(connection)
        self.designer.on_network_changed()
        
        return connection_id
    
    def draw_connection(self, connection: Connection):
        """Нарисовать соединение между узлами"""
        source_node = self.nodes[connection.source_id]
        target_node = self.nodes[connection.target_id]
        
        # Координаты начала и конца соединения
        x1 = source_node.x + source_node.width
        y1 = source_node.y + source_node.height / 2
        x2 = target_node.x
        y2 = target_node.y + target_node.height / 2
        
        # Рисуем стрелку
        self.create_line(
            x1, y1, x2, y2,
            arrow=tk.LAST, fill="black", width=2,
            tags=(connection.id, "connection")
        )
    
    def on_click(self, event):
        """Обработка клика мыши"""
        clicked_item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(clicked_item)
        
        if "node" in tags:
            # Клик по узлу
            node_id = tags[0]
            self.selected_node = node_id
            self.dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.designer.select_node(node_id)
        else:
            # Клик по пустому месту
            self.selected_node = None
            self.designer.select_node(None)
    
    def on_drag(self, event):
        """Обработка перетаскивания"""
        if self.dragging and self.selected_node:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            
            # Перемещаем узел
            self.move(self.selected_node, dx, dy)
            
            # Обновляем координаты в модели
            node = self.nodes[self.selected_node]
            node.x += dx
            node.y += dy
            
            # Перерисовываем соединения
            self._redraw_connections_for_node(self.selected_node)
            
            self.drag_start_x = event.x
            self.drag_start_y = event.y
    
    def on_release(self, event):
        """Обработка отпускания мыши"""
        self.dragging = False
    
    def on_double_click(self, event):
        """Обработка двойного клика"""
        if self.selected_node:
            self.designer.edit_node_properties(self.selected_node)
    
    def on_right_click(self, event):
        """Обработка правого клика"""
        clicked_item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(clicked_item)
        
        if "node" in tags:
            node_id = tags[0]
            self._show_node_context_menu(event, node_id)
        else:
            self._show_canvas_context_menu(event)
    
    def _show_node_context_menu(self, event, node_id: str):
        """Показать контекстное меню для узла"""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Редактировать", 
                        command=lambda: self.designer.edit_node_properties(node_id))
        menu.add_command(label="Удалить", 
                        command=lambda: self.delete_node(node_id))
        menu.add_separator()
        menu.add_command(label="Дублировать", 
                        command=lambda: self.duplicate_node(node_id))
        
        menu.tk_popup(event.x_root, event.y_root)
    
    def _show_canvas_context_menu(self, event):
        """Показать контекстное меню для канваса"""
        menu = tk.Menu(self, tearoff=0)
        
        # Подменю для добавления узлов
        add_menu = tk.Menu(menu, tearoff=0)
        for node_type in NodeType:
            add_menu.add_command(
                label=node_type.value.title(),
                command=lambda nt=node_type: self.add_node(nt, event.x, event.y)
            )
        
        menu.add_cascade(label="Добавить узел", menu=add_menu)
        menu.add_separator()
        menu.add_command(label="Очистить все", command=self.clear_all)
        
        menu.tk_popup(event.x_root, event.y_root)
    
    def delete_node(self, node_id: str):
        """Удалить узел"""
        if node_id in self.nodes:
            # Удаляем все соединения с этим узлом
            connections_to_remove = []
            for conn_id, conn in self.connections.items():
                if conn.source_id == node_id or conn.target_id == node_id:
                    connections_to_remove.append(conn_id)
            
            for conn_id in connections_to_remove:
                self.delete_connection(conn_id)
            
            # Удаляем узел с канваса
            self.delete(node_id)
            
            # Удаляем из модели
            del self.nodes[node_id]
            
            self.designer.on_network_changed()
    
    def delete_connection(self, connection_id: str):
        """Удалить соединение"""
        if connection_id in self.connections:
            self.delete(connection_id)
            del self.connections[connection_id]
    
    def duplicate_node(self, node_id: str):
        """Дублировать узел"""
        if node_id in self.nodes:
            original = self.nodes[node_id]
            new_node_id = self.add_node(
                original.type, 
                original.x + 150, 
                original.y + 50
            )
            # Копируем параметры
            self.nodes[new_node_id].parameters = original.parameters.copy()
    
    def clear_all(self):
        """Очистить весь канвас"""
        if messagebox.askyesno("Подтверждение", "Удалить все узлы и соединения?"):
            self.delete("all")
            self.nodes.clear()
            self.connections.clear()
            self.designer.on_network_changed()
    
    def _redraw_connections_for_node(self, node_id: str):
        """Перерисовать все соединения для узла"""
        for conn_id, conn in self.connections.items():
            if conn.source_id == node_id or conn.target_id == node_id:
                self.delete(conn_id)
                self.draw_connection(conn)


class ComponentPalette(ttk.Frame):
    """Палитра компонентов для drag & drop"""
    
    def __init__(self, parent, designer):
        super().__init__(parent)
        self.designer = designer
        
        # Заголовок
        title_label = ttk.Label(self, text="Компоненты", font=("Arial", 12, "bold"))
        title_label.pack(pady=5)
        
        # Создаем кнопки для каждого типа узла
        self.create_component_buttons()
    
    def create_component_buttons(self):
        """Создать кнопки компонентов"""
        categories = {
            "Основные": [NodeType.INPUT, NodeType.OUTPUT, NodeType.DENSE],
            "Сверточные": [NodeType.CONV2D, NodeType.MAXPOOL, NodeType.AVGPOOL],
            "Рекуррентные": [NodeType.LSTM, NodeType.GRU],
            "Transformer": [NodeType.TRANSFORMER, NodeType.ATTENTION, NodeType.EMBEDDING],
            "Нормализация": [NodeType.BATCH_NORM, NodeType.LAYER_NORM, NodeType.DROPOUT],
            "Активация": [NodeType.ACTIVATION]
        }
        
        for category, node_types in categories.items():
            # Заголовок категории
            category_frame = ttk.LabelFrame(self, text=category)
            category_frame.pack(fill="x", padx=5, pady=2)
            
            # Кнопки компонентов
            for node_type in node_types:
                btn = ttk.Button(
                    category_frame,
                    text=node_type.value.title(),
                    command=lambda nt=node_type: self.designer.set_add_mode(nt)
                )
                btn.pack(fill="x", padx=2, pady=1)


class PropertiesPanel(ttk.Frame):
    """Панель свойств выбранного узла"""
    
    def __init__(self, parent, designer):
        super().__init__(parent)
        self.designer = designer
        self.current_node_id = None
        self.property_vars = {}
        
        # Заголовок
        self.title_label = ttk.Label(self, text="Свойства", font=("Arial", 12, "bold"))
        self.title_label.pack(pady=5)
        
        # Скроллируемая область для свойств
        self.create_scrollable_area()
    
    def create_scrollable_area(self):
        """Создать скроллируемую область для свойств"""
        # Canvas для скроллинга
        self.canvas = tk.Canvas(self, height=400)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def show_node_properties(self, node_id: str):
        """Показать свойства узла"""
        self.current_node_id = node_id
        
        # Очищаем предыдущие свойства
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.property_vars.clear()
        
        if node_id and node_id in self.designer.canvas.nodes:
            node = self.designer.canvas.nodes[node_id]
            
            # Название узла
            ttk.Label(self.scrollable_frame, text=f"Узел: {node.name}").pack(anchor="w", pady=2)
            ttk.Label(self.scrollable_frame, text=f"Тип: {node.type.value}").pack(anchor="w", pady=2)
            
            ttk.Separator(self.scrollable_frame, orient="horizontal").pack(fill="x", pady=5)
            
            # Параметры узла
            for param_name, param_value in node.parameters.items():
                self.create_property_widget(param_name, param_value)
            
            # Кнопка применения изменений
            ttk.Button(
                self.scrollable_frame,
                text="Применить изменения",
                command=self.apply_changes
            ).pack(pady=10)
    
    def create_property_widget(self, param_name: str, param_value: Any):
        """Создать виджет для редактирования параметра"""
        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill="x", pady=2)
        
        # Метка параметра
        ttk.Label(frame, text=f"{param_name}:").pack(side="left")
        
        # Виджет для редактирования
        if isinstance(param_value, bool):
            var = tk.BooleanVar(value=param_value)
            widget = ttk.Checkbutton(frame, variable=var)
        elif isinstance(param_value, (int, float)):
            var = tk.StringVar(value=str(param_value))
            widget = ttk.Entry(frame, textvariable=var, width=15)
        elif isinstance(param_value, str):
            var = tk.StringVar(value=param_value)
            widget = ttk.Entry(frame, textvariable=var, width=15)
        elif isinstance(param_value, list):
            var = tk.StringVar(value=str(param_value))
            widget = ttk.Entry(frame, textvariable=var, width=15)
        else:
            var = tk.StringVar(value=str(param_value))
            widget = ttk.Entry(frame, textvariable=var, width=15)
        
        widget.pack(side="right")
        self.property_vars[param_name] = var
    
    def apply_changes(self):
        """Применить изменения к узлу"""
        if self.current_node_id and self.current_node_id in self.designer.canvas.nodes:
            node = self.designer.canvas.nodes[self.current_node_id]
            
            # Обновляем параметры
            for param_name, var in self.property_vars.items():
                try:
                    value = var.get()
                    
                    # Пытаемся преобразовать к правильному типу
                    if param_name in node.parameters:
                        original_type = type(node.parameters[param_name])
                        if original_type == bool:
                            node.parameters[param_name] = bool(value)
                        elif original_type == int:
                            node.parameters[param_name] = int(value)
                        elif original_type == float:
                            node.parameters[param_name] = float(value)
                        elif original_type == list:
                            node.parameters[param_name] = eval(value)
                        else:
                            node.parameters[param_name] = value
                    else:
                        node.parameters[param_name] = value
                        
                except (ValueError, SyntaxError) as e:
                    messagebox.showerror("Ошибка", f"Неверное значение для {param_name}: {e}")
                    return
            
            self.designer.on_network_changed()
            messagebox.showinfo("Успех", "Изменения применены")


class CodeGenerator:
    """Генератор AnamorphX кода из визуальной схемы"""
    
    def __init__(self, designer):
        self.designer = designer
    
    def generate_anamorphx_code(self) -> str:
        """Генерировать AnamorphX код"""
        canvas = self.designer.canvas
        
        if not canvas.nodes:
            return "// Пустая сеть"
        
        code_lines = []
        code_lines.append("// Сгенерировано Visual Network Designer")
        code_lines.append("")
        
        # Определяем сеть
        network_name = "VisualNetwork"
        code_lines.append(f"network {network_name} {{")
        
        # Добавляем узлы
        for node_id, node in canvas.nodes.items():
            code_lines.extend(self._generate_node_code(node))
            code_lines.append("")
        
        # Добавляем соединения
        if canvas.connections:
            code_lines.append("    // Соединения")
            for conn_id, conn in canvas.connections.items():
                source_node = canvas.nodes[conn.source_id]
                target_node = canvas.nodes[conn.target_id]
                code_lines.append(f"    synap {source_node.name} -> {target_node.name}")
            code_lines.append("")
        
        # Параметры обучения
        code_lines.append("    // Параметры обучения")
        code_lines.append("    optimizer: adam")
        code_lines.append("    learning_rate: 0.001")
        code_lines.append("    loss: categorical_crossentropy")
        
        code_lines.append("}")
        
        return "\n".join(code_lines)
    
    def _generate_node_code(self, node: NodeConfig) -> List[str]:
        """Генерировать код для узла"""
        lines = []
        lines.append(f"    neuron {node.name} {{")
        lines.append(f"        layer_type: \"{node.type.value}\"")
        
        # Добавляем параметры
        for param_name, param_value in node.parameters.items():
            if isinstance(param_value, str):
                lines.append(f"        {param_name}: \"{param_value}\"")
            else:
                lines.append(f"        {param_name}: {param_value}")
        
        lines.append("    }")
        
        return lines
    
    def generate_pytorch_code(self) -> str:
        """Генерировать PyTorch код"""
        try:
            from neural_backend.pytorch_generator import PyTorchGenerator
            from neural_backend.network_parser import NetworkConfig, NeuronConfig
            
            # Преобразуем визуальную схему в конфигурацию
            neurons = []
            for node_id, node in self.designer.canvas.nodes.items():
                neuron_config = NeuronConfig(
                    name=node.name,
                    layer_type=node.type.value,
                    **node.parameters
                )
                neurons.append(neuron_config)
            
            network_config = NetworkConfig(
                name="VisualNetwork",
                neurons=neurons
            )
            
            # Генерируем PyTorch код
            generator = PyTorchGenerator()
            return generator.generate_model_class(network_config)
            
        except ImportError:
            return "// Ошибка: Neural Backend недоступен"
        except Exception as e:
            return f"// Ошибка генерации PyTorch кода: {e}"


class VisualNetworkDesigner(ttk.Frame):
    """Главный класс визуального дизайнера сетей"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.add_mode_type = None
        
        self.create_interface()
        self.code_generator = CodeGenerator(self)
    
    def create_interface(self):
        """Создать интерфейс дизайнера"""
        # Главный контейнер
        main_paned = ttk.PanedWindow(self, orient="horizontal")
        main_paned.pack(fill="both", expand=True)
        
        # Левая панель (палитра компонентов)
        left_frame = ttk.Frame(main_paned, width=200)
        self.palette = ComponentPalette(left_frame, self)
        self.palette.pack(fill="both", expand=True)
        main_paned.add(left_frame)
        
        # Центральная панель (канвас)
        center_frame = ttk.Frame(main_paned)
        
        # Панель инструментов
        toolbar = ttk.Frame(center_frame)
        toolbar.pack(fill="x", pady=2)
        
        ttk.Button(toolbar, text="Новая сеть", command=self.new_network).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Загрузить", command=self.load_network).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Сохранить", command=self.save_network).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", padx=5)
        ttk.Button(toolbar, text="Генерировать код", command=self.show_generated_code).pack(side="left", padx=2)
        ttk.Button(toolbar, text="PyTorch", command=self.show_pytorch_code).pack(side="left", padx=2)
        
        # Канвас
        canvas_frame = ttk.Frame(center_frame)
        canvas_frame.pack(fill="both", expand=True)
        
        self.canvas = NetworkCanvas(canvas_frame, self)
        self.canvas.pack(fill="both", expand=True)
        
        main_paned.add(center_frame)
        
        # Правая панель (свойства)
        right_frame = ttk.Frame(main_paned, width=250)
        self.properties = PropertiesPanel(right_frame, self)
        self.properties.pack(fill="both", expand=True)
        main_paned.add(right_frame)
    
    def set_add_mode(self, node_type: NodeType):
        """Установить режим добавления узла"""
        self.add_mode_type = node_type
        self.canvas.config(cursor="crosshair")
        
        # Привязываем обработчик клика для добавления узла
        self.canvas.bind("<Button-1>", self.add_node_at_position, add="+")
    
    def add_node_at_position(self, event):
        """Добавить узел в позицию клика"""
        if self.add_mode_type:
            self.canvas.add_node(self.add_mode_type, event.x, event.y)
            self.add_mode_type = None
            self.canvas.config(cursor="")
            self.canvas.unbind("<Button-1>")
    
    def select_node(self, node_id: Optional[str]):
        """Выбрать узел"""
        self.properties.show_node_properties(node_id)
    
    def edit_node_properties(self, node_id: str):
        """Редактировать свойства узла"""
        self.select_node(node_id)
    
    def on_network_changed(self):
        """Обработчик изменения сети"""
        # Здесь можно добавить автосохранение или валидацию
        pass
    
    def new_network(self):
        """Создать новую сеть"""
        if messagebox.askyesno("Подтверждение", "Создать новую сеть? Текущая сеть будет потеряна."):
            self.canvas.clear_all()
    
    def save_network(self):
        """Сохранить сеть в файл"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                network_data = {
                    "nodes": {node_id: asdict(node) for node_id, node in self.canvas.nodes.items()},
                    "connections": {conn_id: asdict(conn) for conn_id, conn in self.canvas.connections.items()}
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(network_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Успех", f"Сеть сохранена в {filename}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")
    
    def load_network(self):
        """Загрузить сеть из файла"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    network_data = json.load(f)
                
                # Очищаем текущую сеть
                self.canvas.clear_all()
                
                # Загружаем узлы
                for node_id, node_data in network_data.get("nodes", {}).items():
                    node_data["type"] = NodeType(node_data["type"])
                    node = NodeConfig(**node_data)
                    self.canvas.nodes[node_id] = node
                    self.canvas.draw_node(node)
                
                # Загружаем соединения
                for conn_id, conn_data in network_data.get("connections", {}).items():
                    conn = Connection(**conn_data)
                    self.canvas.connections[conn_id] = conn
                    self.canvas.draw_connection(conn)
                
                messagebox.showinfo("Успех", f"Сеть загружена из {filename}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")
    
    def show_generated_code(self):
        """Показать сгенерированный AnamorphX код"""
        code = self.code_generator.generate_anamorphx_code()
        self._show_code_window("AnamorphX код", code)
    
    def show_pytorch_code(self):
        """Показать сгенерированный PyTorch код"""
        code = self.code_generator.generate_pytorch_code()
        self._show_code_window("PyTorch код", code)
    
    def _show_code_window(self, title: str, code: str):
        """Показать окно с кодом"""
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("800x600")
        
        # Текстовое поле с кодом
        text_frame = ttk.Frame(window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap="none", font=("Consolas", 10))
        scrollbar_y = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        scrollbar_x = ttk.Scrollbar(text_frame, orient="horizontal", command=text_widget.xview)
        
        text_widget.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        text_widget.insert("1.0", code)
        text_widget.config(state="disabled")
        
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        
        # Кнопки
        button_frame = ttk.Frame(window)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(button_frame, text="Копировать", 
                  command=lambda: self._copy_to_clipboard(code)).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Сохранить в файл", 
                  command=lambda: self._save_code_to_file(code)).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Закрыть", 
                  command=window.destroy).pack(side="right", padx=5)
    
    def _copy_to_clipboard(self, text: str):
        """Копировать текст в буфер обмена"""
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Успех", "Код скопирован в буфер обмена")
    
    def _save_code_to_file(self, code: str):
        """Сохранить код в файл"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                messagebox.showinfo("Успех", f"Код сохранен в {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")


def main():
    """Тестирование Visual Network Designer"""
    root = tk.Tk()
    root.title("AnamorphX Visual Network Designer")
    root.geometry("1200x800")
    
    designer = VisualNetworkDesigner(root)
    designer.pack(fill="both", expand=True)
    
    root.mainloop()


if __name__ == "__main__":
    main() 