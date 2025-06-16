"""
Визуализация профайлера для AnamorphX IDE

Возможности:
- Интерактивные графики производительности
- Диаграммы использования памяти
- Анализ горячих функций
- Рекомендации по оптимизации
- Экспорт отчетов
"""

import tkinter as tk
from tkinter import ttk, Canvas, Scrollbar, Frame, Label, Button
import math
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .profiler import AnamorphProfiler, FunctionProfile, NeuralProfile, PerformanceAnalyzer


class ChartType(Enum):
    """Типы диаграмм"""
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    LINE_CHART = "line_chart"
    FLAME_GRAPH = "flame_graph"


@dataclass
class ChartData:
    """Данные для построения диаграммы"""
    labels: List[str]
    values: List[float]
    colors: List[str]
    title: str
    x_label: str = ""
    y_label: str = ""


class SimpleChart:
    """Простая система построения диаграмм на Canvas"""
    
    def __init__(self, canvas: Canvas):
        self.canvas = canvas
        self.width = 400
        self.height = 300
        self.margin = 40
        
        # Цветовая схема
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
            '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA'
        ]
    
    def clear(self):
        """Очистить canvas"""
        self.canvas.delete("all")
    
    def draw_bar_chart(self, data: ChartData):
        """Нарисовать столбчатую диаграмму"""
        self.clear()
        
        if not data.values:
            return
        
        # Размеры области построения
        plot_width = self.width - 2 * self.margin
        plot_height = self.height - 2 * self.margin
        
        # Масштабирование
        max_value = max(data.values) if data.values else 1
        scale_y = plot_height / max_value if max_value > 0 else 1
        
        # Ширина столбца
        bar_width = plot_width / len(data.values) * 0.8
        bar_spacing = plot_width / len(data.values)
        
        # Отрисовка столбцов
        for i, (value, label) in enumerate(zip(data.values, data.labels)):
            x = self.margin + i * bar_spacing + (bar_spacing - bar_width) / 2
            y = self.height - self.margin
            bar_height = value * scale_y
            
            # Цвет столбца
            color = data.colors[i] if i < len(data.colors) else self.colors[i % len(self.colors)]
            
            # Столбец
            self.canvas.create_rectangle(
                x, y, x + bar_width, y - bar_height,
                fill=color, outline='black', width=1
            )
            
            # Значение на столбце
            if bar_height > 20:  # Если столбец достаточно высокий
                self.canvas.create_text(
                    x + bar_width / 2, y - bar_height / 2,
                    text=f"{value:.2f}", font=('Arial', 8), fill='white'
                )
            
            # Подпись под столбцом
            self.canvas.create_text(
                x + bar_width / 2, y + 15,
                text=label[:10] + ('...' if len(label) > 10 else ''),
                font=('Arial', 8), angle=45 if len(label) > 8 else 0
            )
        
        # Заголовок
        self.canvas.create_text(
            self.width / 2, 20,
            text=data.title, font=('Arial', 12, 'bold')
        )
        
        # Оси
        self._draw_axes(data.x_label, data.y_label)
    
    def draw_pie_chart(self, data: ChartData):
        """Нарисовать круговую диаграмму"""
        self.clear()
        
        if not data.values:
            return
        
        # Центр и радиус
        center_x = self.width / 2
        center_y = self.height / 2 + 20
        radius = min(self.width, self.height) / 3
        
        # Общая сумма
        total = sum(data.values)
        if total == 0:
            return
        
        # Отрисовка секторов
        start_angle = 0
        legend_y = 50
        
        for i, (value, label) in enumerate(zip(data.values, data.labels)):
            if value == 0:
                continue
            
            # Угол сектора
            angle = (value / total) * 360
            
            # Цвет
            color = data.colors[i] if i < len(data.colors) else self.colors[i % len(self.colors)]
            
            # Сектор
            self.canvas.create_arc(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                start=start_angle, extent=angle,
                fill=color, outline='black', width=1
            )
            
            # Подпись в легенде
            legend_x = self.width - 150
            self.canvas.create_rectangle(
                legend_x, legend_y, legend_x + 15, legend_y + 15,
                fill=color, outline='black'
            )
            
            percentage = (value / total) * 100
            self.canvas.create_text(
                legend_x + 20, legend_y + 7,
                text=f"{label}: {percentage:.1f}%", 
                font=('Arial', 9), anchor='w'
            )
            
            legend_y += 20
            start_angle += angle
        
        # Заголовок
        self.canvas.create_text(
            self.width / 2, 20,
            text=data.title, font=('Arial', 12, 'bold')
        )
    
    def draw_line_chart(self, data: ChartData):
        """Нарисовать линейную диаграмму"""
        self.clear()
        
        if len(data.values) < 2:
            return
        
        # Размеры области построения
        plot_width = self.width - 2 * self.margin
        plot_height = self.height - 2 * self.margin
        
        # Масштабирование
        max_value = max(data.values) if data.values else 1
        min_value = min(data.values) if data.values else 0
        value_range = max_value - min_value if max_value != min_value else 1
        
        # Точки линии
        points = []
        for i, value in enumerate(data.values):
            x = self.margin + (i / (len(data.values) - 1)) * plot_width
            y = self.height - self.margin - ((value - min_value) / value_range) * plot_height
            points.extend([x, y])
        
        # Отрисовка линии
        if len(points) >= 4:
            self.canvas.create_line(
                points, fill='blue', width=2, smooth=True
            )
        
        # Точки данных
        for i in range(0, len(points), 2):
            x, y = points[i], points[i + 1]
            self.canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3,
                fill='red', outline='black'
            )
        
        # Заголовок
        self.canvas.create_text(
            self.width / 2, 20,
            text=data.title, font=('Arial', 12, 'bold')
        )
        
        # Оси
        self._draw_axes(data.x_label, data.y_label)
    
    def draw_flame_graph(self, call_stack_data: List[Dict]):
        """Нарисовать flame graph для стека вызовов"""
        self.clear()
        
        if not call_stack_data:
            return
        
        # Размеры
        plot_width = self.width - 2 * self.margin
        plot_height = self.height - 2 * self.margin
        
        # Высота одного уровня
        level_height = 20
        max_levels = plot_height // level_height
        
        # Группировка по уровням
        levels = {}
        for call in call_stack_data:
            level = call.get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(call)
        
        # Отрисовка уровней
        for level, calls in levels.items():
            if level >= max_levels:
                break
            
            y = self.margin + level * level_height
            total_time = sum(call.get('time', 0) for call in calls)
            
            if total_time == 0:
                continue
            
            x_offset = self.margin
            
            for call in calls:
                call_time = call.get('time', 0)
                width = (call_time / total_time) * plot_width
                
                if width < 1:
                    continue
                
                # Цвет зависит от времени
                intensity = min(255, int((call_time / max(total_time, 1)) * 255))
                color = f"#{intensity:02x}{255-intensity:02x}00"
                
                # Прямоугольник функции
                self.canvas.create_rectangle(
                    x_offset, y, x_offset + width, y + level_height,
                    fill=color, outline='black', width=1
                )
                
                # Название функции (если помещается)
                if width > 50:
                    function_name = call.get('function', 'unknown')
                    self.canvas.create_text(
                        x_offset + width / 2, y + level_height / 2,
                        text=function_name[:int(width/8)],
                        font=('Arial', 8), fill='white'
                    )
                
                x_offset += width
        
        # Заголовок
        self.canvas.create_text(
            self.width / 2, 20,
            text="Flame Graph - Профиль выполнения", 
            font=('Arial', 12, 'bold')
        )
    
    def _draw_axes(self, x_label: str, y_label: str):
        """Нарисовать оси координат"""
        # Ось X
        self.canvas.create_line(
            self.margin, self.height - self.margin,
            self.width - self.margin, self.height - self.margin,
            fill='black', width=2
        )
        
        # Ось Y
        self.canvas.create_line(
            self.margin, self.margin,
            self.margin, self.height - self.margin,
            fill='black', width=2
        )
        
        # Подписи осей
        if x_label:
            self.canvas.create_text(
                self.width / 2, self.height - 10,
                text=x_label, font=('Arial', 10)
            )
        
        if y_label:
            self.canvas.create_text(
                10, self.height / 2,
                text=y_label, font=('Arial', 10), angle=90
            )


class FunctionPerformancePanel:
    """Панель производительности функций"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Заголовок
        ttk.Label(self.frame, text="Производительность функций", 
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        
        # Notebook для разных представлений
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Вкладка "Топ функций"
        self.top_functions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.top_functions_frame, text="Топ функций")
        
        # Таблица топ функций
        self.functions_tree = ttk.Treeview(
            self.top_functions_frame,
            columns=('time', 'calls', 'avg_time', 'percentage'),
            show='tree headings',
            height=10
        )
        
        self.functions_tree.heading('#0', text='Функция')
        self.functions_tree.heading('time', text='Время (s)')
        self.functions_tree.heading('calls', text='Вызовов')
        self.functions_tree.heading('avg_time', text='Среднее (ms)')
        self.functions_tree.heading('percentage', text='%')
        
        self.functions_tree.column('#0', width=200)
        self.functions_tree.column('time', width=80)
        self.functions_tree.column('calls', width=80)
        self.functions_tree.column('avg_time', width=80)
        self.functions_tree.column('percentage', width=60)
        
        self.functions_tree.pack(fill='both', expand=True)
        
        # Вкладка "Диаграмма времени"
        self.time_chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.time_chart_frame, text="Диаграмма времени")
        
        self.time_chart_canvas = Canvas(self.time_chart_frame, width=500, height=300, bg='white')
        self.time_chart_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.time_chart = SimpleChart(self.time_chart_canvas)
        
        # Вкладка "Распределение вызовов"
        self.calls_chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.calls_chart_frame, text="Вызовы")
        
        self.calls_chart_canvas = Canvas(self.calls_chart_frame, width=500, height=300, bg='white')
        self.calls_chart_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.calls_chart = SimpleChart(self.calls_chart_canvas)
    
    def update_data(self, function_profiles: List[FunctionProfile], total_time: float):
        """Обновить данные производительности"""
        # Очистка таблицы
        for item in self.functions_tree.get_children():
            self.functions_tree.delete(item)
        
        # Заполнение таблицы
        for profile in function_profiles[:20]:  # Топ 20
            percentage = (profile.total_time / total_time * 100) if total_time > 0 else 0
            
            self.functions_tree.insert('', 'end', 
                text=profile.name,
                values=(
                    f"{profile.total_time:.3f}",
                    str(profile.call_count),
                    f"{profile.avg_time * 1000:.2f}",
                    f"{percentage:.1f}%"
                )
            )
        
        # Обновление диаграмм
        self._update_charts(function_profiles[:10])  # Топ 10 для диаграмм
    
    def _update_charts(self, profiles: List[FunctionProfile]):
        """Обновить диаграммы"""
        if not profiles:
            return
        
        # Диаграмма времени
        time_data = ChartData(
            labels=[p.name[:15] for p in profiles],
            values=[p.total_time for p in profiles],
            colors=self.time_chart.colors[:len(profiles)],
            title="Время выполнения функций",
            x_label="Функции",
            y_label="Время (s)"
        )
        
        self.time_chart.draw_bar_chart(time_data)
        
        # Диаграмма вызовов
        calls_data = ChartData(
            labels=[p.name[:15] for p in profiles],
            values=[float(p.call_count) for p in profiles],
            colors=self.calls_chart.colors[:len(profiles)],
            title="Количество вызовов",
            x_label="Функции", 
            y_label="Вызовов"
        )
        
        self.calls_chart.draw_bar_chart(calls_data)


class MemoryUsagePanel:
    """Панель использования памяти"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Заголовок
        ttk.Label(self.frame, text="Использование памяти", 
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        
        # График временной линии памяти
        self.memory_canvas = Canvas(self.frame, width=500, height=200, bg='white')
        self.memory_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.memory_chart = SimpleChart(self.memory_canvas)
        
        # Статистика памяти
        self.stats_frame = ttk.Frame(self.frame)
        self.stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.peak_label = ttk.Label(self.stats_frame, text="Пик: 0 MB")
        self.peak_label.pack(side='left', padx=10)
        
        self.avg_label = ttk.Label(self.stats_frame, text="Среднее: 0 MB")
        self.avg_label.pack(side='left', padx=10)
        
        self.current_label = ttk.Label(self.stats_frame, text="Текущее: 0 MB")
        self.current_label.pack(side='left', padx=10)
    
    def update_data(self, memory_timeline: List[Tuple[float, float]]):
        """Обновить данные памяти"""
        if not memory_timeline:
            return
        
        # Извлечение данных
        times = [t[0] for t in memory_timeline]
        memory_values = [t[1] for t in memory_timeline]
        
        # Создание временных меток (каждая 10-я точка)
        step = max(1, len(times) // 10)
        labels = [f"{t:.1f}s" if i % step == 0 else "" for i, t in enumerate(times)]
        
        # График временной линии
        memory_data = ChartData(
            labels=labels,
            values=memory_values,
            colors=['blue'] * len(memory_values),
            title="Использование памяти во времени",
            x_label="Время",
            y_label="Память (MB)"
        )
        
        self.memory_chart.draw_line_chart(memory_data)
        
        # Обновление статистики
        if memory_values:
            peak = max(memory_values)
            avg = sum(memory_values) / len(memory_values)
            current = memory_values[-1]
            
            self.peak_label.config(text=f"Пик: {peak:.1f} MB")
            self.avg_label.config(text=f"Среднее: {avg:.1f} MB") 
            self.current_label.config(text=f"Текущее: {current:.1f} MB")


class NeuralPerformancePanel:
    """Панель производительности нейронных операций"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Заголовок
        ttk.Label(self.frame, text="Нейронная производительность", 
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        
        # Таблица нейронных операций
        self.neural_tree = ttk.Treeview(
            self.frame,
            columns=('type', 'object', 'time', 'count', 'avg_time', 'activations'),
            show='tree headings',
            height=8
        )
        
        self.neural_tree.heading('#0', text='Операция')
        self.neural_tree.heading('type', text='Тип')
        self.neural_tree.heading('object', text='Объект')
        self.neural_tree.heading('time', text='Время (s)')
        self.neural_tree.heading('count', text='Выполнений')
        self.neural_tree.heading('avg_time', text='Среднее (ms)')
        self.neural_tree.heading('activations', text='Активации')
        
        self.neural_tree.pack(fill='both', expand=True)
        
        # График распределения операций
        self.neural_canvas = Canvas(self.frame, width=500, height=200, bg='white')
        self.neural_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.neural_chart = SimpleChart(self.neural_canvas)
    
    def update_data(self, neural_profiles: List[NeuralProfile]):
        """Обновить данные нейронных операций"""
        # Очистка таблицы
        for item in self.neural_tree.get_children():
            self.neural_tree.delete(item)
        
        # Заполнение таблицы
        for profile in neural_profiles:
            self.neural_tree.insert('', 'end',
                text=f"{profile.neural_object}::{profile.operation_type}",
                values=(
                    profile.operation_type,
                    profile.neural_object,
                    f"{profile.total_time:.3f}",
                    str(profile.execution_count),
                    f"{profile.avg_time_per_operation * 1000:.2f}",
                    str(profile.total_activations)
                )
            )
        
        # Диаграмма распределения времени по типам операций
        if neural_profiles:
            operation_times = {}
            for profile in neural_profiles:
                op_type = profile.operation_type
                if op_type not in operation_times:
                    operation_times[op_type] = 0
                operation_times[op_type] += profile.total_time
            
            if operation_times:
                pie_data = ChartData(
                    labels=list(operation_times.keys()),
                    values=list(operation_times.values()),
                    colors=self.neural_chart.colors[:len(operation_times)],
                    title="Распределение времени по типам операций"
                )
                
                self.neural_chart.draw_pie_chart(pie_data)


class RecommendationsPanel:
    """Панель рекомендаций по оптимизации"""
    
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        # Заголовок
        ttk.Label(self.frame, text="Рекомендации по оптимизации", 
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)
        
        # Текстовое поле для рекомендаций
        self.recommendations_text = tk.Text(
            self.frame,
            height=15,
            wrap=tk.WORD,
            font=('Arial', 10),
            state='disabled'
        )
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(self.frame, orient='vertical', 
                                 command=self.recommendations_text.yview)
        self.recommendations_text.configure(yscrollcommand=scrollbar.set)
        
        # Размещение
        self.recommendations_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Настройка тегов для разных типов рекомендаций
        self.recommendations_text.tag_configure('critical', foreground='red', font=('Arial', 10, 'bold'))
        self.recommendations_text.tag_configure('warning', foreground='orange', font=('Arial', 10, 'bold'))
        self.recommendations_text.tag_configure('info', foreground='blue', font=('Arial', 10, 'bold'))
        self.recommendations_text.tag_configure('tip', foreground='green', font=('Arial', 10, 'bold'))
    
    def update_recommendations(self, analysis: Dict[str, Any]):
        """Обновить рекомендации"""
        self.recommendations_text.config(state='normal')
        self.recommendations_text.delete('1.0', 'end')
        
        # Анализ критических проблем
        cpu_hotspots = analysis.get('cpu_hotspots', [])
        memory_hotspots = analysis.get('memory_hotspots', [])
        neural_bottlenecks = analysis.get('neural_bottlenecks', [])
        recommendations = analysis.get('recommendations', [])
        
        # Критические проблемы производительности
        if cpu_hotspots:
            self.recommendations_text.insert('end', "🔥 КРИТИЧЕСКИЕ ПРОБЛЕМЫ CPU\n", 'critical')
            for hotspot in cpu_hotspots[:3]:
                text = f"• {hotspot['function']}: {hotspot['total_time']:.3f}s ({hotspot['percentage']:.1f}%)\n"
                text += f"  Вызовов: {hotspot['call_count']}, Среднее: {hotspot['avg_time']*1000:.2f}ms\n\n"
                self.recommendations_text.insert('end', text)
        
        # Проблемы с памятью
        if memory_hotspots:
            self.recommendations_text.insert('end', "💾 ПРОБЛЕМЫ С ПАМЯТЬЮ\n", 'warning')
            for hotspot in memory_hotspots[:3]:
                if hotspot['avg_memory'] > 0:
                    text = f"• {hotspot['function']}: {hotspot['avg_memory']:.1f}MB в среднем\n"
                    text += f"  Аллокаций: {hotspot['total_allocations']}, Пик: {hotspot['peak_memory']:.1f}MB\n\n"
                    self.recommendations_text.insert('end', text)
        
        # Нейронные узкие места
        if neural_bottlenecks:
            self.recommendations_text.insert('end', "🧠 НЕЙРОННЫЕ УЗКИЕ МЕСТА\n", 'warning')
            for bottleneck in neural_bottlenecks[:3]:
                text = f"• {bottleneck['operation']}: {bottleneck['total_time']:.3f}s\n"
                text += f"  Выполнений: {bottleneck['execution_count']}, Среднее: {bottleneck['avg_time']*1000:.2f}ms\n\n"
                self.recommendations_text.insert('end', text)
        
        # Общие рекомендации
        if recommendations:
            self.recommendations_text.insert('end', "💡 РЕКОМЕНДАЦИИ\n", 'tip')
            for i, rec in enumerate(recommendations, 1):
                self.recommendations_text.insert('end', f"{i}. {rec}\n\n")
        
        # Конкретные советы по оптимизации
        self.recommendations_text.insert('end', "🎯 КОНКРЕТНЫЕ СОВЕТЫ\n", 'info')
        
        optimization_tips = [
            "Используйте кэширование для часто вызываемых функций",
            "Оптимизируйте алгоритмы с высокой временной сложностью",
            "Применяйте векторизацию для нейронных операций",
            "Рассмотрите использование асинхронного выполнения",
            "Оптимизируйте структуры данных для уменьшения использования памяти",
            "Используйте профилирование для выявления узких мест",
            "Рассмотрите возможность параллелизации вычислений",
            "Оптимизируйте доступ к данным для лучшей кэш-локальности"
        ]
        
        for tip in optimization_tips:
            self.recommendations_text.insert('end', f"• {tip}\n")
        
        self.recommendations_text.config(state='disabled')


class ProfilerVisualizer:
    """Главный класс визуализации профайлера"""
    
    def __init__(self, ide_components):
        self.ide = ide_components
        self.profiler: Optional[AnamorphProfiler] = None
        
        # Панели визуализации
        self.function_panel: Optional[FunctionPerformancePanel] = None
        self.memory_panel: Optional[MemoryUsagePanel] = None
        self.neural_panel: Optional[NeuralPerformancePanel] = None
        self.recommendations_panel: Optional[RecommendationsPanel] = None
        
        # Создание UI
        self._create_profiler_ui()
    
    def _create_profiler_ui(self):
        """Создание интерфейса профайлера"""
        # Создание новой вкладки в bottom_notebook
        if hasattr(self.ide, 'bottom_notebook'):
            # Фрейм профайлера
            self.profiler_frame = ttk.Frame(self.ide.bottom_notebook)
            self.ide.bottom_notebook.add(self.profiler_frame, text="Профайлер")
            
            # Notebook для разных панелей
            self.profiler_notebook = ttk.Notebook(self.profiler_frame)
            self.profiler_notebook.pack(fill='both', expand=True)
            
            # Панель функций
            functions_frame = ttk.Frame(self.profiler_notebook)
            self.profiler_notebook.add(functions_frame, text="Функции")
            self.function_panel = FunctionPerformancePanel(functions_frame)
            self.function_panel.frame.pack(fill='both', expand=True)
            
            # Панель памяти
            memory_frame = ttk.Frame(self.profiler_notebook)
            self.profiler_notebook.add(memory_frame, text="Память")
            self.memory_panel = MemoryUsagePanel(memory_frame)
            self.memory_panel.frame.pack(fill='both', expand=True)
            
            # Панель нейронных операций
            neural_frame = ttk.Frame(self.profiler_notebook)
            self.profiler_notebook.add(neural_frame, text="Нейронные")
            self.neural_panel = NeuralPerformancePanel(neural_frame)
            self.neural_panel.frame.pack(fill='both', expand=True)
            
            # Панель рекомендаций
            recommendations_frame = ttk.Frame(self.profiler_notebook)
            self.profiler_notebook.add(recommendations_frame, text="Рекомендации")
            self.recommendations_panel = RecommendationsPanel(recommendations_frame)
            self.recommendations_panel.frame.pack(fill='both', expand=True)
            
            # Кнопки управления
            self._create_control_buttons()
    
    def _create_control_buttons(self):
        """Создание кнопок управления"""
        if not hasattr(self, 'profiler_frame'):
            return
        
        control_frame = ttk.Frame(self.profiler_frame)
        control_frame.pack(side='bottom', fill='x', pady=5)
        
        ttk.Button(control_frame, text="📊 Обновить", 
                  command=self.refresh_data).pack(side='left', padx=5)
        ttk.Button(control_frame, text="💾 Экспорт HTML", 
                  command=self.export_html_report).pack(side='left', padx=5)
        ttk.Button(control_frame, text="💾 Экспорт JSON", 
                  command=self.export_json_report).pack(side='left', padx=5)
        ttk.Button(control_frame, text="🗑️ Очистить", 
                  command=self.clear_data).pack(side='left', padx=5)
    
    def set_profiler(self, profiler: AnamorphProfiler):
        """Установить профайлер"""
        self.profiler = profiler
    
    def update_data(self, report: Dict[str, Any]):
        """Обновить данные профайлера"""
        if not report:
            return
        
        # Извлечение данных из отчета
        summary = report.get('summary', {})
        top_functions = report.get('top_functions', [])
        neural_performance = report.get('neural_performance', [])
        memory_analysis = report.get('memory_analysis', {})
        performance_analysis = report.get('performance_analysis', {})
        
        # Преобразование в объекты профилей
        function_profiles = []
        for func_data in top_functions:
            profile = FunctionProfile(
                name=func_data['name'],
                file_path=func_data.get('file_path', ''),
                line=0,
                call_count=func_data['call_count'],
                total_time=func_data['total_time'],
                max_time=0,
                min_time=0
            )
            function_profiles.append(profile)
        
        neural_profiles = []
        for neural_data in neural_performance:
            profile = NeuralProfile(
                operation_type=neural_data['operation'].split('::')[1] if '::' in neural_data['operation'] else neural_data['operation'],
                neural_object=neural_data['operation'].split('::')[0] if '::' in neural_data['operation'] else 'unknown',
                execution_count=neural_data['execution_count'],
                total_time=neural_data['total_time'],
                total_activations=neural_data.get('activations', 0)
            )
            neural_profiles.append(profile)
        
        # Обновление панелей
        if self.function_panel:
            total_time = summary.get('total_execution_time', 1)
            self.function_panel.update_data(function_profiles, total_time)
        
        if self.memory_panel:
            memory_timeline = memory_analysis.get('timeline', [])
            self.memory_panel.update_data(memory_timeline)
        
        if self.neural_panel:
            self.neural_panel.update_data(neural_profiles)
        
        if self.recommendations_panel:
            self.recommendations_panel.update_recommendations(performance_analysis)
    
    def refresh_data(self):
        """Обновить данные из активного профайлера"""
        if self.profiler and hasattr(self.profiler, 'active_sessions'):
            for session_name, analyzer in self.profiler.active_sessions.items():
                if analyzer.is_profiling:
                    # Генерация промежуточного отчета
                    temp_report = analyzer.generate_report()
                    self.update_data(temp_report)
                    break
    
    def export_html_report(self):
        """Экспорт HTML отчета"""
        if self.profiler and hasattr(self.profiler, 'active_sessions'):
            for session_name, analyzer in self.profiler.active_sessions.items():
                filename = f"profiler_report_{int(time.time())}.html"
                analyzer.export_report(filename, "html")
                print(f"HTML отчет сохранен: {filename}")
                break
    
    def export_json_report(self):
        """Экспорт JSON отчета"""
        if self.profiler and hasattr(self.profiler, 'active_sessions'):
            for session_name, analyzer in self.profiler.active_sessions.items():
                filename = f"profiler_report_{int(time.time())}.json"
                analyzer.export_report(filename, "json")
                print(f"JSON отчет сохранен: {filename}")
                break
    
    def clear_data(self):
        """Очистить данные"""
        # Очистка всех панелей
        if self.function_panel:
            self.function_panel.update_data([], 0)
        
        if self.memory_panel:
            self.memory_panel.update_data([])
        
        if self.neural_panel:
            self.neural_panel.update_data([])
        
        if self.recommendations_panel:
            self.recommendations_panel.update_recommendations({})


def integrate_profiler_visualizer(ide_components) -> ProfilerVisualizer:
    """Интеграция визуализации профайлера с IDE"""
    visualizer = ProfilerVisualizer(ide_components)
    return visualizer


if __name__ == "__main__":
    # Тестирование визуализации профайлера
    import tkinter as tk
    from .profiler import FunctionProfile, NeuralProfile
    
    # Создание тестового окна
    root = tk.Tk()
    root.title("Тест визуализации профайлера")
    root.geometry("800x600")
    
    # Имитация IDE
    class MockIDE:
        def __init__(self):
            self.bottom_notebook = ttk.Notebook(root)
            self.bottom_notebook.pack(fill='both', expand=True)
    
    ide = MockIDE()
    
    # Создание визуализатора
    visualizer = ProfilerVisualizer(ide)
    
    # Тестовые данные
    test_report = {
        'summary': {
            'total_execution_time': 10.5,
            'total_function_calls': 1500,
            'peak_memory_usage': 125.6
        },
        'top_functions': [
            {'name': 'factorial', 'total_time': 3.2, 'call_count': 100, 'file_path': 'test.py'},
            {'name': 'matrix_mult', 'total_time': 2.8, 'call_count': 50, 'file_path': 'test.py'},
            {'name': 'neural_forward', 'total_time': 2.1, 'call_count': 200, 'file_path': 'neural.py'},
            {'name': 'data_preprocessing', 'total_time': 1.4, 'call_count': 25, 'file_path': 'data.py'},
            {'name': 'loss_calculation', 'total_time': 1.0, 'call_count': 75, 'file_path': 'loss.py'}
        ],
        'neural_performance': [
            {'operation': 'neuron1::activation', 'total_time': 1.5, 'execution_count': 100, 'activations': 1000},
            {'operation': 'network::forward_pass', 'total_time': 1.2, 'execution_count': 50, 'activations': 500},
            {'operation': 'synapse1::weight_update', 'total_time': 0.8, 'execution_count': 75, 'activations': 0}
        ],
        'memory_analysis': {
            'timeline': [(i*0.1, 50 + i*2 + (i%5)*10) for i in range(100)]
        },
        'performance_analysis': {
            'cpu_hotspots': [
                {'function': 'factorial', 'total_time': 3.2, 'percentage': 30.5, 'call_count': 100, 'avg_time': 0.032}
            ],
            'memory_hotspots': [
                {'function': 'matrix_mult', 'avg_memory': 45.2, 'total_allocations': 50, 'peak_memory': 67.8}
            ],
            'neural_bottlenecks': [
                {'operation': 'neuron1::activation', 'total_time': 1.5, 'execution_count': 100, 'avg_time': 0.015}
            ],
            'recommendations': [
                '🔥 Найдена медленная функция factorial - рассмотрите мемоизацию',
                '💾 Высокое использование памяти в matrix_mult - оптимизируйте алгоритм',
                '🧠 Медленная нейронная активация - используйте векторизацию'
            ]
        }
    }
    
    # Загрузка тестовых данных
    visualizer.update_data(test_report)
    
    print("📊 Визуализация профайлера готова к тестированию")
    print("Переключайтесь между вкладками для просмотра разных аспектов производительности")
    
    root.mainloop() 