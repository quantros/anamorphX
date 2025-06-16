"""
Инкрементальная подсветка синтаксиса для AnamorphX

Особенности:
- Подсветка в реальном времени при вводе
- Оптимизированная перерисовка только измененных областей
- Поддержка больших файлов
- Кэширование результатов
- Асинхронная обработка
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import re
from enum import Enum

from .syntax_highlighter import (
    AnamorphSyntaxHighlighter, 
    HighlightToken, 
    TokenType, 
    HighlightTheme
)


class ChangeType(Enum):
    """Типы изменений в тексте"""
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass
class TextChange:
    """Изменение в тексте"""
    type: ChangeType
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    old_text: str
    new_text: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class HighlightRegion:
    """Регион для подсветки"""
    start_line: int
    end_line: int
    tokens: List[HighlightToken]
    last_updated: float = field(default_factory=time.time)
    is_dirty: bool = False


@dataclass
class ParsedLine:
    """Распарсенная строка кода"""
    line_number: int
    content: str
    tokens: List[HighlightToken]
    hash_value: int
    dependencies: Set[int] = field(default_factory=set)  # Зависимые строки
    last_parsed: float = field(default_factory=time.time)


class IncrementalHighlighter:
    """Инкрементальная подсветка синтаксиса"""
    
    def __init__(self, theme: Optional[HighlightTheme] = None, 
                 chunk_size: int = 50, 
                 cache_size: int = 1000):
        
        self.base_highlighter = AnamorphSyntaxHighlighter(theme)
        self.chunk_size = chunk_size  # Размер блока для обработки
        self.cache_size = cache_size
        
        # Кэш парсинга по строкам
        self.line_cache: Dict[int, ParsedLine] = {}
        self.region_cache: Dict[Tuple[int, int], HighlightRegion] = {}
        
        # Очередь изменений
        self.change_queue: deque = deque(maxlen=1000)
        self.pending_regions: Set[Tuple[int, int]] = set()
        
        # Состояние подсветки
        self.lines: List[str] = []
        self.total_lines = 0
        self.last_full_parse = 0.0
        
        # Асинхронная обработка
        self.processing_thread: Optional[threading.Thread] = None
        self.should_stop = False
        self.processing_lock = threading.Lock()
        
        # Статистика
        self.stats = {
            'total_changes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'full_reparses': 0,
            'incremental_updates': 0,
            'processing_time': 0.0
        }
        
        # Обработчики событий
        self.on_highlight_updated: Optional[callable] = None
        self.on_region_updated: Optional[callable] = None
    
    def set_text(self, text: str):
        """Установить весь текст"""
        self.lines = text.split('\n')
        self.total_lines = len(self.lines)
        
        # Очистка кэшей
        self.line_cache.clear()
        self.region_cache.clear()
        self.pending_regions.clear()
        
        # Полная перепарсировка в фоне
        self._schedule_full_reparse()
    
    def apply_change(self, change: TextChange):
        """Применить изменение к тексту"""
        self.change_queue.append(change)
        self.stats['total_changes'] += 1
        
        # Определение затронутых строк
        affected_lines = self._get_affected_lines(change)
        
        # Применение изменения к lines
        self._apply_change_to_lines(change)
        
        # Инвалидация кэша для затронутых строк
        self._invalidate_cache(affected_lines)
        
        # Планирование обновления
        self._schedule_incremental_update(affected_lines)
    
    def _get_affected_lines(self, change: TextChange) -> Set[int]:
        """Получить список затронутых строк"""
        affected = set()
        
        # Прямо затронутые строки
        for line in range(change.start_line, change.end_line + 1):
            affected.add(line)
        
        # Зависимые строки (например, многострочные комментарии)
        for line in affected.copy():
            if line in self.line_cache:
                affected.update(self.line_cache[line].dependencies)
        
        # Добавляем буферные строки для контекста
        min_line = max(0, min(affected) - 2)
        max_line = min(self.total_lines - 1, max(affected) + 2)
        
        for line in range(min_line, max_line + 1):
            affected.add(line)
        
        return affected
    
    def _apply_change_to_lines(self, change: TextChange):
        """Применить изменение к массиву строк"""
        if change.type == ChangeType.INSERT:
            # Вставка текста
            if change.start_line == change.end_line:
                # Вставка в одну строку
                line = self.lines[change.start_line]
                new_line = (line[:change.start_column] + 
                           change.new_text + 
                           line[change.start_column:])
                self.lines[change.start_line] = new_line
            else:
                # Многострочная вставка
                new_lines = change.new_text.split('\n')
                
                # Обновление первой строки
                first_line = self.lines[change.start_line]
                self.lines[change.start_line] = first_line[:change.start_column] + new_lines[0]
                
                # Вставка новых строк
                for i, new_line in enumerate(new_lines[1:], 1):
                    self.lines.insert(change.start_line + i, new_line)
                
                # Обновление последней строки
                if len(new_lines) > 1:
                    last_line = self.lines[change.start_line + len(new_lines) - 1]
                    original_end = first_line[change.start_column:]
                    self.lines[change.start_line + len(new_lines) - 1] = last_line + original_end
        
        elif change.type == ChangeType.DELETE:
            # Удаление текста
            if change.start_line == change.end_line:
                # Удаление в одной строке
                line = self.lines[change.start_line]
                new_line = line[:change.start_column] + line[change.end_column:]
                self.lines[change.start_line] = new_line
            else:
                # Многострочное удаление
                first_line = self.lines[change.start_line][:change.start_column]
                last_line = self.lines[change.end_line][change.end_column:]
                
                # Удаление строк
                del self.lines[change.start_line:change.end_line + 1]
                
                # Объединение оставшихся частей
                self.lines.insert(change.start_line, first_line + last_line)
        
        elif change.type == ChangeType.REPLACE:
            # Замена текста
            delete_change = TextChange(
                type=ChangeType.DELETE,
                start_line=change.start_line,
                start_column=change.start_column,
                end_line=change.end_line,
                end_column=change.end_column,
                old_text=change.old_text,
                new_text=""
            )
            
            insert_change = TextChange(
                type=ChangeType.INSERT,
                start_line=change.start_line,
                start_column=change.start_column,
                end_line=change.start_line,
                end_column=change.start_column,
                old_text="",
                new_text=change.new_text
            )
            
            self._apply_change_to_lines(delete_change)
            self._apply_change_to_lines(insert_change)
        
        # Обновление общего количества строк
        self.total_lines = len(self.lines)
    
    def _invalidate_cache(self, affected_lines: Set[int]):
        """Инвалидация кэша для затронутых строк"""
        # Инвалидация кэша строк
        for line in affected_lines:
            if line in self.line_cache:
                del self.line_cache[line]
        
        # Инвалидация кэша регионов
        regions_to_remove = []
        for (start, end) in self.region_cache:
            if any(line in range(start, end + 1) for line in affected_lines):
                regions_to_remove.append((start, end))
        
        for region in regions_to_remove:
            del self.region_cache[region]
    
    def _schedule_incremental_update(self, affected_lines: Set[int]):
        """Планирование инкрементального обновления"""
        # Группировка строк в регионы
        sorted_lines = sorted(affected_lines)
        
        regions = []
        current_start = sorted_lines[0]
        current_end = sorted_lines[0]
        
        for line in sorted_lines[1:]:
            if line <= current_end + self.chunk_size:
                current_end = line
            else:
                regions.append((current_start, current_end))
                current_start = line
                current_end = line
        
        regions.append((current_start, current_end))
        
        # Добавление регионов в очередь
        for start, end in regions:
            self.pending_regions.add((start, end))
        
        # Запуск обработки
        self._start_processing_thread()
    
    def _schedule_full_reparse(self):
        """Планирование полной перепарсировки"""
        # Разбиение на чанки
        for start in range(0, self.total_lines, self.chunk_size):
            end = min(start + self.chunk_size - 1, self.total_lines - 1)
            self.pending_regions.add((start, end))
        
        self._start_processing_thread()
    
    def _start_processing_thread(self):
        """Запуск потока обработки"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.should_stop = False
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def _processing_loop(self):
        """Цикл обработки в фоновом потоке"""
        while not self.should_stop and self.pending_regions:
            with self.processing_lock:
                if not self.pending_regions:
                    break
                
                # Получение следующего региона
                region = self.pending_regions.pop()
                start_line, end_line = region
                
                # Обработка региона
                start_time = time.perf_counter()
                
                try:
                    self._process_region(start_line, end_line)
                    self.stats['incremental_updates'] += 1
                except Exception as e:
                    print(f"Ошибка обработки региона {start_line}-{end_line}: {e}")
                
                processing_time = time.perf_counter() - start_time
                self.stats['processing_time'] += processing_time
                
                # Уведомление об обновлении
                if self.on_region_updated:
                    self.on_region_updated(start_line, end_line)
            
            # Небольшая пауза для предотвращения блокировки UI
            time.sleep(0.001)
    
    def _process_region(self, start_line: int, end_line: int):
        """Обработка региона строк"""
        # Подготовка текста региона
        region_lines = []
        for line_num in range(start_line, min(end_line + 1, self.total_lines)):
            if line_num < len(self.lines):
                region_lines.append(self.lines[line_num])
            else:
                region_lines.append("")
        
        region_text = '\n'.join(region_lines)
        
        # Проверка кэша
        region_hash = hash(region_text)
        cache_key = (start_line, end_line)
        
        if cache_key in self.region_cache:
            cached_region = self.region_cache[cache_key]
            if not cached_region.is_dirty:
                self.stats['cache_hits'] += 1
                return cached_region.tokens
        
        self.stats['cache_misses'] += 1
        
        # Токенизация региона
        tokens = self.base_highlighter.tokenize(region_text)
        
        # Корректировка номеров строк
        for token in tokens:
            token.line += start_line
        
        # Кэширование по строкам
        for line_num in range(start_line, min(end_line + 1, self.total_lines)):
            line_tokens = [t for t in tokens if t.line == line_num]
            
            if line_num < len(self.lines):
                line_content = self.lines[line_num]
                
                parsed_line = ParsedLine(
                    line_number=line_num,
                    content=line_content,
                    tokens=line_tokens,
                    hash_value=hash(line_content)
                )
                
                self.line_cache[line_num] = parsed_line
        
        # Кэширование региона
        region = HighlightRegion(
            start_line=start_line,
            end_line=end_line,
            tokens=tokens,
            is_dirty=False
        )
        
        self.region_cache[cache_key] = region
        
        # Очистка старого кэша
        self._cleanup_cache()
        
        return tokens
    
    def _cleanup_cache(self):
        """Очистка старого кэша"""
        # Ограничение размера кэша строк
        if len(self.line_cache) > self.cache_size:
            # Удаление самых старых записей
            sorted_items = sorted(
                self.line_cache.items(),
                key=lambda x: x[1].last_parsed
            )
            
            items_to_remove = len(self.line_cache) - self.cache_size + 100
            for i in range(items_to_remove):
                line_num = sorted_items[i][0]
                del self.line_cache[line_num]
        
        # Ограничение размера кэша регионов
        if len(self.region_cache) > self.cache_size // 10:
            sorted_regions = sorted(
                self.region_cache.items(),
                key=lambda x: x[1].last_updated
            )
            
            regions_to_remove = len(self.region_cache) - self.cache_size // 10 + 10
            for i in range(regions_to_remove):
                region_key = sorted_regions[i][0]
                del self.region_cache[region_key]
    
    def get_line_tokens(self, line_number: int) -> List[HighlightToken]:
        """Получить токены для строки"""
        if line_number in self.line_cache:
            self.stats['cache_hits'] += 1
            return self.line_cache[line_number].tokens
        
        # Запрос региона, содержащего эту строку
        region_start = (line_number // self.chunk_size) * self.chunk_size
        region_end = min(region_start + self.chunk_size - 1, self.total_lines - 1)
        
        self._process_region(region_start, region_end)
        
        if line_number in self.line_cache:
            return self.line_cache[line_number].tokens
        
        return []
    
    def get_region_tokens(self, start_line: int, end_line: int) -> List[HighlightToken]:
        """Получить токены для региона"""
        all_tokens = []
        
        # Разбиение на чанки при необходимости
        current_start = start_line
        
        while current_start <= end_line:
            chunk_end = min(current_start + self.chunk_size - 1, end_line)
            
            tokens = self._process_region(current_start, chunk_end)
            all_tokens.extend(tokens)
            
            current_start = chunk_end + 1
        
        return all_tokens
    
    def get_visible_tokens(self, first_visible_line: int, last_visible_line: int) -> List[HighlightToken]:
        """Получить токены для видимых строк"""
        return self.get_region_tokens(first_visible_line, last_visible_line)
    
    def stop_processing(self):
        """Остановить фоновую обработку"""
        self.should_stop = True
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
    
    def get_statistics(self) -> Dict:
        """Получить статистику работы"""
        return {
            **self.stats,
            'cache_size': len(self.line_cache),
            'region_cache_size': len(self.region_cache),
            'pending_regions': len(self.pending_regions),
            'total_lines': self.total_lines,
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            ) * 100
        }
    
    def clear_cache(self):
        """Очистить весь кэш"""
        with self.processing_lock:
            self.line_cache.clear()
            self.region_cache.clear()
            self.pending_regions.clear()


# Адаптер для интеграции с текстовыми виджетами
class TextWidgetHighlighter:
    """Адаптер для интеграции с текстовыми виджетами"""
    
    def __init__(self, text_widget, theme: Optional[HighlightTheme] = None):
        self.text_widget = text_widget
        self.highlighter = IncrementalHighlighter(theme)
        
        # Настройка обработчиков
        self.highlighter.on_region_updated = self._on_region_updated
        
        # Привязка событий виджета
        self.text_widget.bind('<KeyRelease>', self._on_text_changed)
        self.text_widget.bind('<Button-1>', self._on_text_changed)
        self.text_widget.bind('<Paste>', self._on_text_changed)
        
        # Настройка тегов для подсветки
        self._setup_tags()
    
    def _setup_tags(self):
        """Настройка тегов для подсветки"""
        for token_type, style in self.highlighter.base_highlighter.theme.styles.items():
            tag_name = f"token_{token_type.value}"
            
            self.text_widget.tag_configure(
                tag_name,
                foreground=style.color,
                background=style.background,
                font=('Consolas', 12, 
                      ('bold' if style.bold else 'normal') + 
                      (' italic' if style.italic else ''))
            )
    
    def _on_text_changed(self, event=None):
        """Обработка изменения текста"""
        # Получение текущего содержимого
        content = self.text_widget.get('1.0', 'end-1c')
        
        # Обновление highlighter'а
        self.highlighter.set_text(content)
        
        # Запуск подсветки видимой области
        self._highlight_visible_area()
    
    def _on_region_updated(self, start_line: int, end_line: int):
        """Обработка обновления региона"""
        # Применение подсветки к обновленному региону
        self._apply_highlighting(start_line, end_line)
    
    def _highlight_visible_area(self):
        """Подсветка видимой области"""
        try:
            # Получение видимой области
            first_line = int(self.text_widget.index('@0,0').split('.')[0]) - 1
            last_line = int(self.text_widget.index('@0,%d' % self.text_widget.winfo_height()).split('.')[0]) - 1
            
            # Добавление буфера
            first_line = max(0, first_line - 10)
            last_line = min(self.highlighter.total_lines - 1, last_line + 10)
            
            # Запуск подсветки
            tokens = self.highlighter.get_visible_tokens(first_line, last_line)
            self._apply_highlighting(first_line, last_line)
            
        except Exception as e:
            print(f"Ошибка подсветки видимой области: {e}")
    
    def _apply_highlighting(self, start_line: int, end_line: int):
        """Применение подсветки к региону"""
        try:
            # Очистка старых тегов в регионе
            start_pos = f"{start_line + 1}.0"
            end_pos = f"{end_line + 2}.0"
            
            for tag in self.text_widget.tag_names():
                if tag.startswith('token_'):
                    self.text_widget.tag_remove(tag, start_pos, end_pos)
            
            # Получение токенов для региона
            tokens = self.highlighter.get_region_tokens(start_line, end_line)
            
            # Применение подсветки
            for token in tokens:
                if token.type.value in ['whitespace', 'newline']:
                    continue
                
                tag_name = f"token_{token.type.value}"
                
                # Вычисление позиции в тексте
                token_start = f"{token.line + 1}.{token.column - 1}"
                token_end = f"{token.line + 1}.{token.column - 1 + len(token.value)}"
                
                self.text_widget.tag_add(tag_name, token_start, token_end)
                
        except Exception as e:
            print(f"Ошибка применения подсветки: {e}")


if __name__ == "__main__":
    # Тестирование инкрементальной подсветки
    from .syntax_highlighter import THEMES
    
    # Создание highlighter'а
    highlighter = IncrementalHighlighter(THEMES['dark'])
    
    # Тестовый код
    test_code = '''
    def factorial(n):
        if n <= 1:
            return 1
        else:
            return n * factorial(n - 1)
    
    neuron test_neuron {
        activation: "relu"
        threshold: 0.5
    }
    
    signal input_signal {
        value: 0.8
    }
    '''
    
    print("⚡ Тестирование инкрементальной подсветки...")
    
    # Установка текста
    highlighter.set_text(test_code)
    
    print(f"📊 Исходная статистика: {highlighter.get_statistics()}")
    
    # Ожидание завершения обработки
    time.sleep(0.5)
    
    # Получение токенов для первых строк
    tokens = highlighter.get_line_tokens(0)
    print(f"🎨 Токены для строки 0: {len(tokens)}")
    
    # Симуляция изменений
    changes = [
        TextChange(
            type=ChangeType.INSERT,
            start_line=1,
            start_column=0,
            end_line=1,
            end_column=0,
            old_text="",
            new_text="    # Новый комментарий\n"
        ),
        TextChange(
            type=ChangeType.REPLACE,
            start_line=2,
            start_column=8,
            end_line=2,
            end_column=9,
            old_text="n",
            new_text="number"
        )
    ]
    
    for change in changes:
        print(f"🔄 Применение изменения: {change.type.value}")
        highlighter.apply_change(change)
    
    # Ожидание обработки
    time.sleep(0.2)
    
    # Финальная статистика
    final_stats = highlighter.get_statistics()
    print(f"📈 Финальная статистика: {final_stats}")
    
    # Остановка обработки
    highlighter.stop_processing()
    
    print("✅ Тестирование инкрементальной подсветки завершено") 