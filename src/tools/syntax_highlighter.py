"""
Система подсветки синтаксиса для языка Anamorph

Поддерживает:
- Различные темы (светлая/темная)
- Экспорт в HTML, JSON, XML
- Настраиваемые стили
- Поддержка VS Code, Sublime Text, Vim
"""

import re
import json
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class TokenType(Enum):
    """Типы токенов для подсветки"""
    # Ключевые слова
    KEYWORD = "keyword"
    NEURAL_KEYWORD = "neural_keyword"
    
    # Идентификаторы
    IDENTIFIER = "identifier"
    FUNCTION_NAME = "function_name"
    CLASS_NAME = "class_name"
    VARIABLE = "variable"
    
    # Литералы
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    
    # Операторы
    OPERATOR = "operator"
    ASSIGNMENT = "assignment"
    COMPARISON = "comparison"
    LOGICAL = "logical"
    
    # Разделители
    DELIMITER = "delimiter"
    BRACKET = "bracket"
    PARENTHESIS = "parenthesis"
    
    # Комментарии
    COMMENT = "comment"
    BLOCK_COMMENT = "block_comment"
    DOC_COMMENT = "doc_comment"
    
    # Нейронные элементы
    NEURON = "neuron"
    SYNAPSE = "synapse" 
    SIGNAL = "signal"
    PULSE = "pulse"
    NETWORK = "network"
    
    # Типы данных
    TYPE = "type"
    NEURAL_TYPE = "neural_type"
    
    # Специальные
    WHITESPACE = "whitespace"
    NEWLINE = "newline"
    ERROR = "error"


@dataclass
class StyleConfig:
    """Конфигурация стиля для токена"""
    color: str = "#000000"
    background: Optional[str] = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    
    def to_css(self) -> str:
        """Преобразование в CSS стиль"""
        styles = [f"color: {self.color}"]
        
        if self.background:
            styles.append(f"background-color: {self.background}")
        if self.bold:
            styles.append("font-weight: bold")
        if self.italic:
            styles.append("font-style: italic")
        if self.underline:
            styles.append("text-decoration: underline")
        if self.strikethrough:
            styles.append("text-decoration: line-through")
            
        return "; ".join(styles)


@dataclass
class HighlightToken:
    """Токен для подсветки"""
    type: TokenType
    value: str
    start: int
    end: int
    line: int
    column: int
    style: Optional[StyleConfig] = None


class HighlightTheme:
    """Тема подсветки синтаксиса"""
    
    def __init__(self, name: str, is_dark: bool = False):
        self.name = name
        self.is_dark = is_dark
        self.styles: Dict[TokenType, StyleConfig] = {}
        self._init_default_styles()
    
    def _init_default_styles(self):
        """Инициализация стилей по умолчанию"""
        if self.is_dark:
            self._init_dark_theme()
        else:
            self._init_light_theme()
    
    def _init_light_theme(self):
        """Светлая тема"""
        self.styles = {
            # Ключевые слова
            TokenType.KEYWORD: StyleConfig("#0000FF", bold=True),
            TokenType.NEURAL_KEYWORD: StyleConfig("#8B00FF", bold=True),
            
            # Идентификаторы
            TokenType.IDENTIFIER: StyleConfig("#000000"),
            TokenType.FUNCTION_NAME: StyleConfig("#795E26", bold=True),
            TokenType.CLASS_NAME: StyleConfig("#267F99", bold=True),
            TokenType.VARIABLE: StyleConfig("#001080"),
            
            # Литералы
            TokenType.NUMBER: StyleConfig("#098658"),
            TokenType.STRING: StyleConfig("#A31515"),
            TokenType.BOOLEAN: StyleConfig("#0000FF"),
            
            # Операторы
            TokenType.OPERATOR: StyleConfig("#000000"),
            TokenType.ASSIGNMENT: StyleConfig("#000000"),
            TokenType.COMPARISON: StyleConfig("#000000"),
            TokenType.LOGICAL: StyleConfig("#0000FF"),
            
            # Разделители
            TokenType.DELIMITER: StyleConfig("#000000"),
            TokenType.BRACKET: StyleConfig("#000000"),
            TokenType.PARENTHESIS: StyleConfig("#000000"),
            
            # Комментарии
            TokenType.COMMENT: StyleConfig("#008000", italic=True),
            TokenType.BLOCK_COMMENT: StyleConfig("#008000", italic=True),
            TokenType.DOC_COMMENT: StyleConfig("#008000", italic=True, bold=True),
            
            # Нейронные элементы
            TokenType.NEURON: StyleConfig("#FF4500", bold=True),
            TokenType.SYNAPSE: StyleConfig("#FF6347", bold=True),
            TokenType.SIGNAL: StyleConfig("#FFA500", bold=True),
            TokenType.PULSE: StyleConfig("#FFD700", bold=True),
            TokenType.NETWORK: StyleConfig("#DC143C", bold=True),
            
            # Типы данных
            TokenType.TYPE: StyleConfig("#267F99"),
            TokenType.NEURAL_TYPE: StyleConfig("#8B00FF"),
            
            # Специальные
            TokenType.ERROR: StyleConfig("#FF0000", background="#FFDDDD", bold=True)
        }
    
    def _init_dark_theme(self):
        """Темная тема"""
        self.styles = {
            # Ключевые слова
            TokenType.KEYWORD: StyleConfig("#569CD6", bold=True),
            TokenType.NEURAL_KEYWORD: StyleConfig("#C586C0", bold=True),
            
            # Идентификаторы
            TokenType.IDENTIFIER: StyleConfig("#D4D4D4"),
            TokenType.FUNCTION_NAME: StyleConfig("#DCDCAA", bold=True),
            TokenType.CLASS_NAME: StyleConfig("#4EC9B0", bold=True),
            TokenType.VARIABLE: StyleConfig("#9CDCFE"),
            
            # Литералы
            TokenType.NUMBER: StyleConfig("#B5CEA8"),
            TokenType.STRING: StyleConfig("#CE9178"),
            TokenType.BOOLEAN: StyleConfig("#569CD6"),
            
            # Операторы
            TokenType.OPERATOR: StyleConfig("#D4D4D4"),
            TokenType.ASSIGNMENT: StyleConfig("#D4D4D4"),
            TokenType.COMPARISON: StyleConfig("#D4D4D4"),
            TokenType.LOGICAL: StyleConfig("#569CD6"),
            
            # Разделители
            TokenType.DELIMITER: StyleConfig("#D4D4D4"),
            TokenType.BRACKET: StyleConfig("#D4D4D4"),
            TokenType.PARENTHESIS: StyleConfig("#D4D4D4"),
            
            # Комментарии
            TokenType.COMMENT: StyleConfig("#6A9955", italic=True),
            TokenType.BLOCK_COMMENT: StyleConfig("#6A9955", italic=True),
            TokenType.DOC_COMMENT: StyleConfig("#6A9955", italic=True, bold=True),
            
            # Нейронные элементы
            TokenType.NEURON: StyleConfig("#FF7F50", bold=True),
            TokenType.SYNAPSE: StyleConfig("#FF6B6B", bold=True),
            TokenType.SIGNAL: StyleConfig("#FFD93D", bold=True),
            TokenType.PULSE: StyleConfig("#6BCF7F", bold=True),
            TokenType.NETWORK: StyleConfig("#FF8C94", bold=True),
            
            # Типы данных
            TokenType.TYPE: StyleConfig("#4EC9B0"),
            TokenType.NEURAL_TYPE: StyleConfig("#C586C0"),
            
            # Специальные
            TokenType.ERROR: StyleConfig("#FF0000", background="#441111", bold=True)
        }
    
    def get_style(self, token_type: TokenType) -> StyleConfig:
        """Получить стиль для типа токена"""
        return self.styles.get(token_type, StyleConfig())
    
    def set_style(self, token_type: TokenType, style: StyleConfig):
        """Установить стиль для типа токена"""
        self.styles[token_type] = style


class AnamorphSyntaxHighlighter:
    """Подсветка синтаксиса для языка Anamorph"""
    
    def __init__(self, theme: Optional[HighlightTheme] = None):
        self.theme = theme or HighlightTheme("default")
        self.tokens: List[HighlightToken] = []
        self._init_patterns()
    
    def _init_patterns(self):
        """Инициализация паттернов для распознавания токенов"""
        self.patterns = [
            # Комментарии (должны быть первыми)
            (r'//.*$', TokenType.COMMENT),
            (r'/\*[\s\S]*?\*/', TokenType.BLOCK_COMMENT),
            (r'///.*$', TokenType.DOC_COMMENT),
            
            # Строки
            (r'"([^"\\]|\\.)*"', TokenType.STRING),
            (r"'([^'\\]|\\.)*'", TokenType.STRING),
            
            # Числа
            (r'\b\d+\.?\d*([eE][+-]?\d+)?\b', TokenType.NUMBER),
            
            # Нейронные ключевые слова
            (r'\b(neuron|synapse|signal|pulse|network|layer|activation|weight|bias|gradient|backprop|forward|train|learn|predict|classify)\b', TokenType.NEURAL_KEYWORD),
            
            # Обычные ключевые слова
            (r'\b(if|else|elif|while|for|in|break|continue|return|def|class|import|from|as|try|except|finally|with|lambda|yield|pass|del|global|nonlocal|assert|raise|and|or|not|is|None|True|False)\b', TokenType.KEYWORD),
            
            # Нейронные типы
            (r'\b(Neuron|Synapse|Signal|Pulse|NeuralNetwork|Layer|Activation|Weight|Bias|Gradient)\b', TokenType.NEURAL_TYPE),
            
            # Обычные типы
            (r'\b(int|float|str|bool|list|dict|tuple|set|type|object)\b', TokenType.TYPE),
            
            # Булевы значения
            (r'\b(True|False|None)\b', TokenType.BOOLEAN),
            
            # Идентификаторы (функции, переменные, классы)  
            (r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', TokenType.IDENTIFIER),
            
            # Операторы
            (r'[+\-*//%=<>!&|^~]', TokenType.OPERATOR),
            (r'==|!=|<=|>=|<<|>>|//|\*\*', TokenType.COMPARISON),
            (r'and|or|not', TokenType.LOGICAL),
            (r'=', TokenType.ASSIGNMENT),
            
            # Разделители
            (r'[,;:]', TokenType.DELIMITER),
            (r'[(){}\[\]]', TokenType.BRACKET),
            
            # Пробелы и переносы
            (r'\n', TokenType.NEWLINE),
            (r'\s+', TokenType.WHITESPACE),
        ]
        
        # Компиляция регулярных выражений
        self.compiled_patterns = [(re.compile(pattern, re.MULTILINE), token_type) 
                                 for pattern, token_type in self.patterns]
    
    def tokenize(self, code: str) -> List[HighlightToken]:
        """Токенизация кода"""
        self.tokens = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            self._tokenize_line(line, line_num)
        
        return self.tokens
    
    def _tokenize_line(self, line: str, line_num: int):
        """Токенизация одной строки"""
        pos = 0
        
        while pos < len(line):
            matched = False
            
            for pattern, token_type in self.compiled_patterns:
                match = pattern.match(line, pos)
                if match:
                    value = match.group(0)
                    
                    # Пропускаем пробелы (если не нужны)
                    if token_type != TokenType.WHITESPACE:
                        token = HighlightToken(
                            type=token_type,
                            value=value,
                            start=pos,
                            end=match.end(),
                            line=line_num,
                            column=pos + 1,
                            style=self.theme.get_style(token_type)
                        )
                        self.tokens.append(token)
                    
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                # Неизвестный символ
                token = HighlightToken(
                    type=TokenType.ERROR,
                    value=line[pos],
                    start=pos,
                    end=pos + 1,
                    line=line_num,
                    column=pos + 1,
                    style=self.theme.get_style(TokenType.ERROR)
                )
                self.tokens.append(token)
                pos += 1
    
    def highlight_to_html(self, code: str, include_line_numbers: bool = True) -> str:
        """Подсветка кода в HTML формате"""
        tokens = self.tokenize(code)
        html_parts = []
        
        if include_line_numbers:
            html_parts.append('<div class="code-container">')
            html_parts.append('<div class="line-numbers">')
            
            # Номера строк
            max_line = max(token.line for token in tokens) if tokens else 1
            for i in range(1, max_line + 1):
                html_parts.append(f'<span class="line-number">{i}</span>')
            
            html_parts.append('</div>')
            html_parts.append('<div class="code-content">')
        
        html_parts.append('<pre><code>')
        
        current_line = 1
        for token in tokens:
            if token.line > current_line:
                # Добавляем переносы строк
                html_parts.extend(['<br>'] * (token.line - current_line))
                current_line = token.line
            
            if token.type != TokenType.WHITESPACE:
                css_class = f"token-{token.type.value}"
                style = token.style.to_css() if token.style else ""
                html_parts.append(f'<span class="{css_class}" style="{style}">{self._escape_html(token.value)}</span>')
            else:
                html_parts.append(token.value)
        
        html_parts.append('</code></pre>')
        
        if include_line_numbers:
            html_parts.append('</div>')
            html_parts.append('</div>')
        
        return ''.join(html_parts)
    
    def _escape_html(self, text: str) -> str:
        """Экранирование HTML символов"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))
    
    def generate_css(self) -> str:
        """Генерация CSS стилей для подсветки"""
        css_parts = [
            '.code-container { display: flex; font-family: "Courier New", monospace; }',
            '.line-numbers { padding: 10px; background: #f0f0f0; border-right: 1px solid #ccc; }',
            '.line-number { display: block; color: #666; }',
            '.code-content { padding: 10px; flex: 1; }',
            'pre { margin: 0; }',
            'code { font-family: inherit; }'
        ]
        
        # Стили для токенов
        for token_type, style in self.theme.styles.items():
            css_class = f".token-{token_type.value}"
            css_rule = f"{css_class} {{ {style.to_css()} }}"
            css_parts.append(css_rule)
        
        return '\n'.join(css_parts)
    
    def export_to_json(self, code: str) -> str:
        """Экспорт в JSON формат"""
        tokens = self.tokenize(code)
        
        token_data = []
        for token in tokens:
            token_data.append({
                'type': token.type.value,
                'value': token.value,
                'start': token.start,
                'end': token.end,
                'line': token.line,
                'column': token.column,
                'style': {
                    'color': token.style.color if token.style else '#000000',
                    'bold': token.style.bold if token.style else False,
                    'italic': token.style.italic if token.style else False
                }
            })
        
        return json.dumps({
            'theme': self.theme.name,
            'tokens': token_data
        }, indent=2, ensure_ascii=False)
    
    def export_vs_code_theme(self) -> Dict:
        """Экспорт темы для VS Code"""
        theme_data = {
            'name': f'Anamorph {self.theme.name}',
            'type': 'dark' if self.theme.is_dark else 'light',
            'colors': {
                'editor.background': '#1E1E1E' if self.theme.is_dark else '#FFFFFF',
                'editor.foreground': '#D4D4D4' if self.theme.is_dark else '#000000',
                'editorLineNumber.foreground': '#6E7681' if self.theme.is_dark else '#237893'
            },
            'tokenColors': []
        }
        
        # Маппинг токенов на TextMate scopes
        scope_mapping = {
            TokenType.KEYWORD: ['keyword.control'],
            TokenType.NEURAL_KEYWORD: ['keyword.other.neural', 'entity.name.type.neural'],
            TokenType.STRING: ['string.quoted'],
            TokenType.NUMBER: ['constant.numeric'],
            TokenType.COMMENT: ['comment.line'],
            TokenType.FUNCTION_NAME: ['entity.name.function'],
            TokenType.CLASS_NAME: ['entity.name.type.class'],
            TokenType.VARIABLE: ['variable'],
            TokenType.TYPE: ['entity.name.type'],
            TokenType.NEURAL_TYPE: ['entity.name.type.neural']
        }
        
        for token_type, style in self.theme.styles.items():
            if token_type in scope_mapping:
                token_color = {
                    'scope': scope_mapping[token_type],
                    'settings': {
                        'foreground': style.color
                    }
                }
                
                if style.bold:
                    token_color['settings']['fontStyle'] = 'bold'
                if style.italic:
                    token_color['settings']['fontStyle'] = 'italic'
                
                theme_data['tokenColors'].append(token_color)
        
        return theme_data


# Предопределенные темы
THEMES = {
    'light': HighlightTheme('Light', is_dark=False),
    'dark': HighlightTheme('Dark', is_dark=True),
    'vs_code_light': HighlightTheme('VS Code Light', is_dark=False),
    'vs_code_dark': HighlightTheme('VS Code Dark', is_dark=True)
}


def highlight_anamorph_code(code: str, theme: str = 'dark', format: str = 'html') -> str:
    """
    Быстрая функция для подсветки кода Anamorph
    
    Args:
        code: Исходный код
        theme: Тема ('light', 'dark', 'vs_code_light', 'vs_code_dark')
        format: Формат вывода ('html', 'json')
    
    Returns:
        Подсвеченный код в указанном формате
    """
    highlighter = AnamorphSyntaxHighlighter(THEMES.get(theme, THEMES['dark']))
    
    if format == 'html':
        return highlighter.highlight_to_html(code)
    elif format == 'json':
        return highlighter.export_to_json(code)
    else:
        raise ValueError(f"Неподдерживаемый формат: {format}")


if __name__ == "__main__":
    # Тестовый код Anamorph
    test_code = '''
    // Простая нейронная сеть
    neuron input_layer {
        activation: "relu"
        size: 784
    }
    
    synapse hidden_connection {
        from: input_layer
        to: hidden_layer
        weight: 0.5
        bias: 0.1
    }
    
    signal data_flow {
        type: "forward"
        data: [1, 2, 3, 4, 5]
    }
    
    def train_network(epochs=100):
        for epoch in range(epochs):
            # Прямое распространение
            output = network.forward(input_data)
            
            # Вычисление ошибки
            loss = calculate_loss(output, target)
            
            # Обратное распространение
            network.backward(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    # Создание нейронной сети
    network = NeuralNetwork([
        input_layer,
        hidden_layer,
        output_layer
    ])
    
    # Обучение
    train_network(epochs=1000)
    '''
    
    # Тестирование подсветки
    highlighter = AnamorphSyntaxHighlighter(THEMES['dark'])
    
    # HTML версия
    html_result = highlighter.highlight_to_html(test_code)
    print("HTML подсветка сгенерирована")
    
    # CSS стили
    css_styles = highlighter.generate_css()
    print("CSS стили сгенерированы")
    
    # JSON экспорт
    json_result = highlighter.export_to_json(test_code)
    print("JSON экспорт выполнен")
    
    # VS Code тема
    vs_code_theme = highlighter.export_vs_code_theme()
    print("VS Code тема сгенерирована") 