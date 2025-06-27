"""
AnamorphX PyTorch Code Generator
Генерация PyTorch кода из конфигурации нейронных сетей
"""

from typing import List, Dict, Any, Optional
from .network_parser import NetworkConfig, NeuronConfig


class PyTorchGenerator:
    """Генератор PyTorch кода из конфигурации сети"""
    
    def __init__(self):
        self.activation_mapping = {
            'relu': 'F.relu',
            'sigmoid': 'torch.sigmoid',
            'tanh': 'torch.tanh',
            'softmax': 'F.softmax',
            'linear': '',  # No activation
            'leaky_relu': 'F.leaky_relu',
            'gelu': 'F.gelu',
            'swish': 'F.silu',  # SiLU is Swish in PyTorch
            'mish': 'F.mish',
            'elu': 'F.elu',
            'selu': 'F.selu'
        }
        
        self.optimizer_mapping = {
            'adam': 'torch.optim.Adam',
            'sgd': 'torch.optim.SGD',
            'adamw': 'torch.optim.AdamW',
            'rmsprop': 'torch.optim.RMSprop',
            'adagrad': 'torch.optim.Adagrad',
            'adadelta': 'torch.optim.Adadelta',
            'adamax': 'torch.optim.Adamax'
        }
        
        self.loss_mapping = {
            'mse': 'nn.MSELoss',
            'categorical_crossentropy': 'nn.CrossEntropyLoss',
            'binary_crossentropy': 'nn.BCEWithLogitsLoss',
            'sparse_categorical_crossentropy': 'nn.CrossEntropyLoss',
            'huber': 'nn.HuberLoss',
            'mae': 'nn.L1Loss',
            'focal': 'FocalLoss',  # Custom implementation needed
            'dice': 'DiceLoss',    # Custom implementation needed
            'triplet': 'nn.TripletMarginLoss'
        }
    
    def generate_model_class(self, network: NetworkConfig) -> str:
        """Генерация класса PyTorch модели"""
        class_name = network.name
        
        # Импорты
        imports = self._generate_imports(network)
        
        # Вспомогательные классы
        helper_classes = self._generate_helper_classes(network)
        
        # Класс модели
        class_def = f"""
class {class_name}(nn.Module):
    def __init__(self, input_size=None, input_channels=3, num_classes=10):
        super({class_name}, self).__init__()
        
        # Сохраняем параметры
        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Определение слоев
{self._generate_layer_definitions(network.neurons)}
        
        # Dropout слои
{self._generate_dropout_layers(network.neurons)}
        
        # Batch Normalization слои
{self._generate_batch_norm_layers(network.neurons)}
        
        # Positional Encoding для Transformer
{self._generate_positional_encoding(network.neurons)}
    
    def forward(self, x):
{self._generate_forward_method(network.neurons)}
        return x
"""
        
        return imports + helper_classes + class_def
    
    def _generate_imports(self, network: NetworkConfig) -> str:
        """Генерация импортов"""
        base_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
        
        # Дополнительные импорты для Transformer
        has_transformer = any(n.layer_type == 'transformer' for n in network.neurons)
        if has_transformer:
            base_imports += "from torch.nn import TransformerEncoder, TransformerEncoderLayer\n"
        
        # Импорты для ResNet
        has_resnet = any(getattr(n, 'skip_connection', False) for n in network.neurons)
        if has_resnet:
            base_imports += "from torchvision.models.resnet import BasicBlock, Bottleneck\n"
        
        return base_imports + "\n"
    
    def _generate_helper_classes(self, network: NetworkConfig) -> str:
        """Генерация вспомогательных классов"""
        helper_classes = []
        
        # Positional Encoding для Transformer
        has_transformer = any(n.layer_type == 'transformer' for n in network.neurons)
        if has_transformer:
            helper_classes.append("""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
""")
        
        # ResNet Block
        has_resnet = any(getattr(n, 'skip_connection', False) for n in network.neurons)
        if has_resnet:
            helper_classes.append("""
class ResNetBlock(nn.Module):
    \"\"\"Базовый ResNet блок\"\"\"
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNetBottleneckBlock(nn.Module):
    \"\"\"Bottleneck ResNet блок для ResNet-50/101/152\"\"\"
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class PreActResNetBlock(nn.Module):
    \"\"\"Pre-activation ResNet блок (современная версия)\"\"\"
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PreActResNetBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = F.relu(out)
        
        if self.downsample is not None:
            identity = self.downsample(out)
        
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        out += identity
        
        return out


class DepthwiseSeparableConv(nn.Module):
    \"\"\"Depthwise Separable Convolution для эффективных CNN\"\"\"
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x
""")
        
        # Attention механизм
        has_attention = any(n.layer_type == 'attention' for n in network.neurons)
        if has_attention:
            helper_classes.append("""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Attention calculation
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_dim ** (1/2)), dim=3)
        attention = self.dropout(attention)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out
""")
        
        # RNN/LSTM helper classes
        helper_classes.append(self._generate_rnn_helper_classes(network))
        
        # Vision Transformer (ViT) helper classes
        helper_classes.append(self._generate_vit_helper_classes(network))
        
        return "\n".join(helper_classes)
    
    def _generate_rnn_helper_classes(self, network: NetworkConfig) -> str:
        """Генерация RNN/LSTM вспомогательных классов"""
        helper_classes = []
        
        # Проверяем наличие RNN слоев
        has_lstm = any(n.layer_type == 'lstm' for n in network.neurons)
        has_gru = any(n.layer_type == 'gru' for n in network.neurons)
        has_rnn = any(n.layer_type == 'rnn' for n in network.neurons)
        
        if has_lstm or has_gru or has_rnn:
            helper_classes.append("""
class BidirectionalLSTM(nn.Module):
    \"\"\"Bidirectional LSTM с конфигурируемыми параметрами\"\"\"
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, batch_first=True):
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=batch_first
        )
        
        # Линейный слой для объединения направлений
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size) if batch_first=True
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # lstm_out: (batch, seq_len, hidden_size * 2)
        # Применяем линейный слой для объединения направлений
        output = self.fc(lstm_out)
        
        return output, (h_n, c_n)


class EnhancedGRU(nn.Module):
    \"\"\"Enhanced GRU с attention механизмом\"\"\"
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, 
                 bidirectional=False, attention=False, batch_first=True):
        super(EnhancedGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        self.batch_first = batch_first
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Attention механизм
        if attention:
            attention_hidden = hidden_size * 2 if bidirectional else hidden_size
            self.attention_layer = nn.Linear(attention_hidden, 1)
            self.softmax = nn.Softmax(dim=1)
        
        # Выходной слой
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.output_projection = nn.Linear(output_size, hidden_size)
        
    def forward(self, x, hidden=None):
        # GRU forward pass
        gru_out, h_n = self.gru(x, hidden)
        
        if self.attention:
            # Применяем attention
            attention_weights = self.attention_layer(gru_out)
            attention_weights = self.softmax(attention_weights)
            
            # Взвешенная сумма
            context = torch.sum(gru_out * attention_weights, dim=1, keepdim=True)
            output = self.output_projection(context)
        else:
            # Берем последний выход
            if self.batch_first:
                output = gru_out[:, -1, :]  # (batch, hidden_size)
            else:
                output = gru_out[-1, :, :]  # (seq_len, batch, hidden_size)
            
            output = self.output_projection(output)
        
        return output, h_n


class StackedLSTM(nn.Module):
    \"\"\"Стек LSTM слоев с residual connections\"\"\"
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2, 
                 residual=True, batch_first=True):
        super(StackedLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.residual = residual
        self.batch_first = batch_first
        
        # Создаем слои LSTM
        self.lstm_layers = nn.ModuleList()
        
        # Первый слой
        self.lstm_layers.append(
            nn.LSTM(input_size, hidden_size, 1, batch_first=batch_first)
        )
        
        # Остальные слои
        for i in range(1, num_layers):
            self.lstm_layers.append(
                nn.LSTM(hidden_size, hidden_size, 1, batch_first=batch_first)
            )
        
        # Dropout между слоями
        self.dropout = nn.Dropout(dropout)
        
        # Проекционные слои для residual connections
        if residual and num_layers > 1:
            self.projection_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)
            ])
        
    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        
        outputs = []
        current_input = x
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            # LSTM forward pass
            lstm_out, hidden_state = lstm_layer(current_input, hidden_states[i])
            
            # Residual connection (кроме первого слоя)
            if self.residual and i > 0:
                # Проекция для совпадения размерностей
                projected_input = self.projection_layers[i-1](current_input)
                if projected_input.shape == lstm_out.shape:
                    lstm_out = lstm_out + projected_input
            
            # Dropout (кроме последнего слоя)
            if i < self.num_layers - 1:
                lstm_out = self.dropout(lstm_out)
            
            outputs.append(hidden_state)
            current_input = lstm_out
        
        return current_input, outputs


class LSTMWithAttention(nn.Module):
    \"\"\"LSTM с механизмом внимания\"\"\"
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, 
                 attention_dim=None, batch_first=True):
        super(LSTMWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim or hidden_size // 2
        self.batch_first = batch_first
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=batch_first
        )
        
        # Attention механизм
        self.attention_W = nn.Linear(hidden_size, self.attention_dim)
        self.attention_u = nn.Linear(self.attention_dim, 1, bias=False)
        self.attention_softmax = nn.Softmax(dim=1)
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, hidden=None):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Attention mechanism
        # lstm_out: (batch, seq_len, hidden_size)
        attention_weights = self.attention_W(lstm_out)  # (batch, seq_len, attention_dim)
        attention_weights = torch.tanh(attention_weights)
        attention_weights = self.attention_u(attention_weights)  # (batch, seq_len, 1)
        attention_weights = self.attention_softmax(attention_weights)  # (batch, seq_len, 1)
        
        # Применяем attention weights
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size)
        
        # Финальный выход
        output = self.output_layer(context_vector)
        
        return output, (h_n, c_n)
""")
        
        return "\n".join(helper_classes)
    
    def _generate_vit_helper_classes(self, network: NetworkConfig) -> str:
        """Генерация Vision Transformer (ViT) вспомогательных классов"""
        helper_classes = []
        
        # Проверяем наличие ViT слоев
        has_vit = any(n.layer_type == 'vision_transformer' for n in network.neurons)
        has_patch_embed = any(n.layer_type == 'patch_embedding' for n in network.neurons)
        
        if has_vit or has_patch_embed:
            helper_classes.append("""
class PatchEmbedding(nn.Module):
    \"\"\"Patch Embedding для Vision Transformer\"\"\"
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Projection layer for patch embedding
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)        # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch_size, n_patches, embed_dim)
        return x


class VisionTransformerBlock(nn.Module):
    \"\"\"Блок Vision Transformer\"\"\"
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # MLP with residual connection
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        return x


class VisionTransformer(nn.Module):
    \"\"\"Complete Vision Transformer\"\"\"
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_layers=12, num_heads=12, mlp_ratio=4.0, num_classes=1000, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)  # (batch_size, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Classification head (use cls token)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits


class DeiTBlock(nn.Module):
    \"\"\"Data-efficient Image Transformer блок\"\"\"
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, dropout=0.1, 
                 drop_path=0.0):
        super(DeiTBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Drop path (stochastic depth)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection and drop path
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_output)
        
        # MLP with residual connection and drop path
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + self.drop_path(mlp_output)
        
        return x


class SwinTransformerBlock(nn.Module):
    \"\"\"Swin Transformer блок с shifted window attention\"\"\"
    def __init__(self, embed_dim=96, num_heads=3, window_size=7, shift_size=0,
                 mlp_ratio=4.0, dropout=0.1):
        super(SwinTransformerBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Window-based multi-head self attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Simplified Swin implementation (without actual window partitioning)
        # This is a basic version for demonstration
        
        # Layer norm and attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # Layer norm and MLP
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        return x
""")
        
        return "\n".join(helper_classes)
    
    def _generate_layer_definitions(self, neurons: List[NeuronConfig]) -> str:
        """Генерация определений слоев"""
        layer_defs = []
        
        for i, neuron in enumerate(neurons):
            layer_name = neuron.name.lower()
            
            if neuron.layer_type == 'dense':
                units = neuron.units or 128
                layer_def = f"        self.{layer_name} = nn.Linear(input_size, {units})"
            
            elif neuron.layer_type == 'conv':
                filters = neuron.filters or 32
                kernel_size = neuron.kernel_size or 3
                padding = neuron.padding or 1
                stride = neuron.stride or 1
                
                # Проверяем на depthwise separable convolution
                if getattr(neuron, 'depthwise_separable', False):
                    layer_def = f"        self.{layer_name} = DepthwiseSeparableConv(input_channels, {filters}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
                else:
                    layer_def = f"        self.{layer_name} = nn.Conv2d(input_channels, {filters}, kernel_size={kernel_size}, padding={padding}, stride={stride})"
            
            elif neuron.layer_type == 'pool':
                pool_size = neuron.pool_size or 2
                stride = neuron.stride or pool_size
                pool_type = getattr(neuron, 'pool_type', 'max')
                
                if pool_type == 'avg':
                    layer_def = f"        self.{layer_name} = nn.AvgPool2d(kernel_size={pool_size}, stride={stride})"
                elif pool_type == 'adaptive_avg':
                    output_size = getattr(neuron, 'output_size', 1)
                    layer_def = f"        self.{layer_name} = nn.AdaptiveAvgPool2d({output_size})"
                else:
                    layer_def = f"        self.{layer_name} = nn.MaxPool2d(kernel_size={pool_size}, stride={stride})"
            
            elif neuron.layer_type == 'lstm':
                units = neuron.units or 128
                num_layers = getattr(neuron, 'num_layers', 1)
                bidirectional = getattr(neuron, 'bidirectional', False)
                attention = getattr(neuron, 'attention', False)
                residual = getattr(neuron, 'residual', False)
                
                # Выбираем тип LSTM в зависимости от параметров
                if bidirectional and not attention and not residual:
                    layer_def = f"        self.{layer_name} = BidirectionalLSTM(input_size, {units}, num_layers={num_layers}, batch_first=True)"
                elif attention:
                    attention_dim = getattr(neuron, 'attention_dim', units // 2)
                    layer_def = f"        self.{layer_name} = LSTMWithAttention(input_size, {units}, num_layers={num_layers}, attention_dim={attention_dim}, batch_first=True)"
                elif residual and num_layers > 1:
                    dropout = neuron.dropout or 0.2
                    layer_def = f"        self.{layer_name} = StackedLSTM(input_size, {units}, num_layers={num_layers}, dropout={dropout}, residual=True, batch_first=True)"
                else:
                    # Стандартный LSTM
                    dropout = neuron.dropout or 0.0
                    layer_def = f"        self.{layer_name} = nn.LSTM(input_size, {units}, num_layers={num_layers}, dropout={dropout if num_layers > 1 else 0}, bidirectional={bidirectional}, batch_first=True)"
            
            elif neuron.layer_type == 'gru':
                units = neuron.units or 128
                num_layers = getattr(neuron, 'num_layers', 1)
                bidirectional = getattr(neuron, 'bidirectional', False)
                attention = getattr(neuron, 'attention', False)
                
                # Enhanced GRU с attention или стандартный
                if attention or bidirectional:
                    dropout = neuron.dropout or 0.0
                    layer_def = f"        self.{layer_name} = EnhancedGRU(input_size, {units}, num_layers={num_layers}, dropout={dropout}, bidirectional={bidirectional}, attention={attention}, batch_first=True)"
                else:
                    # Стандартный GRU
                    dropout = neuron.dropout or 0.0
                    layer_def = f"        self.{layer_name} = nn.GRU(input_size, {units}, num_layers={num_layers}, dropout={dropout if num_layers > 1 else 0}, batch_first=True)"
            
            elif neuron.layer_type == 'rnn':
                units = neuron.units or 128
                num_layers = getattr(neuron, 'num_layers', 1)
                nonlinearity = getattr(neuron, 'nonlinearity', 'tanh')  # 'tanh' или 'relu'
                bidirectional = getattr(neuron, 'bidirectional', False)
                dropout = neuron.dropout or 0.0
                
                layer_def = f"        self.{layer_name} = nn.RNN(input_size, {units}, num_layers={num_layers}, nonlinearity='{nonlinearity}', dropout={dropout if num_layers > 1 else 0}, bidirectional={bidirectional}, batch_first=True)"
            
            elif neuron.layer_type == 'transformer':
                embed_dim = neuron.embed_dim or 512
                num_heads = neuron.num_heads or 8
                ff_dim = neuron.ff_dim or 2048
                num_layers = getattr(neuron, 'num_layers', 6)
                dropout = neuron.dropout or 0.1
                
                layer_def = f"""        # Transformer layer {layer_name}
        encoder_layer = TransformerEncoderLayer(
            d_model={embed_dim},
            nhead={num_heads},
            dim_feedforward={ff_dim},
            dropout={dropout},
            batch_first=True
        )
        self.{layer_name} = TransformerEncoder(encoder_layer, num_layers={num_layers})"""
            
            elif neuron.layer_type == 'attention':
                embed_dim = neuron.embed_dim or 512
                num_heads = neuron.num_heads or 8
                dropout = neuron.dropout or 0.1
                
                # Выбираем тип attention
                attention_type = getattr(neuron, 'attention_type', 'multihead')
                if attention_type == 'self':
                    layer_def = f"        self.{layer_name} = MultiHeadSelfAttention({embed_dim}, {num_heads}, {dropout})"
                else:
                    layer_def = f"        self.{layer_name} = nn.MultiheadAttention({embed_dim}, {num_heads}, dropout={dropout}, batch_first=True)"
            
            elif neuron.layer_type == 'resnet_block':
                in_channels = getattr(neuron, 'in_channels', 64)
                out_channels = getattr(neuron, 'out_channels', 64)
                stride = neuron.stride or 1
                layer_def = f"        self.{layer_name} = ResNetBlock({in_channels}, {out_channels}, stride={stride})"
            
            elif neuron.layer_type == 'embedding':
                vocab_size = getattr(neuron, 'vocab_size', 10000)
                embed_dim = neuron.embed_dim or 512
                layer_def = f"        self.{layer_name} = nn.Embedding({vocab_size}, {embed_dim})"
            
            elif neuron.layer_type == 'layer_norm':
                normalized_shape = getattr(neuron, 'normalized_shape', 512)
                layer_def = f"        self.{layer_name} = nn.LayerNorm({normalized_shape})"
            
            elif neuron.layer_type == 'vision_transformer':
                img_size = getattr(neuron, 'img_size', 224)
                patch_size = getattr(neuron, 'patch_size', 16)
                in_channels = getattr(neuron, 'in_channels', 3)
                embed_dim = neuron.embed_dim or 768
                num_layers = getattr(neuron, 'num_layers', 12)
                num_heads = neuron.num_heads or 12
                mlp_ratio = getattr(neuron, 'mlp_ratio', 4.0)
                num_classes = getattr(neuron, 'num_classes', 1000)
                dropout = neuron.dropout or 0.1
                
                layer_def = f"        self.{layer_name} = VisionTransformer(img_size={img_size}, patch_size={patch_size}, in_channels={in_channels}, embed_dim={embed_dim}, num_layers={num_layers}, num_heads={num_heads}, mlp_ratio={mlp_ratio}, num_classes={num_classes}, dropout={dropout})"
            
            elif neuron.layer_type == 'patch_embedding':
                img_size = getattr(neuron, 'img_size', 224)
                patch_size = getattr(neuron, 'patch_size', 16)
                in_channels = getattr(neuron, 'in_channels', 3)
                embed_dim = neuron.embed_dim or 768
                
                layer_def = f"        self.{layer_name} = PatchEmbedding(img_size={img_size}, patch_size={patch_size}, in_channels={in_channels}, embed_dim={embed_dim})"
            
            elif neuron.layer_type == 'vit_block':
                embed_dim = neuron.embed_dim or 768
                num_heads = neuron.num_heads or 12
                mlp_ratio = getattr(neuron, 'mlp_ratio', 4.0)
                dropout = neuron.dropout or 0.1
                
                layer_def = f"        self.{layer_name} = VisionTransformerBlock(embed_dim={embed_dim}, num_heads={num_heads}, mlp_ratio={mlp_ratio}, dropout={dropout})"
            
            elif neuron.layer_type == 'deit_block':
                embed_dim = neuron.embed_dim or 384
                num_heads = neuron.num_heads or 6
                mlp_ratio = getattr(neuron, 'mlp_ratio', 4.0)
                dropout = neuron.dropout or 0.1
                drop_path = getattr(neuron, 'drop_path', 0.0)
                
                layer_def = f"        self.{layer_name} = DeiTBlock(embed_dim={embed_dim}, num_heads={num_heads}, mlp_ratio={mlp_ratio}, dropout={dropout}, drop_path={drop_path})"
            
            elif neuron.layer_type == 'swin_block':
                embed_dim = neuron.embed_dim or 96
                num_heads = neuron.num_heads or 3
                window_size = getattr(neuron, 'window_size', 7)
                shift_size = getattr(neuron, 'shift_size', 0)
                mlp_ratio = getattr(neuron, 'mlp_ratio', 4.0)
                dropout = neuron.dropout or 0.1
                
                layer_def = f"        self.{layer_name} = SwinTransformerBlock(embed_dim={embed_dim}, num_heads={num_heads}, window_size={window_size}, shift_size={shift_size}, mlp_ratio={mlp_ratio}, dropout={dropout})"
            
            else:
                # Fallback to Dense
                units = neuron.units or 128
                layer_def = f"        self.{layer_name} = nn.Linear(input_size, {units})"
            
            layer_defs.append(layer_def)
        
        return "\n".join(layer_defs)
    
    def _generate_dropout_layers(self, neurons: List[NeuronConfig]) -> str:
        """Генерация dropout слоев"""
        dropout_defs = []
        
        for neuron in neurons:
            if neuron.dropout:
                layer_name = neuron.name.lower()
                dropout_def = f"        self.dropout_{layer_name} = nn.Dropout({neuron.dropout})"
                dropout_defs.append(dropout_def)
        
        return "\n".join(dropout_defs) if dropout_defs else "        # No dropout layers"
    
    def _generate_batch_norm_layers(self, neurons: List[NeuronConfig]) -> str:
        """Генерация batch normalization слоев"""
        bn_defs = []
        
        for neuron in neurons:
            if getattr(neuron, 'batch_norm', False):
                layer_name = neuron.name.lower()
                
                if neuron.layer_type == 'conv':
                    filters = neuron.filters or 32
                    bn_def = f"        self.bn_{layer_name} = nn.BatchNorm2d({filters})"
                elif neuron.layer_type == 'dense':
                    units = neuron.units or 128
                    bn_def = f"        self.bn_{layer_name} = nn.BatchNorm1d({units})"
                else:
                    continue
                
                bn_defs.append(bn_def)
        
        return "\n".join(bn_defs) if bn_defs else "        # No batch normalization layers"
    
    def _generate_positional_encoding(self, neurons: List[NeuronConfig]) -> str:
        """Генерация positional encoding для Transformer"""
        has_transformer = any(n.layer_type == 'transformer' for n in neurons)
        
        if has_transformer:
            # Находим embed_dim из первого transformer слоя
            embed_dim = 512
            for neuron in neurons:
                if neuron.layer_type == 'transformer':
                    embed_dim = neuron.embed_dim or 512
                    break
            
            return f"        self.pos_encoding = PositionalEncoding({embed_dim})"
        
        return "        # No positional encoding needed"
    
    def _generate_forward_method(self, neurons: List[NeuronConfig]) -> str:
        """Генерация forward метода"""
        forward_lines = []
        
        for i, neuron in enumerate(neurons):
            layer_name = neuron.name.lower()
            
            # Применение слоя
            if neuron.layer_type == 'transformer':
                # Добавляем positional encoding
                forward_lines.append(f"        # Transformer with positional encoding")
                forward_lines.append(f"        x = self.pos_encoding(x)")
                forward_lines.append(f"        x = self.{layer_name}(x)")
            
            elif neuron.layer_type == 'attention':
                attention_type = getattr(neuron, 'attention_type', 'multihead')
                if attention_type == 'self':
                    forward_lines.append(f"        x = self.{layer_name}(x, x, x)")
                else:
                    forward_lines.append(f"        x, _ = self.{layer_name}(x, x, x)")
            
            elif neuron.layer_type in ['lstm', 'gru']:
                forward_lines.append(f"        x, _ = self.{layer_name}(x)")
            
            elif neuron.layer_type == 'resnet_block':
                forward_lines.append(f"        x = self.{layer_name}(x)")
            
            elif neuron.layer_type == 'embedding':
                forward_lines.append(f"        x = self.{layer_name}(x)")
            
            elif neuron.layer_type == 'layer_norm':
                forward_lines.append(f"        x = self.{layer_name}(x)")
            
            elif neuron.layer_type in ['vision_transformer', 'patch_embedding', 'vit_block', 'deit_block', 'swin_block']:
                forward_lines.append(f"        x = self.{layer_name}(x)")
            
            else:
                forward_lines.append(f"        x = self.{layer_name}(x)")
            
            # Применение batch normalization
            if getattr(neuron, 'batch_norm', False):
                forward_lines.append(f"        x = self.bn_{layer_name}(x)")
            
            # Применение активации
            if neuron.activation and neuron.activation in self.activation_mapping:
                activation_func = self.activation_mapping[neuron.activation]
                if activation_func:
                    if neuron.activation == 'softmax':
                        forward_lines.append(f"        x = {activation_func}(x, dim=-1)")
                    else:
                        forward_lines.append(f"        x = {activation_func}(x)")
            
            # Применение dropout
            if neuron.dropout:
                forward_lines.append(f"        x = self.dropout_{layer_name}(x)")
            
            # Пустая строка между слоями для читаемости
            if i < len(neurons) - 1:
                forward_lines.append("")
        
        return "\n".join(forward_lines)
    
    def generate_training_script(self, network: NetworkConfig) -> str:
        """Генерация скрипта обучения"""
        class_name = network.name
        optimizer = network.optimizer or 'adam'
        learning_rate = network.learning_rate or 0.001
        loss_func = network.loss or 'mse'
        batch_size = network.batch_size or 32
        epochs = network.epochs or 100
        
        optimizer_class = self.optimizer_mapping.get(optimizer, 'torch.optim.Adam')
        loss_class = self.loss_mapping.get(loss_func, 'nn.MSELoss')
        
        training_script = f"""#!/usr/bin/env python3
\"\"\"
Training script for {class_name}
Generated by AnamorphX Neural Backend
\"\"\"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import {class_name}


def load_dataset():
    """Load training dataset from disk."""
    try:
        X = np.load('train_x.npy')
        y = np.load('train_y.npy')
    except FileNotFoundError:
        # Fallback to dummy data if files are missing
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000, 1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {{device}}')
    
    # Model initialization
    model = {class_name}().to(device)
    
    # Loss and optimizer
    criterion = {loss_class}()
    optimizer = {optimizer_class}(model.parameters(), lr={learning_rate})
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training parameters
    batch_size = {batch_size}
    epochs = {epochs}
    
    # Load training data
    X_train, y_train = load_dataset()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {{epoch+1}}/{{epochs}}, Batch {{batch_idx}}, Loss: {{loss.item():.6f}}')
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {{epoch+1}}/{{epochs}} completed. Average Loss: {{avg_loss:.6f}}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({{
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }}, f'{class_name.lower()}_checkpoint_epoch_{{epoch+1}}.pth')
    
    # Save final model
    torch.save(model.state_dict(), '{class_name.lower()}_final.pth')
    print('Training completed!')

if __name__ == '__main__':
    train_model()
"""
        
        return training_script
    
    def generate_inference_script(self, network: NetworkConfig) -> str:
        """Генерация скрипта инференса"""
        class_name = network.name
        
        inference_script = f"""#!/usr/bin/env python3
\"\"\"
Inference script for {class_name}
Generated by AnamorphX Neural Backend
\"\"\"

import torch
import torch.nn.functional as F
import numpy as np
from model import {class_name}

class {class_name}Predictor:
    def __init__(self, model_path='{class_name.lower()}_final.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = {class_name}().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f'Model loaded on {{self.device}}')
    
    def predict(self, input_data):
        \"\"\"
        Make predictions on input data
        
        Args:
            input_data: numpy array or torch tensor
            
        Returns:
            predictions: numpy array
        \"\"\"
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.FloatTensor(input_data).to(self.device)
            else:
                input_tensor = input_data.to(self.device)
            
            # Add batch dimension if needed
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Convert to numpy
            predictions = output.cpu().numpy()
            
            return predictions
    
    def predict_batch(self, batch_data, batch_size=32):
        \"\"\"
        Make predictions on batch of data
        
        Args:
            batch_data: numpy array or torch tensor
            batch_size: int, batch size for processing
            
        Returns:
            predictions: numpy array
        \"\"\"
        all_predictions = []
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i+batch_size]
            predictions = self.predict(batch)
            all_predictions.append(predictions)
        
        return np.concatenate(all_predictions, axis=0)
    
    def predict_proba(self, input_data):
        \"\"\"
        Get prediction probabilities (for classification)
        \"\"\"
        predictions = self.predict(input_data)
        probabilities = F.softmax(torch.FloatTensor(predictions), dim=-1).numpy()
        return probabilities

def main():
    # Example usage
    predictor = {class_name}Predictor()
    
    # Example prediction
    sample_input = np.random.randn(1, 10)  # Replace with your input shape
    prediction = predictor.predict(sample_input)
    print(f'Prediction: {{prediction}}')
    
    # Batch prediction example
    batch_input = np.random.randn(100, 10)  # Replace with your input shape
    batch_predictions = predictor.predict_batch(batch_input)
    print(f'Batch predictions shape: {{batch_predictions.shape}}')

if __name__ == '__main__':
    main()
"""
        
        return inference_script 