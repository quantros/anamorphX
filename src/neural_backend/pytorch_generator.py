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
        
        # Depthwise Separable Convolution
        has_depthwise = any(getattr(n, 'depthwise_separable', False) for n in network.neurons)
        if has_depthwise:
            helper_classes.append("""
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
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
                layer_def = f"        self.{layer_name} = nn.LSTM(input_size, {units}, num_layers={num_layers}, bidirectional={bidirectional}, batch_first=True)"
            
            elif neuron.layer_type == 'gru':
                units = neuron.units or 128
                num_layers = getattr(neuron, 'num_layers', 1)
                bidirectional = getattr(neuron, 'bidirectional', False)
                layer_def = f"        self.{layer_name} = nn.GRU(input_size, {units}, num_layers={num_layers}, bidirectional={bidirectional}, batch_first=True)"
            
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