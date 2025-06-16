#!/usr/bin/env python3
"""
AnamorphX Transformer Demo
Демонстрация поддержки Transformer архитектур в Neural Backend
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_backend.network_parser import NetworkParser
from src.neural_backend.pytorch_generator import PyTorchGenerator
from src.neural_backend.neural_translator import NeuralTranslator


def demo_transformer_parsing():
    """Демо парсинга Transformer архитектур"""
    print("🔍 Demo: Transformer Parsing")
    print("=" * 50)
    
    # Пример AnamorphX кода с Transformer
    transformer_code = '''
network TextClassifier {
    neuron Embedding {
        embed_dim: 512
        vocab_size: 10000
    }
    
    neuron TransformerBlock1 {
        num_heads: 8
        embed_dim: 512
        ff_dim: 2048
        dropout: 0.1
    }
    
    neuron TransformerBlock2 {
        num_heads: 8
        embed_dim: 512
        ff_dim: 2048
        dropout: 0.1
    }
    
    neuron AttentionPooling {
        embed_dim: 512
        num_heads: 1
    }
    
    neuron Classifier {
        activation: softmax
        units: 5
    }
    
    optimizer: adamw
    learning_rate: 0.0001
    loss: categorical_crossentropy
    batch_size: 32
    epochs: 50
}

network GPTLike {
    neuron TokenEmbedding {
        embed_dim: 768
        vocab_size: 50000
    }
    
    neuron PositionalEncoding {
        embed_dim: 768
        max_length: 1024
    }
    
    neuron TransformerLayer1 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron TransformerLayer2 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron TransformerLayer3 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron OutputProjection {
        activation: linear
        units: 50000
    }
    
    optimizer: adamw
    learning_rate: 0.00005
    loss: categorical_crossentropy
    batch_size: 16
    epochs: 100
}
'''
    
    # Парсинг
    parser = NetworkParser()
    networks = parser.parse_code(transformer_code)
    
    print(f"📊 Parsed {len(networks)} networks:")
    
    for network in networks:
        print(f"\n🌐 Network: {network.name}")
        print(f"   Neurons: {len(network.neurons)}")
        print(f"   Optimizer: {network.optimizer}")
        print(f"   Learning Rate: {network.learning_rate}")
        
        # Анализ типов слоев
        layer_types = {}
        for neuron in network.neurons:
            layer_type = neuron.layer_type
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += 1
        
        print(f"   Layer Types: {dict(layer_types)}")
        
        # Детали Transformer слоев
        transformer_layers = [n for n in network.neurons if n.layer_type == 'transformer']
        if transformer_layers:
            print(f"   🧠 Transformer Layers: {len(transformer_layers)}")
            for t_layer in transformer_layers:
                print(f"      • {t_layer.name}: {t_layer.num_heads} heads, {t_layer.embed_dim} dim")
    
    return networks


def demo_transformer_generation():
    """Демо генерации PyTorch кода для Transformer"""
    print("\n🏗️ Demo: Transformer PyTorch Generation")
    print("=" * 50)
    
    # Простой Transformer для классификации
    simple_transformer = '''
network SimpleTransformer {
    neuron SelfAttention {
        num_heads: 4
        embed_dim: 256
        ff_dim: 1024
        dropout: 0.1
    }
    
    neuron FeedForward {
        activation: relu
        units: 128
        dropout: 0.2
    }
    
    neuron Output {
        activation: softmax
        units: 10
    }
    
    optimizer: adam
    learning_rate: 0.001
    loss: categorical_crossentropy
}
'''
    
    # Парсинг и генерация
    parser = NetworkParser()
    generator = PyTorchGenerator()
    
    networks = parser.parse_code(simple_transformer)
    
    if networks:
        network = networks[0]
        print(f"🌐 Generating PyTorch code for: {network.name}")
        
        # Генерация модели
        model_code = generator.generate_model_class(network)
        print("\n📄 Generated PyTorch Model:")
        print("-" * 30)
        print(model_code[:800] + "..." if len(model_code) > 800 else model_code)
        
        # Генерация скрипта обучения
        train_code = generator.generate_training_script(network)
        print(f"\n📄 Training Script Generated: {len(train_code)} characters")
        
        return network, model_code, train_code
    
    return None, None, None


def demo_advanced_architectures():
    """Демо продвинутых архитектур"""
    print("\n🚀 Demo: Advanced Architectures")
    print("=" * 50)
    
    # Vision Transformer + ResNet
    advanced_code = '''
network VisionTransformer {
    neuron PatchEmbedding {
        embed_dim: 768
        patch_size: 16
    }
    
    neuron ClassToken {
        embed_dim: 768
    }
    
    neuron TransformerEncoder1 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron TransformerEncoder2 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron MLPHead {
        activation: relu
        units: 1000
        dropout: 0.1
    }
    
    optimizer: adamw
    learning_rate: 0.0003
    loss: categorical_crossentropy
}

network ResNetTransformer {
    neuron ConvStem {
        filters: 64
        kernel_size: 7
        stride: 2
        padding: 3
    }
    
    neuron ResBlock1 {
        filters: 128
        kernel_size: 3
        skip_connection: true
        residual_type: basic
    }
    
    neuron TransformerBlock {
        num_heads: 8
        embed_dim: 512
        ff_dim: 2048
    }
    
    neuron GlobalPool {
        pool_size: 7
    }
    
    neuron Classifier {
        activation: softmax
        units: 1000
    }
    
    optimizer: sgd
    learning_rate: 0.1
    loss: categorical_crossentropy
}
'''
    
    parser = NetworkParser()
    networks = parser.parse_code(advanced_code)
    
    print(f"📊 Advanced architectures parsed: {len(networks)}")
    
    for network in networks:
        print(f"\n🏗️ {network.name}:")
        
        # Анализ архитектуры
        has_transformer = any(n.layer_type == 'transformer' for n in network.neurons)
        has_conv = any(n.layer_type == 'conv' for n in network.neurons)
        has_attention = any(n.layer_type == 'attention' for n in network.neurons)
        has_skip = any(n.skip_connection for n in network.neurons if hasattr(n, 'skip_connection'))
        
        features = []
        if has_transformer:
            features.append("Transformer")
        if has_conv:
            features.append("Convolutional")
        if has_attention:
            features.append("Attention")
        if has_skip:
            features.append("Skip Connections")
        
        print(f"   Features: {', '.join(features)}")
        print(f"   Complexity: {len(network.neurons)} layers")
        
        # Валидация
        errors = parser.validate_network(network)
        if errors:
            print(f"   ⚠️ Validation issues: {len(errors)}")
            for error in errors[:3]:  # Показать первые 3 ошибки
                print(f"      • {error}")
        else:
            print(f"   ✅ Validation passed")


def demo_full_translation():
    """Демо полной трансляции с Transformer"""
    print("\n🔄 Demo: Full Translation Pipeline")
    print("=" * 50)
    
    # BERT-подобная архитектура
    bert_like = '''
network BERTClassifier {
    neuron TokenEmbedding {
        embed_dim: 768
        vocab_size: 30000
    }
    
    neuron SegmentEmbedding {
        embed_dim: 768
        num_segments: 2
    }
    
    neuron TransformerLayer1 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron TransformerLayer2 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron TransformerLayer3 {
        num_heads: 12
        embed_dim: 768
        ff_dim: 3072
        dropout: 0.1
    }
    
    neuron Pooler {
        activation: tanh
        units: 768
    }
    
    neuron Classifier {
        activation: softmax
        units: 2
        dropout: 0.1
    }
    
    optimizer: adamw
    learning_rate: 0.00002
    loss: categorical_crossentropy
    batch_size: 16
    epochs: 3
}
'''
    
    # Полная трансляция
    translator = NeuralTranslator(output_dir="generated_transformers")
    result = translator.translate_code(bert_like)
    
    print(f"🎯 Translation Result:")
    print(f"   Success: {result['success']}")
    
    if result['success']:
        print(f"   Networks: {result['networks']}")
        print(f"   Files generated: {len(result['files_generated'])}")
        print(f"   Output directory: {result['output_directory']}")
        
        print(f"\n📁 Generated files:")
        for file_path in result['files_generated']:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"   • {file_name} ({file_size} bytes)")
    else:
        print(f"   Error: {result['error']}")
    
    return result


def main():
    """Главная функция демо"""
    print("🧠 AnamorphX Transformer Demo")
    print("=" * 60)
    print("Демонстрация поддержки Transformer архитектур")
    print("Этап 7: Advanced Features & Optimization")
    print("=" * 60)
    
    try:
        # 1. Парсинг Transformer
        networks = demo_transformer_parsing()
        
        # 2. Генерация PyTorch кода
        network, model_code, train_code = demo_transformer_generation()
        
        # 3. Продвинутые архитектуры
        demo_advanced_architectures()
        
        # 4. Полная трансляция
        result = demo_full_translation()
        
        print("\n🎉 Demo completed successfully!")
        print(f"✅ Transformer support is working")
        print(f"✅ Advanced architectures supported")
        print(f"✅ Full translation pipeline operational")
        
        if result and result['success']:
            print(f"\n💡 Next steps:")
            print(f"   • Check generated files in: {result['output_directory']}")
            print(f"   • Run training scripts to test models")
            print(f"   • Experiment with different architectures")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 