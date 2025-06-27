"""
🧠 AnamorphX Neural Engine - Enterprise Edition
Продвинутая нейронная библиотека для enterprise применений

Features:
- Advanced Neural Networks (Transformer, LSTM, CNN, GRU)
- Distributed Computing & Cluster Management  
- Real-time Analytics & Monitoring
- AI Model Optimization (Quantization, Pruning, AutoML)
- Blockchain Integration & Decentralized ML
- Enterprise Security & Authentication
- Auto-scaling & Load Balancing
- Progressive Web App Support
"""

# Core modules
from .core.neural_engine import NeuralEngine
from .core.model_manager import ModelManager

# Advanced core
try:
    from .core.advanced_neural_engine import AdvancedNeuralEngine, ModelType, TrainingConfig
except ImportError:
    pass

# Backend modules
from .backend.api_server import APIServer

# Frontend modules  
from .frontend.spa_handler import SPAHandler

# Security modules
from .security.jwt_auth import JWTAuth
from .security.rate_limiter import RateLimiter

# Monitoring modules
from .monitoring.metrics_collector import MetricsCollector

# Utils
from .utils.config_manager import ConfigManager
from .utils.logger import setup_logger

# Enterprise modules
try:
    from .enterprise.distributed_computing import (
        ClusterManager, DistributedTaskManager, DistributedNeuralNetwork
    )
    from .enterprise.ai_optimization import (
        AutoMLOptimizer, ModelProfiler, ModelQuantizer, ModelPruner
    )
    from .enterprise.realtime_analytics import (
        RealTimeAnalytics, MetricAggregator, AlertManager
    )
    from .enterprise.blockchain_integration import (
        BlockchainIntegration, BlockchainModelRegistry, DecentralizedTraining
    )
    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False

__version__ = "2.0.0-enterprise"
__author__ = "AnamorphX Team"

__all__ = [
    # Core
    'NeuralEngine',
    'ModelManager',
    
    # Backend/Frontend
    'APIServer',
    'SPAHandler',
    
    # Security
    'JWTAuth', 
    'RateLimiter',
    
    # Monitoring
    'MetricsCollector',
    
    # Utils
    'ConfigManager',
    'setup_logger'
]

# Add enterprise exports if available
if ENTERPRISE_AVAILABLE:
    __all__.extend([
        'AdvancedNeuralEngine',
        'ModelType',
        'TrainingConfig',
        'ClusterManager',
        'DistributedTaskManager',
        'DistributedNeuralNetwork', 
        'AutoMLOptimizer',
        'ModelProfiler',
        'ModelQuantizer',
        'ModelPruner',
        'RealTimeAnalytics',
        'MetricAggregator',
        'AlertManager',
        'BlockchainIntegration',
        'BlockchainModelRegistry',
        'DecentralizedTraining'
    ])

def get_enterprise_features():
    """Получить список доступных enterprise функций"""
    if not ENTERPRISE_AVAILABLE:
        return []
    
    return [
        "🧠 Advanced Neural Networks",
        "🌐 Distributed Computing", 
        "📊 Real-time Analytics",
        "🤖 AI Optimization",
        "⛓️ Blockchain Integration",
        "🔐 Enterprise Security",
        "📈 Auto-scaling",
        "🎨 Progressive Web Apps"
    ]

def print_welcome():
    """Вывод приветствия библиотеки"""
    print("=" * 80)
    print("🧠 AnamorphX Neural Engine - Enterprise Edition")
    print("=" * 80)
    print(f"📦 Version: {__version__}")
    print(f"🏢 Enterprise Features: {'✅ Available' if ENTERPRISE_AVAILABLE else '❌ Not Available'}")
    
    if ENTERPRISE_AVAILABLE:
        features = get_enterprise_features()
        print("\n🚀 Available Enterprise Features:")
        for feature in features:
            print(f"   {feature}")
    
    print("\n💡 Quick Start:")
    print("   from anamorph_neural_engine import NeuralEngine")
    print("   engine = NeuralEngine()")
    
    if ENTERPRISE_AVAILABLE:
        print("\n🏢 Enterprise Quick Start:")
        print("   from anamorph_neural_engine import AdvancedNeuralEngine, ClusterManager")
        print("   engine = AdvancedNeuralEngine()")
        print("   cluster = ClusterManager('node1')")
    
    print("=" * 80)

# Print welcome message on import
print_welcome() 