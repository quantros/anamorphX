"""
⛓️ Blockchain Integration для AnamorphX Enterprise
Интеграция с блокчейн сетями для децентрализованного ML
"""

import hashlib
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import uuid

class BlockchainNetwork(Enum):
    """Поддерживаемые блокчейн сети"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    AVALANCHE = "avalanche"
    CUSTOM = "custom"

@dataclass
class ModelContract:
    """Контракт модели в блокчейне"""
    contract_id: str
    model_hash: str
    owner_address: str
    creation_time: float
    model_metadata: Dict[str, Any]
    access_price: float = 0.0
    usage_count: int = 0
    network: BlockchainNetwork = BlockchainNetwork.ETHEREUM

@dataclass
class TrainingRecord:
    """Запись об обучении в блокчейне"""
    record_id: str
    model_contract_id: str
    trainer_address: str
    training_data_hash: str
    accuracy_achieved: float
    training_time: float
    reward_earned: float
    timestamp: float

class BlockchainModelRegistry:
    """Реестр моделей в блокчейне"""
    
    def __init__(self, network: BlockchainNetwork = BlockchainNetwork.ETHEREUM):
        self.network = network
        self.contracts: Dict[str, ModelContract] = {}
        self.training_records: List[TrainingRecord] = []
        self.logger = logging.getLogger(__name__)
        
        print(f"⛓️ Blockchain Model Registry инициализирован")
        print(f"   🌐 Сеть: {network.value}")
    
    async def register_model(self, model_data: bytes, metadata: Dict[str, Any],
                           owner_address: str, access_price: float = 0.0) -> str:
        """Регистрация модели в блокчейне"""
        # Создание хэша модели
        model_hash = hashlib.sha256(model_data).hexdigest()
        
        # Генерация ID контракта
        contract_id = f"contract_{uuid.uuid4().hex[:16]}"
        
        # Создание контракта
        contract = ModelContract(
            contract_id=contract_id,
            model_hash=model_hash,
            owner_address=owner_address,
            creation_time=time.time(),
            model_metadata=metadata,
            access_price=access_price,
            network=self.network
        )
        
        # Сохранение в реестре
        self.contracts[contract_id] = contract
        
        # Симуляция записи в блокчейн
        await self._write_to_blockchain(contract)
        
        self.logger.info(f"📝 Модель зарегистрирована: {contract_id}")
        return contract_id
    
    async def _write_to_blockchain(self, data: Any):
        """Симуляция записи в блокчейн"""
        # В реальной реализации здесь была бы отправка транзакции
        await asyncio.sleep(0.1)  # Симуляция времени записи
        self.logger.debug(f"⛓️ Данные записаны в {self.network.value}")
    
    async def verify_model_integrity(self, contract_id: str, model_data: bytes) -> bool:
        """Проверка целостности модели"""
        if contract_id not in self.contracts:
            return False
        
        contract = self.contracts[contract_id]
        current_hash = hashlib.sha256(model_data).hexdigest()
        
        return current_hash == contract.model_hash
    
    async def purchase_model_access(self, contract_id: str, buyer_address: str) -> Dict[str, Any]:
        """Покупка доступа к модели"""
        if contract_id not in self.contracts:
            return {'success': False, 'error': 'Контракт не найден'}
        
        contract = self.contracts[contract_id]
        
        # Симуляция платежа
        payment_success = await self._process_payment(
            buyer_address, contract.owner_address, contract.access_price
        )
        
        if payment_success:
            contract.usage_count += 1
            
            return {
                'success': True,
                'access_token': f"access_{uuid.uuid4().hex[:16]}",
                'contract_id': contract_id,
                'valid_until': time.time() + 86400  # 24 часа
            }
        else:
            return {'success': False, 'error': 'Ошибка платежа'}
    
    async def _process_payment(self, from_address: str, to_address: str, amount: float) -> bool:
        """Симуляция обработки платежа"""
        # В реальной реализации здесь была бы отправка токенов
        await asyncio.sleep(0.2)
        return amount >= 0  # Простая проверка
    
    def get_model_contract(self, contract_id: str) -> Optional[ModelContract]:
        """Получение контракта модели"""
        return self.contracts.get(contract_id)
    
    def search_models(self, query: Dict[str, Any]) -> List[ModelContract]:
        """Поиск моделей по критериям"""
        results = []
        
        for contract in self.contracts.values():
            matches = True
            
            # Проверка метаданных
            if 'task_type' in query:
                if contract.model_metadata.get('task_type') != query['task_type']:
                    matches = False
            
            if 'min_accuracy' in query:
                if contract.model_metadata.get('accuracy', 0) < query['min_accuracy']:
                    matches = False
            
            if 'max_price' in query:
                if contract.access_price > query['max_price']:
                    matches = False
            
            if matches:
                results.append(contract)
        
        return results

class DecentralizedTraining:
    """Децентрализованное обучение"""
    
    def __init__(self, registry: BlockchainModelRegistry):
        self.registry = registry
        self.training_nodes: Dict[str, Dict[str, Any]] = {}
        self.active_training_tasks: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def register_training_node(self, node_address: str, 
                                   capabilities: Dict[str, Any]) -> bool:
        """Регистрация узла для обучения"""
        self.training_nodes[node_address] = {
            'capabilities': capabilities,
            'reputation': 100,  # Начальная репутация
            'completed_tasks': 0,
            'registration_time': time.time()
        }
        
        self.logger.info(f"🖥️ Узел обучения зарегистрирован: {node_address}")
        return True
    
    async def submit_training_task(self, contract_id: str, training_config: Dict[str, Any],
                                 reward_amount: float) -> str:
        """Отправка задачи на обучение"""
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        
        # Поиск подходящих узлов
        suitable_nodes = self._find_suitable_nodes(training_config)
        
        if not suitable_nodes:
            raise ValueError("Нет подходящих узлов для обучения")
        
        # Выбор лучшего узла
        selected_node = max(suitable_nodes, key=lambda n: self.training_nodes[n]['reputation'])
        
        # Создание задачи
        task = {
            'task_id': task_id,
            'contract_id': contract_id,
            'assigned_node': selected_node,
            'training_config': training_config,
            'reward_amount': reward_amount,
            'status': 'pending',
            'created_at': time.time()
        }
        
        self.active_training_tasks[task_id] = task
        
        # Уведомление узла
        await self._notify_training_node(selected_node, task)
        
        self.logger.info(f"📋 Задача обучения создана: {task_id}")
        return task_id
    
    def _find_suitable_nodes(self, requirements: Dict[str, Any]) -> List[str]:
        """Поиск подходящих узлов"""
        suitable = []
        
        for node_address, node_info in self.training_nodes.items():
            capabilities = node_info['capabilities']
            
            # Проверка требований
            if requirements.get('min_gpu_memory', 0) <= capabilities.get('gpu_memory_gb', 0):
                if requirements.get('min_cpu_cores', 0) <= capabilities.get('cpu_cores', 0):
                    if node_info['reputation'] >= requirements.get('min_reputation', 50):
                        suitable.append(node_address)
        
        return suitable
    
    async def _notify_training_node(self, node_address: str, task: Dict[str, Any]):
        """Уведомление узла о задаче"""
        # В реальной реализации здесь была бы отправка сообщения узлу
        self.logger.info(f"📤 Уведомление отправлено узлу {node_address}")
    
    async def complete_training_task(self, task_id: str, results: Dict[str, Any],
                                   trainer_address: str) -> bool:
        """Завершение задачи обучения"""
        if task_id not in self.active_training_tasks:
            return False
        
        task = self.active_training_tasks[task_id]
        
        # Проверка результатов
        if self._validate_training_results(results):
            # Создание записи об обучении
            record = TrainingRecord(
                record_id=f"record_{uuid.uuid4().hex[:16]}",
                model_contract_id=task['contract_id'],
                trainer_address=trainer_address,
                training_data_hash=results.get('data_hash', ''),
                accuracy_achieved=results.get('accuracy', 0.0),
                training_time=results.get('training_time', 0.0),
                reward_earned=task['reward_amount'],
                timestamp=time.time()
            )
            
            self.registry.training_records.append(record)
            
            # Обновление репутации узла
            if trainer_address in self.training_nodes:
                self.training_nodes[trainer_address]['reputation'] += 10
                self.training_nodes[trainer_address]['completed_tasks'] += 1
            
            # Обновление статуса задачи
            task['status'] = 'completed'
            task['completed_at'] = time.time()
            task['results'] = results
            
            # Выплата награды
            await self._distribute_reward(trainer_address, task['reward_amount'])
            
            self.logger.info(f"✅ Задача обучения завершена: {task_id}")
            return True
        else:
            task['status'] = 'failed'
            self.logger.warning(f"❌ Задача обучения провалена: {task_id}")
            return False
    
    def _validate_training_results(self, results: Dict[str, Any]) -> bool:
        """Валидация результатов обучения"""
        required_fields = ['accuracy', 'training_time', 'model_hash']
        
        for field in required_fields:
            if field not in results:
                return False
        
        # Проверка разумности значений
        if not (0 <= results['accuracy'] <= 1):
            return False
        
        if results['training_time'] <= 0:
            return False
        
        return True
    
    async def _distribute_reward(self, recipient_address: str, amount: float):
        """Выплата награды"""
        # В реальной реализации здесь была бы отправка токенов
        self.logger.info(f"💰 Награда {amount} отправлена на {recipient_address}")

class NFTModelMarketplace:
    """NFT маркетплейс для моделей"""
    
    def __init__(self, registry: BlockchainModelRegistry):
        self.registry = registry
        self.nft_tokens: Dict[str, Dict[str, Any]] = {}
        self.marketplace_listings: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def mint_model_nft(self, contract_id: str, owner_address: str,
                           nft_metadata: Dict[str, Any]) -> str:
        """Создание NFT для модели"""
        if contract_id not in self.registry.contracts:
            raise ValueError("Контракт модели не найден")
        
        nft_id = f"nft_{uuid.uuid4().hex[:16]}"
        
        nft_token = {
            'nft_id': nft_id,
            'contract_id': contract_id,
            'owner_address': owner_address,
            'metadata': nft_metadata,
            'minted_at': time.time(),
            'transfer_history': []
        }
        
        self.nft_tokens[nft_id] = nft_token
        
        # Симуляция минтинга в блокчейне
        await self.registry._write_to_blockchain(nft_token)
        
        self.logger.info(f"🎨 NFT модели создан: {nft_id}")
        return nft_id
    
    async def list_nft_for_sale(self, nft_id: str, price: float,
                              seller_address: str) -> bool:
        """Выставление NFT на продажу"""
        if nft_id not in self.nft_tokens:
            return False
        
        nft = self.nft_tokens[nft_id]
        
        if nft['owner_address'] != seller_address:
            return False
        
        listing_id = f"listing_{uuid.uuid4().hex[:16]}"
        
        self.marketplace_listings[listing_id] = {
            'listing_id': listing_id,
            'nft_id': nft_id,
            'seller_address': seller_address,
            'price': price,
            'listed_at': time.time(),
            'status': 'active'
        }
        
        self.logger.info(f"🏷️ NFT выставлен на продажу: {nft_id} за {price}")
        return True
    
    async def purchase_nft(self, listing_id: str, buyer_address: str) -> Dict[str, Any]:
        """Покупка NFT"""
        if listing_id not in self.marketplace_listings:
            return {'success': False, 'error': 'Лот не найден'}
        
        listing = self.marketplace_listings[listing_id]
        
        if listing['status'] != 'active':
            return {'success': False, 'error': 'Лот неактивен'}
        
        nft_id = listing['nft_id']
        nft = self.nft_tokens[nft_id]
        
        # Симуляция платежа
        payment_success = await self.registry._process_payment(
            buyer_address, listing['seller_address'], listing['price']
        )
        
        if payment_success:
            # Перевод ownership
            nft['owner_address'] = buyer_address
            nft['transfer_history'].append({
                'from': listing['seller_address'],
                'to': buyer_address,
                'price': listing['price'],
                'timestamp': time.time()
            })
            
            # Обновление статуса лота
            listing['status'] = 'sold'
            listing['sold_at'] = time.time()
            listing['buyer_address'] = buyer_address
            
            self.logger.info(f"💎 NFT продан: {nft_id}")
            
            return {
                'success': True,
                'nft_id': nft_id,
                'transaction_hash': f"tx_{uuid.uuid4().hex[:16]}"
            }
        else:
            return {'success': False, 'error': 'Ошибка платежа'}
    
    def get_marketplace_listings(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Получение списка лотов на маркетплейсе"""
        listings = []
        
        for listing in self.marketplace_listings.values():
            if listing['status'] != 'active':
                continue
            
            # Применение фильтров
            if filters:
                if 'max_price' in filters and listing['price'] > filters['max_price']:
                    continue
                if 'min_price' in filters and listing['price'] < filters['min_price']:
                    continue
            
            # Добавление информации о NFT
            nft = self.nft_tokens[listing['nft_id']]
            contract = self.registry.contracts[nft['contract_id']]
            
            listing_info = listing.copy()
            listing_info['nft_metadata'] = nft['metadata']
            listing_info['model_metadata'] = contract.model_metadata
            
            listings.append(listing_info)
        
        return listings

class BlockchainIntegration:
    """Главный класс интеграции с блокчейном"""
    
    def __init__(self, network: BlockchainNetwork = BlockchainNetwork.ETHEREUM):
        self.network = network
        self.registry = BlockchainModelRegistry(network)
        self.training = DecentralizedTraining(self.registry)
        self.marketplace = NFTModelMarketplace(self.registry)
        self.logger = logging.getLogger(__name__)
        
        print(f"⛓️ Blockchain Integration инициализирован")
        print(f"   🌐 Сеть: {network.value}")
    
    async def deploy_model_to_blockchain(self, model_data: bytes, 
                                       metadata: Dict[str, Any],
                                       owner_address: str) -> Dict[str, str]:
        """Развертывание модели в блокчейне"""
        # Регистрация контракта
        contract_id = await self.registry.register_model(
            model_data, metadata, owner_address
        )
        
        # Создание NFT
        nft_id = await self.marketplace.mint_model_nft(
            contract_id, owner_address, {
                'name': metadata.get('name', 'Unnamed Model'),
                'description': metadata.get('description', ''),
                'image': metadata.get('image_url', ''),
                'attributes': metadata.get('attributes', [])
            }
        )
        
        return {
            'contract_id': contract_id,
            'nft_id': nft_id
        }
    
    async def start_decentralized_training(self, model_config: Dict[str, Any],
                                         reward_pool: float) -> str:
        """Запуск децентрализованного обучения"""
        # Создание фиктивного контракта для обучения
        contract_id = await self.registry.register_model(
            b"initial_model_data", model_config, "training_coordinator"
        )
        
        # Отправка задачи на обучение
        task_id = await self.training.submit_training_task(
            contract_id, model_config, reward_pool
        )
        
        return task_id
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Получение статистики блокчейна"""
        return {
            'network': self.network.value,
            'total_contracts': len(self.registry.contracts),
            'total_nfts': len(self.marketplace.nft_tokens),
            'active_listings': len([l for l in self.marketplace.marketplace_listings.values() 
                                  if l['status'] == 'active']),
            'training_nodes': len(self.training.training_nodes),
            'completed_training_tasks': len([t for t in self.training.active_training_tasks.values()
                                           if t['status'] == 'completed']),
            'total_training_records': len(self.registry.training_records)
        } 