import asyncio
import aiohttp
import time
import random
import json

SERVER_URL = "http://localhost:8080"

async def neural_predict(session, idx):
    data = {
        "data": f"test input {idx} " + " ".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10)),
        "model": "enterprise_classifier"
    }
    start = time.perf_counter()
    async with session.post(f"{SERVER_URL}/api/v1/neural/predict", json=data) as resp:
        r = await resp.json()
        duration = time.perf_counter() - start
        success = resp.status == 200 and 'confidence' in r
        print(f"[Predict {idx}] Status: {resp.status} Time: {duration:.3f}s Success: {success}")
        return duration, success

async def neural_train(session, idx):
    data = {
        "training_data": ["sample", "data", str(idx)],
        "config": {"epochs": 5, "batch_size": 16}
    }
    start = time.perf_counter()
    async with session.post(f"{SERVER_URL}/api/v1/neural/train", json=data) as resp:
        r = await resp.json()
        duration = time.perf_counter() - start
        success = resp.status == 200 and r.get('status') == 'submitted'
        print(f"[Train {idx}] Status: {resp.status} Time: {duration:.3f}s Success: {success}")
        return duration, success

async def get_models(session):
    start = time.perf_counter()
    async with session.get(f"{SERVER_URL}/api/v1/neural/models") as resp:
        r = await resp.json()
        duration = time.perf_counter() - start
        success = resp.status == 200 and 'models' in r
        print(f"[Models] Status: {resp.status} Time: {duration:.3f}s Success: {success}")
        return duration, success

async def send_suspicious_request(session, idx):
    # Проверка защиты — запрос с вредоносным payload
    data = {
        "request": {
            "path": "/api/v1/neural/predict",
            "query": "union select password from users",
            "client_ip": f"192.168.1.{idx}"
        }
    }
    start = time.perf_counter()
    async with session.post(f"{SERVER_URL}/api/v1/security/analyze", json=data) as resp:
        r = await resp.json()
        duration = time.perf_counter() - start
        blocked = r.get('blocked', False)
        print(f"[Security {idx}] Status: {resp.status} Time: {duration:.3f}s Blocked: {blocked}")
        return duration, blocked

async def worker(name, session, n_predicts, n_trains, security_test=False):
    results = {'predict': [], 'train': [], 'security': []}
    for i in range(n_predicts):
        d, s = await neural_predict(session, i)
        results['predict'].append((d, s))
        # Имитация пользовательского интервала между запросами
        await asyncio.sleep(random.uniform(0.05, 0.2))
    
    for i in range(n_trains):
        d, s = await neural_train(session, i)
        results['train'].append((d, s))
        await asyncio.sleep(random.uniform(1.0, 3.0))
    
    if security_test:
        for i in range(5):
            d, s = await send_suspicious_request(session, i)
            results['security'].append((d, s))
            await asyncio.sleep(0.1)
    
    return results

async def main(total_workers=50):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(total_workers):
            security_test = (i % 10 == 0)  # Каждый 10-й делает security тесты
            task = worker(f"worker_{i}", session, n_predicts=20, n_trains=2, security_test=security_test)
            tasks.append(task)
        
        all_results = await asyncio.gather(*tasks)
        
        # Собираем статистику
        all_predict_times = [t for res in all_results for t, s in res['predict'] if s]
        all_train_times = [t for res in all_results for t, s in res['train'] if s]
        security_blocks = sum(1 for res in all_results for _, b in res['security'] if b)
        
        print("\n=== Нагрузка завершена ===")
        print(f"Общее успешных предсказаний: {len(all_predict_times)}")
        print(f"Среднее время предсказания: {sum(all_predict_times)/len(all_predict_times):.3f}s")
        print(f"Общее успешных тренировок: {len(all_train_times)}")
        print(f"Среднее время тренировки (симуляция): {sum(all_train_times)/len(all_train_times):.3f}s")
        print(f"Обнаружено блокировок по безопасности: {security_blocks}")

if __name__ == '__main__':
    asyncio.run(main())
