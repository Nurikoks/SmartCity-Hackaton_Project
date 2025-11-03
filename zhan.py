# =======================================================
# БЛОК 1: КОНФИГУРАЦИЯ И ПРОМПТ
# =======================================================
import os
import math
# (Импорты google.genai, LangChain, ChromaDB пойдут сюда)

# Настройка API (с заглушкой)
# ... ваш код с try/except ...
# client = genai.Client(...)

# Ваш SYSTEM_PROMPT (многострочная строка)
SYSTEM_PROMPT = """..."""

# =======================================================
# БЛОК 2: ЛОГИКА ФИЛЬТРАЦИИ И ПОИСКА (RETRIEVER)
# =======================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расчет расстояния (формула Гейверсина)"""
    # ... ваш код haversine_distance ...
    pass


# def search_poi_with_filter(...): # Эту функцию вы напишете, когда ДБЛ-1 даст данные
#     # ... Логика поиска в ChromaDB и фильтрации по haversine_distance ...
#     pass


# =======================================================
# БЛОК 3: ГЕНЕРАЦИЯ (GENERATOR)
# =======================================================

def generate_recommendation_STUB(context: str, query: str) -> str:
    """Заглушка для функции генерации LLM."""
    # ... ваш код заглушки ...
    pass


# =======================================================
# БЛОК 4: ФИНАЛЬНЫЙ ИНТЕРФЕЙС ДЛЯ ДБЛ-3
# =======================================================

def get_final_recommendation(user_query: str, user_lat: float, user_lon: float) -> str:
    """
    Мастер-функция, которую будет вызывать Telegram-бот ДБЛ-3.
    """
    # 1. Поиск: context = search_poi_with_filter(user_query, user_lat, user_lon)
    context = "ВРЕМЕННЫЙ КОНТЕКСТ: Кафе 'Коктем', 51.13, 71.45. Описание:..."
    
    # 2. Генерация:
    final_response = generate_recommendation_STUB(context, user_query)
    
    return final_response


# Только для тестирования ДБЛ-2:
if __name__ == "__main__":
    test_query = "Хочу тихое место с кофе и розеткой недалеко от меня."
    # print(get_final_recommendation(test_query, 51.1290, 71.4328))
    # print(f"Расстояние: {haversine_distance(51.1290, 71.4328, 51.1417, 71.4087):.2f} км")
