import os
import re
import math
import json
import pandas as pd
from typing import List, Tuple

#ИМПОРТЫ ДЛЯ RAG (Chroma + SentenceTransformer) 
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
    from langchain_core.documents import Document
    HAS_RAG_LIBS = True
except Exception as e:
    print(f"⚠️ Ошибка импорта RAG библиотек: {e}")
    HAS_RAG_LIBS = False

#ИМПОРТ ДЛЯ GEMINI (LLM)
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        HAS_GEMINI = True
    else:
        HAS_GEMINI = False
except ImportError:
    genai = None
    HAS_GEMINI = False

# Настройки RAG
VECTOR_DB_PATH = "./chroma_db"
CLEAN_DATA_PATH = "data/gis_clean.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

#SYSTEM_PROMPT 
SYSTEM_PROMPT = """
Ты — дружелюбный эксперт-гид по прогулкам в городе Астана. 
Твоя цель — дать пользователю 1-3 самых релевантных, конкретных и вдохновляющих рекомендаций.

---
[КОНТЕКСТ POI]
{context}

---
ИНСТРУКЦИИ (СЛЕДОВАТЬ СТРОГО):

1. РЕЛЕВАНТНОСТЬ: Используй ТОЛЬКО информацию из блока [КОНТЕКСТ POI].
2. ФОРМАТ: Ответ должен быть кратким и в форме списка (1-3 пункта). Для каждого пункта:
   - Название, Расстояние (в метрах) и примерное время на посещение (если есть).
   - ПОЧЕМУ: Краткое убедительное объяснение (мотивация).
   - ПЛАН ДЕЙСТВИЙ: Конкретная идея "что там сделать" (микро-маршрут/рекомендация).
3. ПРАВИЛА: 
   - Ясно укажи, какие ФИЛЬТРЫ были использованы (локация, категория).
   - Избегай уверенных утверждений о доступности ('открыто сейчас'), если данных нет.
"""

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расчет расстояния между двумя точками на Земле (в километрах)."""
    R = 6371  # радиус Земли в км
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def index_data_to_chroma(data_source_path: str = CLEAN_DATA_PATH):
    """
    ФУНКЦИЯ ВЕКТОРИЗАЦИИ: Загружает данные, создает документы и индексирует в ChromaDB.
    Запускается один раз для создания/обновления векторной БД.
    """
    if not HAS_RAG_LIBS:
        print("⚠️ Модули LangChain/ChromaDB не установлены. Пропуск индексации.")
        return

    if not os.path.exists(data_source_path):
        print(f"❌ ОШИБКА: Файл данных не найден по пути: {data_source_path}")
        return

    df = pd.read_csv(data_source_path, sep=";", dtype=str)
    documents = []

    for _, row in df.iterrows():
        # Контент для векторизации
        content = (
            f"Название: {row.get('Название', '')}. "
            f"Рубрика: {row.get('Рубрика', '')} ({row.get('Подрубрика', '')}). "
            f"Адрес: {row.get('Адрес', '')}. "
            f"Время работы: {row.get('Время работы', '')}."
        )

        # Метаданные для фильтрации по расстоянию
        try:
            lat = float(row.get("Широта", "0") or 0)
            lon = float(row.get("Долгота", "0") or 0)
        except ValueError:
            lat, lon = 0.0, 0.0

        metadata = {
            "Название": row.get("Название"),
            "Широта": lat,
            "Долгота": lon,
            "Адрес": row.get("Адрес"),
            "Время работы": row.get("Время работы"),
            "Категория": row.get("Подрубрика"),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    print(f"✅ Индексация {len(df)} записей завершена. База сохранена в {VECTOR_DB_PATH}.")

#МАППИНГ ПОДРУБРИК НА КАТЕГОРИИ (по реальным данным)
SUBRUBRIKA_TO_INTENT = {
    # coffee
    "Кофейни": "coffee",
    "Кафе": "coffee",
    "Чайные": "coffee",
    "Кондитерские": "coffee",

    # restaurant
    "Рестораны": "restaurant",
    "Фастфуд": "restaurant",
    "Пиццерии": "restaurant",
    "Суши-бары": "restaurant",
    "Бары": "restaurant",
    "Пабы": "restaurant",
    "Столовые": "restaurant",

    # clinic
    "Медицинские центры": "clinic",
    "Поликлиники": "clinic",
    "Больницы": "clinic",
    "Диагностические центры": "clinic",
    "Лаборатории": "clinic",

    # dentist
    "Стоматологии": "dentist",
    "Стоматологические клиники": "dentist",
    "Ортодонты": "dentist",

    # pharmacy
    "Аптеки": "pharmacy",
    "Лекарственные средства": "pharmacy",
    "Витамины": "pharmacy",

    # bank
    "Банки": "bank",
    "Банкоматы": "bank",
    "Ломбарды": "bank",
    "Обмен валют": "bank",
    "Финансовые услуги": "bank",

    # fitness
    "Фитнес-центры": "fitness",
    "Тренажёрные залы": "fitness",
    "Йога-студии": "fitness",
    "Спортивные клубы": "fitness",
    "SPA": "fitness",

    # beauty
    "Салоны красоты": "beauty",
    "Парикмахерские": "beauty",
    "Маникюр": "beauty",
    "Педикюр": "beauty",
    "Косметология": "beauty",
    "Барбершопы": "beauty",

    # shopping
    "Торговые центры": "shopping",
    "Магазины": "shopping",
    "Бутики": "shopping",
    "Супермаркеты": "shopping",
    "Продуктовые": "shopping",

    # cinema
    "Кинотеатры": "cinema",
    "Развлекательные центры": "cinema",
    "Театры": "cinema",
}



def search_poi_with_filter(
    query: str,
    user_lat: float,
    user_lon: float,
    max_distance_km: float = 5.0,
    top_k: int = 500,
) -> str:
    """
    Поиск POI:
    1) Определяем интент из текста запроса (coffee, cinema, restaurant и т.д.).
    2) Берём релевантные документы из Chroma.
    3) Фильтруем по расстоянию и по категории (SUBRUBRIKA_TO_INTENT).
    4) Возвращаем красиво отформатированный текст для Telegram.
    """

    if not HAS_RAG_LIBS or not os.path.exists(VECTOR_DB_PATH):
        return (
            f"[ЗАГЛУШКА] База не найдена или Chroma не инициализирован.\n"
            f"Запрос: {query}\n"
            f"Локация: ({user_lat}, {user_lon})"
        )

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )
    relevant_docs = db.similarity_search_with_score(query, k=top_k)

    # Определяем интент по запросу
    q_lower = (query or "").lower()
    # Токены запроса: ["мне", "нужнен", "кино", "недалеко"]
    query_tokens = re.findall(r"\w+", q_lower)

    intent_keywords = {
        "coffee":     ["кофе", "кафе", "coffee", "капучино", "латте"],
        "restaurant": ["ресторан", "ресторанчик", "фастфуд", "бургер", "пицца", "суши", "донер"],
        "dentist":    ["стоматолог", "стоматология", "зуб", "зубы", "ортодонт", "пломба"],
        "pharmacy":   ["аптека", "лекарство", "лекарства", "витамин", "витамины"],
        "clinic":     ["клиника", "больница", "поликлиника", "медцентр", "врач"],
        "bank":       ["банк", "банкомат", "ломбард", "кредит"],
        "beauty":     ["салон", "салоны", "маникюр", "педикюр", "парикмахерская", "барбершоп"],
        "fitness":    ["фитнес", "спортзал", "спорт", "йога", "gym", "тренажерный"],
        "shopping":   ["магазин", "маркет", "торговый центр", "бутик", "супермаркет"],
        "cinema":     ["кино", "кинотеатр", "cinema", "фильм"],
    }

    def matches_intent_in_query(intent_words, tokens, full_text_lower) -> bool:
        """
        Для однословных ключей — ищем среди токенов.
        Для многословных ("торговый центр") — ищем подстроку в полном тексте.
        """
        for kw in intent_words:
            kw = kw.lower()
            if " " in kw:
                # фраза
                if kw in full_text_lower:
                    return True
            else:
                # одиночное слово
                if kw in tokens:
                    return True
        return False

    matched_intent = None
    for intent, words in intent_keywords.items():
        if matches_intent_in_query(words, query_tokens, q_lower):
            matched_intent = intent
            break

    # Определяем категорию POI по Подрубрике
    def detect_poi_intent(doc):
        """
        Берём doc.metadata["Категория"], в которой лежит Подрубрика.
        Пример: 'Банкетные залы, Бары, Рестораны'
        Разбиваем по запятым и пытаемся сматчить каждую часть с SUBRUBRIKA_TO_INTENT.
        """
        cat_raw = (doc.metadata.get("Категория") or "").strip().lower()
        if not cat_raw:
            return None

        # 'банкетные залы, бары, рестораны' -> ['банкетные залы', 'бары', 'рестораны']
        parts = [c.strip() for c in cat_raw.split(",") if c.strip()]

        for part in parts:
            for subr, intent in SUBRUBRIKA_TO_INTENT.items():
                s = subr.lower()
                # немного "мягкое" сравнение
                if s in part or part in s:
                    return intent

        return None

    # Фильтрация документов
    # будем хранить: (distance_km, name, category, address, time)
    pois: list[tuple[float, str, str, str, str]] = []

    for doc, score in relevant_docs:
        try:
            poi_lat = float(doc.metadata.get("Широта", 0) or 0)
            poi_lon = float(doc.metadata.get("Долгота", 0) or 0)
        except ValueError:
            continue

        distance_km = haversine_distance(user_lat, user_lon, poi_lat, poi_lon)
        if distance_km > max_distance_km:
            continue

        poi_intent = detect_poi_intent(doc)

        # если интент из запроса определён — фильтруем только по нему
        if matched_intent is not None and poi_intent is not None:
            if poi_intent != matched_intent:
                continue
        elif matched_intent is not None and poi_intent is None:
            # строгий режим: если пользователь явно просил кино, а у POI нет категории — пропускаем
            continue

        name = doc.metadata.get("Название") or "Без названия"
        cat = doc.metadata.get("Категория") or ""
        addr = doc.metadata.get("Адрес") or "Адрес не указан"
        time = doc.metadata.get("Время работы") or "Время работы не указано"

        pois.append((distance_km, name, cat, addr, time))

    # Обработка случая, когда ничего не найдено 
    if not pois:
        if matched_intent:
            return (
                f"Я искал места категории '{matched_intent}' "
                f"в радиусе {max_distance_km:.1f} км, но ничего не нашёл.\n"
                "Попробуй изменить запрос или увеличить радиус 🙂"
            )
        else:
            return (
                f"Ничего не нашлось в радиусе {max_distance_km:.1f} км.\n"
                "Попробуй переформулировать запрос 🙂"
            )

    # Сортируем и красиво форматируем для телеги
    pois.sort(key=lambda x: x[0])
    top_pois = pois[:5]

    total_found = len(pois)
    header_lines = []

    header_lines.append(f"✅ Найдено {total_found} мест в радиусе {max_distance_km:.1f} км.")
    if matched_intent:
        header_lines.append(f"Категория: {matched_intent}")
    header = "\n".join(header_lines)

    lines = [header, ""]  # пустая строка после заголовка

    for idx, (dist, name, cat, addr, time) in enumerate(top_pois, start=1):
        block = [
            f"{idx}) {name}",
            f"   📍 {addr}",
            f"   📏 {dist:.2f} км",
            f"   🏷 {cat}",
            f"   🕒 {time}",
        ]
        lines.append("\n".join(block))
        lines.append("")  # пустая строка между карточками

    pretty_text = "\n".join(lines).strip()
    return pretty_text

def get_final_recommendation(user_query: str, user_lat: float, user_lon: float) -> str:
    """
    Упрощённая версия для Telegram-бота:
    - делаем семантический поиск по базе POI (Chroma + Haversine),
    - возвращаем список найденных мест напрямую, без вызова LLM.
    """
    context = search_poi_with_filter(user_query, user_lat, user_lon)
    return context


# Только для локального тестирования
# if __name__ == "__main__":
    test_query = "мне нужна аптека недалеко"
    user_lat = 51.1095   
    user_lon = 71.5260

    print("--- ТЕСТ: ПОИСК POI ---")
    print(search_poi_with_filter(test_query, user_lat, user_lon))

    # РАССКОМЕНТИРОВАТЬ ПОСЛЕ СОЗДАНИЯ БДШКИ!!!
    # print("\n--- ТЕСТ: ИНДЕКСАЦИЯ ---")
    # index_data_to_chroma(CLEAN_DATA_PATH)

