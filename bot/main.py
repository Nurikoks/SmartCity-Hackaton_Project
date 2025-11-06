import os
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv

from index import get_final_recommendation

# Загружаем токен из .env
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN or len(TOKEN) < 30:
    raise RuntimeError("❌ TELEGRAM_TOKEN не найден или слишком короткий. Проверь .env файл.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [KeyboardButton("📍 Отправить геолокацию", request_location=True)]
    ]
    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True)

    await update.message.reply_text(
        "Привет! Я ИИ-помощник по прогулкам в Астане 🌆\n\n"
        "1️⃣ Нажми кнопку, чтобы отправить свою локацию\n"
        "2️⃣ Напиши, что ты хочешь (например: 'тихое кафе с кофе на 1 час').",
        reply_markup=reply_markup,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Я помогаю находить интересные места рядом с тобой.\n\n"
        "📍 Сначала отправь геолокацию (кнопка внизу).\n"
        "✍️ Потом напиши запрос, например:\n"
        "• 'я у Mega Silk Way, хочу прогуляться 30–60 минут'\n"
        "• 'мне нужно тихое кафе с розеткой рядом'\n"
        "• 'я с ребёнком, что есть поблизости?'\n"
    )


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Сохраняем координаты пользователя в context.user_data,
    чтобы потом использовать их в RAG.
    """
    loc = update.message.location
    if not loc:
        return

    lat, lon = loc.latitude, loc.longitude
    context.user_data["lat"] = lat
    context.user_data["lon"] = lon

    await update.message.reply_text(
        f"Отлично! Я запомнил твою геолокацию 📍! \n"
        f"lat={lat:.4f}, lon={lon:.4f}\n\n"
        "Теперь напиши, что ты хочешь сделать (кафе, прогулка, музей и т.д.)."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Основной вход для RAG:
    берём текст + координаты → вызываем get_final_recommendation.
    """
    query = update.message.text

    lat = context.user_data.get("lat")
    lon = context.user_data.get("lon")

    # Если геолокацию ещё не отправили
    if lat is None or lon is None:
        await update.message.reply_text(
            "Сначала отправь геолокацию через кнопку «📍 Отправить геолокацию», "
            "а потом напиши свой запрос"
        )
        return

    try:
        # Синхронная функция из index.py — просто вызываем её
        answer = get_final_recommendation(query, lat, lon)
    except Exception as e:
        # На всякий случай логируем ошибку в консоль
        print(f"[ERROR in get_final_recommendation]: {e}")
        answer = "Упс, что-то пошло не так при поиске мест. Попробуй ещё раз чуть позже 🙏"

    await update.message.reply_text(answer)

    # 👑 НОВАЯ СТРОКА ДЛЯ ПРИВАТНОСТИ: 
    # Очищаем геолокацию, как только рекомендация выдана.
    if "lat" in context.user_data:
        del context.user_data["lat"]
    if "lon" in context.user_data:
        del context.user_data["lon"]
    
    # Можно добавить необязательное сообщение для пользователя
    await update.message.reply_text("✨ Твоя геолокация удалена из памяти бота в целях приватности.")


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не найден. Добавь его в .env")

    app = ApplicationBuilder().token(TOKEN).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    # Геолокация
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))

    # Текстовые сообщения (всё, что не команда)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
