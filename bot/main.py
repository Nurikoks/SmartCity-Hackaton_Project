import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv

# грузим переменные окружения из .env
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я ИИ-помощник по прогулкам в Астане 🌆\n"
        "Напиши, где ты и что хочешь сделать (например: 'я у Mega Silk Way, хочу прогуляться 30–60 минут')."
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.reply_text(
        f"Ты написал(а): {user_text}\n\n"
        "Скоро я начну подбирать тебе места рядом 😉"
    )


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не найден. Добавь его в файл .env")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print("Бот запущен. Нажми Ctrl+C чтобы остановить.")
    app.run_polling()


if __name__ == "__main__":
    main()
