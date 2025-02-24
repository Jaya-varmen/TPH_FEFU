import os
import torch
from transformers import pipeline
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, CallbackContext

# Указываем Telegram Token бота
TOKEN = "8194961676:AAENyYv0SV_oi31MzAGAsd8XRSp5tMyFE_M"

# Указываем путь к модели
MODEL_PATH =  r"C:\Users\steks\Desktop\ТПШ\Telegram Bot\saved_model"

# Загружаем модель
classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

# Словарь меток
label_map = {
    0: "грустный",
    1: "негативный",
    2: "радостный",
    3: "нейтральный"
}

# Функция обработки сообщений
async def analyze_sentiment(update: Update, context: CallbackContext):
    text = update.message.text  # Полученпие текста от пользователя
    result = classifier(text)[0]  # Анализ настроения
    label_index = int(result["label"].split("_")[-1])  # Преобразование LABEL_2 -> 2
    sentiment = label_map[label_index]  # Получаем понятную метку

    response = f"*Настроение:* {sentiment}" #вывод пользователю
    
    await update.message.reply_text(response, parse_mode="Markdown")

# Функция обработки команды /start
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Отправь мне текст, и я скажу, какое у него настроение")

# Запускаем бота
def main():
    app = Application.builder().token(TOKEN).build()

    # Добавляем обработчики команд и сообщений
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_sentiment))

    print("Бот запущен и ожидает сообщений")
    app.run_polling()

# Запуск
if __name__ == "__main__":
    main()
