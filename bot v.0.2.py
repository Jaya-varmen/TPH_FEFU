import os
import torch
import torch.nn.functional as F
from transformers import pipeline
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, CallbackContext

# Указываем Telegram Token бота
TOKEN = "тут токен"

# Указываем путь к модели
MODEL_PATH = r"путь тут"

# Загружаем модель
classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, return_all_scores=True)  # return_all_scores=True возвращает вероятности всех классов

# Словарь меток
label_map = {
    0: "грустный",
    1: "негативный",
    2: "радостный",
    3: "нейтральный"
}

# Функция обработки сообщений
async def analyze_sentiment(update: Update, context: CallbackContext):
    text = update.message.text  # Получение текста от пользователя
    results = classifier(text)[0]  # Получаем все вероятности

    # Преобразуем логиты в вероятности с помощью softmax
    probabilities = F.softmax(torch.tensor([res["score"] for res in results]), dim=0).tolist()

    # Формируем сообщение с вероятностями каждого класса
    response = f"📊 *Анализ настроения:*\n\n📝 *Текст:* {text}\n\n"
    for i, res in enumerate(results):
        sentiment = label_map[i]
        probability = probabilities[i] * 100  # Конвертируем в проценты
        response += f"{sentiment}: {probability:.2f}%\n"

    # Определяем наиболее вероятный класс
    best_label_index = probabilities.index(max(probabilities))
    best_sentiment = label_map[best_label_index]

    response += f"\n🎯 *Предсказанное настроение:* {best_sentiment}"

    # Отправляем сообщение пользователю
    await update.message.reply_text(response, parse_mode="Markdown")

# Функция обработки команды /start
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Отправь мне текст, и я скажу, какое у него настроение 😊")

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
