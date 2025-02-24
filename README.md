# TPH_FEFU
Отборочный кейс ТПШ 

кейс NLP анализатор

## **ВАЖНО**
Окончательная версия модели находится не в коммитах ветки, а в релизе с тегом Final. (https://github.com/Jaya-varmen/TPH_FEFU/releases/tag/Final)
Модель необходимо распаковать и следовать инструкции по запуске в ветке telegram_bot (readme файл)


Этот репозиторий содержит обученную модель для анализа настроений текста, основанную на `roberta-base`.  
Модель классифицирует текст на 4 категории:
- **Грустный**
- **Негативный**
- **Радостный**
- **Нейтральный**

---

## **Установка зависимостей**
Перед использованием модели установите необходимые библиотеки:
```bash
pip install transformers torch numpy
```
