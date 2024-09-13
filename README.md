# Задача

В рамках хакатона необходимо разработать систему ответа на вопросы по нормативно-правовым актам Ханты-Мансийского автономного округа. Такое решение позволит создать универсального юридического помощника, который разбирается в вопросах ХМАО. Нам был дан корпус, содержащий различные НПА, а также некоторое количество вопросов для валидации нашей системы.

Основные трудности при создании такого решения - домен задачи, а также русский язык, из-за чего некоторые модели для создания векторных баз данных и LLM могут допускать критические ошибки. 

В рамках решения мы постарались найти лучшие модули для нашей RAG системы.

# Описание решение

В нашем решении используется:

**База данных:** FAISS

**Embedder:** E5-Large с чанками по 256 

**LLM:** Qwen-2-7B

**Dev:** FastAPI, aiogram, Docker, Celery, Redis


![Без названия-3](https://github.com/user-attachments/assets/6f83d3f8-77fc-4b95-b437-96741e67fdac)

# Установка и запуск

Следуйте следующим шагам, чтобы запустить проект:

1. **Создайте .env файл**

   In the copilot directory, create a .env file based on the example provided in copilot/.env_example.

2. **Запустите Docker Compose**

Из директории проекта запустите команду:

    docker compose up


# Описание репозитория

В репозитории присутствует несколько веток: main, api, docker, experiments, faiss, gemini и tg_bot.
Итоговая структура: 

*main = api + docker + faiss + gemini* - код API для взаимодействия с RAGом и сам RAG

*experiments* - код для быстрого запуска экспериментов

*tg_bot* - простой сервис с UI в виде телеграм бота

# Описание main

**ml** - тут находится создание модели

**dialog_processor** - тут находится код API общения с LLM

**database** - наше векторная база данных

**config** - загрузка всех конфигов

# Полезные ссылки 

[1] Retrieval-Augmented Generation for Large Language Models: A Survey - https://arxiv.org/abs/2312.10997

[2] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks - https://arxiv.org/abs/2005.11401

[3] A Tale of Trust and Accuracy: Base vs. Instruct LLMs in RAG Systems - https://arxiv.org/abs/2406.14972v1

[4] Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering - https://arxiv.org/abs/2210.02627v1

[5] Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search and Hybrid Query-Based Retrievers - https://arxiv.org/abs/2404.07220v2

[6] Langchain Github - https://github.com/langchain-ai/langchain

