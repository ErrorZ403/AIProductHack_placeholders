# Описание ветки

В данной ветке находится код для быстрого запуска экспериментов, поделенных на три части.

**embeddings_collector.py** - данный скрипт запускает код сбора векторной базы данных по документам и принимает следующие параметры

- --doc_path - путь до документа/документов
- --model_name - название модели эмбеддера с HuggingFace
- --save_path - путь сохранения базы данных
- --chunk_size - размер чанка для деления документа
- --chunk_overlap - размер пересечения между соседними чанками
- --cuda - использовать GPU или нет
- --batch_size - размер батча в обработке эмбеддера

**question_answering.py** - данный скрипт запускает код сбора ответов LLM на вопросы по найденному контексту и принимает следующие параметры

- --doc_path - путь до документа/документов
- --dataset_path - путь до валидационного датасета
- --model_name - название модели эмбеддера с HuggingFace
- --emb_path - путь до векторной БД
- --model_name_llm - название модели LLM с HuggingFace
- --path_to_save - путь сохранения ответов на вопросы
- --cuda - использовать GPU или нет
- --batch_size - размер батча в обработке эмбеддера
- --bm25 - использовать BM25 или нет
- --rerank - использовать реранкер или нет

**rag_eval.py** - данный скрипт запускает подсчет метрик с помощью RAGAS и принимает следующие параметры

- --dataset_path - путь до датасета с ответами
- --model_name - название модели эмбеддера с HuggingFace
- --model_name_llm - название модели LLM с HuggingFace
- --cuda - использовать GPU или нет
- --batch_size - размер батча в обработке эмбеддера
