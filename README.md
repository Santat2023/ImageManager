# ImageManager

## Описание

**ImageManager** — это приложение на Python с интерфейсом Streamlit для управления коллекциями изображений, их загрузки, поиска по описанию и хранения в MinIO (S3) и ChromaDB.

---

## Быстрый старт

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/yourusername/ImageManager.git
cd ImageManager
```

### 2. Создайте виртуальное окружение

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Установите зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Запустите сервисы через Docker

Убедитесь, что установлен Docker. Запустите MinIO и ChromaDB:

```bash
docker-compose up -d
```

### 5. Запустите приложение

```bash
streamlit run load_images_ui_v2.py
```

---

## Использование

1. **Управление коллекциями** — создавайте и удаляйте коллекции изображений.
2. **Загрузка изображений** — укажите папку с изображениями, выберите коллекцию и загрузите файлы.
3. **Поиск изображений** — ищите изображения по текстовому описанию.

---

## Настройки

- **MinIO** запускается на `localhost:9000` (логин/пароль: admin/admin123).
- **ChromaDB** запускается на `localhost:8000`.
- Все параметры можно изменить в файле `load_images_ui_v2.py`.

---

## Требования

- Python 3.8+
- Docker
- [requirements.txt](requirements.txt) содержит все необходимые библиотеки.

---

## Примечания

- Для работы с CUDA требуется совместимая видеокарта и драйверы.
- Папки `chroma-data` и `minio-data` создаются автоматически для хранения данных.

---

## Лицензия

MIT