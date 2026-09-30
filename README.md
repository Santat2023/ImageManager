# ImageManager

## Описание

**ImageManager** — это приложение на Python с интерфейсом Tkinker для управления коллекциями изображений, их загрузки, поиска по описанию и хранения в MinIO (S3) и Qdrant.

---

## Быстрый старт

### Вариант 1. Готовая сборка (Portable)

1. Скачайте архив [ImageManager_v0.1.zip](https://github.com/Santat2023/ImageManager/releases/tag/v0.0.2).
2. Скачайте архив [MinioQadrantPortable.zip](https://github.com/Santat2023/ImageManager/releases/tag/v0.0.2).
3. Скачайте архив [models.zip](https://github.com/Santat2023/ImageManager/releases/tag/v0.0.2).
4. Распакуйте все удобную папку.
5. Запустите Minio и Qdrant через файл **`start.bat`** в MinioQadrantPortable.
6. Запустите ImageManager_V5.exe

---

### Вариант 2. Запуск из исходников


### 1. Клонируйте репозиторий

```bash
git clone https://github.com/yourusername/ImageManager.git
cd ImageManager
```

### 2. Создайте виртуальное окружение


**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Установите зависимости

```bash
pip install --upgrade pip
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

### 4. Запустите сервисы через Docker

Убедитесь, что установлен Docker. Запустите MinIO и ChromaDB:

```bash
docker-compose up -d
```

### 5. Запустите приложение

```bash
python ImageManager/load_images_ui_tkinker_qdrant.py
```

---

## Использование

1. **Управление коллекциями** — создавайте и удаляйте коллекции изображений.
2. **Загрузка изображений** — укажите папку с изображениями, выберите коллекцию и загрузите файлы.
3. **Поиск изображений** — ищите изображения по текстовому описанию.

---

## Настройки

- **MinIO** запускается на `localhost:9000` (логин/пароль: admin/admin123).
- **Qdrant** запускается на `localhost:6333`.
- Все параметры можно изменить в файле `load_images_ui_tkinker_qdrant.py`.

---

## Требования

- Python 3.8+
- Docker
- [requirements.txt](requirements.txt) содержит все необходимые библиотеки.

---

## Примечания

- Для работы с CUDA требуется совместимая видеокарта и драйверы.
- Папки `qdrant_storage` и `minio-data` создаются автоматически для хранения данных.

---

## Лицензия

MIT