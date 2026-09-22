from datasets import load_dataset
from pathlib import Path
import json
import re

# Делаем имена авторов лаконичными для гибкого поиска совпадений
NEW_ARTISTS_DATASETS = {
    #"Rembrandt": {"repo": "jaddai/openbrush-rembrandt", "search_name": "Rembrandt"},
    "Renoir": {"repo": "jaddai/openbrush-renoir", "search_name": "Renoir"},
}

IMAGES_PER_ARTIST = 20
OUTPUT_DIR = Path("OpenBrush_Selected")

KNOWN_FIELDS = [
    "id", "artist", "style", "genre", "tags", "subject", "action",
    "setting", "mood", "style_description", "lighting", "color",
    "composition", "caption_full", "source_file",
]

def safe_filename(value):
    return re.sub(r'[<>:"/\\|?*]', "_", str(value))

def format_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def save_metadata_txt(item, txt_path):
    lines = []
    for field in KNOWN_FIELDS:
        if field in item:
            lines.append(f"{field}: {format_metadata_value(item[field])}")

    additional_fields = [
        key for key in item.keys()
        if key not in KNOWN_FIELDS and key != "image"
    ]
    if additional_fields:
        lines.append("")
        lines.append("Additional metadata:")
        for field in sorted(additional_fields):
            lines.append(f"{field}: {format_metadata_value(item[field])}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

def decode_image(image_data):
    from PIL import Image as PILImage
    from io import BytesIO

    if image_data is None:
        return None
    if isinstance(image_data, PILImage.Image):
        return image_data
    if isinstance(image_data, bytes):
        return PILImage.open(BytesIO(image_data))
    if isinstance(image_data, dict):
        image_bytes = image_data.get("bytes")
        if image_bytes:
            return PILImage.open(BytesIO(image_bytes))
        image_path = image_data.get("path")
        if image_path:
            return PILImage.open(image_path)
    raise TypeError(f"Неизвестный формат image: {type(image_data)}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Загрузка индивидуальных коллекций (Исправленный поиск по имени)")
    print("=" * 70)

    for folder_name, info in NEW_ARTISTS_DATASETS.items():
        artist_dir = OUTPUT_DIR / folder_name
        artist_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Загрузка] Подключение к репозиторию {info['repo']}...")
        
        try:
            dataset = load_dataset(info["repo"], split="train", streaming=True)
        except Exception as e:
            print(f" -> Ошибка подключения к репозиторию {info['repo']}: {e}")
            continue
        
        count = 0
        for item in dataset:
            # Гибкая проверка: ищем ключевое слово (например, 'Renoir') в поле автора
            artist_field = str(item.get("artist", ""))
            if info["search_name"].lower() not in artist_field.lower():
                continue
                
            count += 1
            item_id = safe_filename(item.get("id", count))
            filename_base = f"{count:02d}_{item_id}"
            image_path = artist_dir / f"{filename_base}.png"
            txt_path = artist_dir / f"{filename_base}.txt"
            
            # Если файлы уже существуют локально — переходим к следующему
            if image_path.exists() and txt_path.exists():
                if count >= IMAGES_PER_ARTIST:
                    break
                continue

            try:
                image = decode_image(item.get("image"))
                if image is None:
                    continue
                
                image.save(image_path, format="PNG")
                save_metadata_txt(item, txt_path)
                
                print(f" -> Сохранено [{count}/{IMAGES_PER_ARTIST}]: {image_path.name}")
                
            except Exception as e:
                print(f" -> Ошибка сохранения файла {filename_base}: {e}")
                count -= 1
                
            if count >= IMAGES_PER_ARTIST:
                break
                
        print(f"[Готово] Для {folder_name} успешно сохранено картинок: {count}/{IMAGES_PER_ARTIST}")

    print("\n" + "=" * 70)
    print(f"Все операции завершены! Итоговые датасеты в папке: {OUTPUT_DIR.resolve()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
