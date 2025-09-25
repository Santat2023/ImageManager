import os
import uuid
import streamlit as st
import torch
import clip
import boto3
import chromadb
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
from streamlit import rerun

# 
# Настройки S3 и ChromaDB
# 
S3_ENDPOINT = "http://localhost:9000"
S3_BUCKET = "images"
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"

# подключение к Chroma и S3
client = chromadb.HttpClient(host="localhost", port=8000)
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)
try:
    s3.create_bucket(Bucket=S3_BUCKET)
    print(f"Bucket '{S3_BUCKET}' created (or already exists).")
except Exception as e:
    print(e)

# модели CLIP и BLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map={"": device}
)

# 
# Вспомогательные функции
# 
def list_collections():
    return [c.name for c in client.list_collections()]

def delete_collection(name):
    try:
        collection = client.get_collection(name)
        results = collection.get(include=["metadatas"])
        if results and "ids" in results:
            for _id, meta in zip(results["ids"], results["metadatas"]):
                key = f"{_id}_{meta['filename']}"
                try:
                    s3.delete_object(Bucket=S3_BUCKET, Key=key)
                except Exception as e:
                    print(f"⚠️ Ошибка удаления {key} из S3: {e}")
        client.delete_collection(name)
        return f"Коллекция {name} и её изображения удалены ✅"
    except Exception as e:
        return f"Ошибка при удалении коллекции {name}: {e}"

def create_collection(name):
    client.create_collection(name)
    return f"Коллекция {name} создана ✅"

def resize_to_min_side(image_path: str, target_size: int = 250) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w <= h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    tmp_path = f"{image_path}_resized.jpg"
    img.save(tmp_path, format="JPEG", quality=95)
    return tmp_path

def describe_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(
            **inputs,
            max_length=50,
            min_length=10,
            num_beams=10,
            repetition_penalty=1.2,
            top_k=50,
            top_p=0.9,
        )
    return blip_processor.decode(out[0], skip_special_tokens=True)

def embed_text(text: str):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        embedding = clip_model.encode_text(tokens)
        return embedding.cpu().numpy().tolist()

def upload_to_s3(file_path: str, key: str):
    s3.upload_file(file_path, S3_BUCKET, key)
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"

def process_directory(directory: str, resolution: int, collection_name: str):
    collection = client.get_collection(collection_name)
    files = [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    progress = st.progress(0)
    total = len(files)

    for idx, filename in enumerate(files, 1):
        path = os.path.join(directory, filename)
        image_id = str(uuid.uuid4())

        # ресайз
        path = resize_to_min_side(path, resolution)

        # 1) описание
        description = describe_image(path)

        # 2) эмбеддинг
        embedding = embed_text(description)

        # 3) в ChromaDB
        collection.add(
            ids=[image_id],
            documents=[description],
            metadatas=[{"filename": filename, "collection": collection_name}],
            embeddings=embedding
        )

        # 4) в S3
        upload_to_s3(path, f"{image_id}_{filename}")

        # обновление прогресса
        progress.progress(idx / total)

def search_images(query: str, collection_name: str, top_k: int = 3):
    collection = client.get_collection(collection_name)
    embedding = embed_text(query)
    res = collection.query(query_embeddings=embedding, n_results=top_k)
    return res

def load_image_from_s3(key: str):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read()))

# 
# Streamlit UI
# 
st.set_page_config(page_title="Image Manager", layout="wide")
st.title("📦 Image Manager")

tabs = st.tabs(["🗑 Управление коллекциями", "⬆️ Загрузка изображений", "🔎 Поиск изображений"])

# Tab 1: Коллекции
with tabs[0]:
    st.subheader("Управление коллекциями")
    collections = list_collections()
    st.write("Существующие коллекции:", collections)

    col1, col2 = st.columns(2)

    with col1:
        new_name = st.text_input("Имя новой коллекции")
        if st.button("Создать коллекцию"):
            if new_name:
                with st.spinner("Создаём коллекцию..."):
                    st.success(create_collection(new_name))
                rerun()
            else:
                st.error("Введите имя коллекции!")

    with col2:
        if collections:
            to_delete = st.selectbox("Удалить коллекцию", collections)
            if st.button("Удалить"):
                with st.spinner("Удаляем коллекцию и связанные изображения..."):
                    st.warning(delete_collection(to_delete))
                rerun()
        else:
            st.info("Нет коллекций для удаления")

# Tab 2: Загрузка
with tabs[1]:
    st.subheader("Загрузка изображений")

    collections = list_collections()
    if not collections:
        st.error("⚠️ Нет коллекций! Сначала создайте хотя бы одну во вкладке 'Управление коллекциями'.")
    else:
        directory = st.text_input("Путь к папке с картинками")
        resolution = st.number_input("Минимальная сторона (px)", min_value=64, max_value=1024, value=250)
        selected_collection = st.selectbox("Коллекция для загрузки", collections)

        if st.button("Загрузить"):
            if directory and os.path.exists(directory):
                with st.spinner("Загружаем изображения..."):
                    process_directory(directory, resolution, selected_collection)
                st.success(f"✅ Изображения загружены в коллекцию {selected_collection}")
            else:
                st.error("Путь к папке неверный!")

# Tab 3: Поиск
with tabs[2]:
    st.subheader("Поиск изображений")
    collections = list_collections()

    if not collections:
        st.error("⚠️ Нет коллекций! Сначала создайте хотя бы одну.")
    else:
        selected_collection = st.selectbox("Коллекция для поиска", collections)
        query = st.text_input("Введите поисковый запрос")
        if st.button("Найти"):
            if query:
                with st.spinner("Ищем изображения..."):
                    results = search_images(query, selected_collection, top_k=3)
                for doc, meta, dist, _id in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    results["ids"][0],
                ):
                    key = f"{_id}_{meta['filename']}"
                    img = load_image_from_s3(key)
                    st.image(img, caption=f"{doc} (📏 {dist:.2f})")
            else:
                st.error("Введите запрос!")
