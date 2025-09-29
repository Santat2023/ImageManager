import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import uuid
import torch
import clip
import boto3
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import botocore 

#
# Настройки S3 и Qdrant
#
S3_ENDPOINT = "http://localhost:9000"
S3_BUCKET = "images"
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"

qdrant = QdrantClient(host="localhost", port=6333)
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)
try:
    s3.create_bucket(Bucket=S3_BUCKET)
    print(f"Bucket {S3_BUCKET} created")
except botocore.exceptions.ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
        print(f"Bucket {S3_BUCKET} already exists, continue...")
    else:
        raise

#
# Модели CLIP и BLIP
#
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map={"": device}
)

#
# Вспомогательные функции
#
def list_collections():
    cols = qdrant.get_collections().collections
    return [c.name for c in cols]

def delete_collection(name):
    try:
        qdrant.delete_collection(name)
        return f"Коллекция {name} удалена ✅"
    except Exception as e:
        return f"Ошибка при удалении {name}: {e}"

def create_collection(name, dim=512):
    try:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        return f"Коллекция {name} создана ✅"
    except Exception as e:
        return f"Ошибка при создании {name}: {e}"

def resize_to_min_side(image_path: str, target_size: int = 250) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w <= h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def describe_image(image: Image.Image) -> str:
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
        return embedding.cpu().numpy()[0]

def upload_to_s3(img: Image.Image, key: str):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    s3.upload_fileobj(buffer, S3_BUCKET, key)
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"

def process_directory(directory: str, resolution: int, collection_name: str, progressbar, app):
    files = [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    total = len(files)
    progressbar["value"] = 0
    progressbar["maximum"] = total
    app.update_idletasks()

    for idx, filename in enumerate(files, 1):
        path = os.path.join(directory, filename)
        image_id = str(uuid.uuid4())

        img = resize_to_min_side(path, resolution)
        description = describe_image(img)
        embedding = embed_text(description)

        qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=image_id,
                    vector=embedding.tolist(),
                    payload={"filename": filename, "description": description}
                )
            ]
        )

        upload_to_s3(img, f"{image_id}_{filename}")

        progressbar["value"] = idx
        app.update_idletasks()

def search_images(query: str, collection_name: str, top_k: int = 1):
    embedding = embed_text(query)
    res = qdrant.search(
        collection_name=collection_name,
        query_vector=embedding.tolist(),
        limit=top_k
    )
    return res

def load_image_from_s3(key: str):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read()))

#
# UI
#
class ImageManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Manager (Qdrant)")
        self.geometry("900x700")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.tab_collections = ttk.Frame(self.notebook)
        self.tab_upload = ttk.Frame(self.notebook)
        self.tab_search = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_collections, text="🗑 Управление коллекциями")
        self.notebook.add(self.tab_upload, text="⬆️ Загрузка изображений")
        self.notebook.add(self.tab_search, text="🔎 Поиск изображений")

        self.init_collections_tab()
        self.init_upload_tab()
        self.init_search_tab()

    def refresh_collections(self):
        new_list = list_collections()
        self.collections_var.set(new_list)
        self.upload_collection_combo["values"] = new_list
        self.search_collection_combo["values"] = new_list

    # Коллекции
    def init_collections_tab(self):
        frame = self.tab_collections

        ttk.Label(frame, text="Существующие коллекции:").pack(anchor="w", padx=10, pady=5)
        self.collections_var = tk.StringVar(value=list_collections())
        self.collections_listbox = tk.Listbox(frame, listvariable=self.collections_var, height=6)
        self.collections_listbox.pack(fill="x", padx=10)

        ttk.Label(frame, text="Имя новой коллекции:").pack(anchor="w", padx=10, pady=5)
        self.new_collection_entry = ttk.Entry(frame)
        self.new_collection_entry.pack(fill="x", padx=10)

        ttk.Button(frame, text="Создать коллекцию", command=self.create_collection).pack(pady=5, padx=10, anchor="w")
        ttk.Button(frame, text="Удалить выбранную", command=self.delete_collection).pack(pady=5, padx=10, anchor="w")

    def create_collection(self):
        name = self.new_collection_entry.get()
        if name:
            msg = create_collection(name, dim=512)
            messagebox.showinfo("Создание коллекции", msg)
            self.refresh_collections()
        else:
            messagebox.showerror("Ошибка", "Введите имя коллекции!")

    def delete_collection(self):
        selection = self.collections_listbox.curselection()
        if selection:
            name = self.collections_listbox.get(selection[0])
            msg = delete_collection(name)
            messagebox.showinfo("Удаление коллекции", msg)
            self.refresh_collections()
        else:
            messagebox.showerror("Ошибка", "Выберите коллекцию!")

    # Загрузка
    def init_upload_tab(self):
        frame = self.tab_upload

        ttk.Label(frame, text="Папка с картинками:").pack(anchor="w", padx=10, pady=5)
        self.dir_entry = ttk.Entry(frame)
        self.dir_entry.pack(fill="x", padx=10)
        ttk.Button(frame, text="Выбрать папку", command=self.choose_directory).pack(pady=5, padx=10, anchor="w")

        ttk.Label(frame, text="Минимальная сторона (px):").pack(anchor="w", padx=10, pady=5)
        self.resolution_spin = ttk.Spinbox(frame, from_=64, to=1024, increment=1)
        self.resolution_spin.set(250)
        self.resolution_spin.pack(fill="x", padx=10)

        ttk.Label(frame, text="Коллекция для загрузки:").pack(anchor="w", padx=10, pady=5)
        self.upload_collection_var = tk.StringVar()
        self.upload_collection_combo = ttk.Combobox(frame, textvariable=self.upload_collection_var, values=list_collections())
        self.upload_collection_combo.pack(fill="x", padx=10)

        self.progressbar = ttk.Progressbar(frame, length=400, mode="determinate")
        self.progressbar.pack(pady=10, padx=10)

        ttk.Button(frame, text="Загрузить", command=self.upload_images).pack(pady=10, padx=10, anchor="w")

    def choose_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)

    def upload_images(self):
        directory = self.dir_entry.get()
        resolution = int(self.resolution_spin.get())
        collection = self.upload_collection_var.get()
        if not collection:
            messagebox.showerror("Ошибка", "Выберите коллекцию!")
            return
        if not directory or not os.path.exists(directory):
            messagebox.showerror("Ошибка", "Путь неверный!")
            return
        try:
            process_directory(directory, resolution, collection, self.progressbar, self)
            messagebox.showinfo("Успех", f"Изображения загружены в {collection}")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    # Поиск
    def init_search_tab(self):
        frame = self.tab_search

        ttk.Label(frame, text="Коллекция:").pack(anchor="w", padx=10, pady=5)
        self.search_collection_var = tk.StringVar()
        self.search_collection_combo = ttk.Combobox(frame, textvariable=self.search_collection_var, values=list_collections())
        self.search_collection_combo.pack(fill="x", padx=10)

        ttk.Label(frame, text="Поисковый запрос:").pack(anchor="w", padx=10, pady=5)
        self.query_entry = ttk.Entry(frame)
        self.query_entry.pack(fill="x", padx=10)

        ttk.Button(frame, text="Найти", command=self.search_images).pack(pady=10, padx=10, anchor="w")

        self.results_frame = ttk.Frame(frame)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def search_images(self):
        collection = self.search_collection_var.get()
        query = self.query_entry.get()
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        if not collection or not query:
            messagebox.showerror("Ошибка", "Выберите коллекцию и введите запрос!")
            return
        try:
            results = search_images(query, collection, top_k=1)
            for r in results:
                key = f"{r.id}_{r.payload['filename']}"
                img = load_image_from_s3(key)
                img_tk = ImageTk.PhotoImage(img.resize((250, 250)))
                lbl = ttk.Label(self.results_frame, text=f"{r.payload['description']} (📏 {r.score:.2f})")
                lbl.pack(anchor="w")
                img_lbl = ttk.Label(self.results_frame, image=img_tk)
                img_lbl.image = img_tk
                img_lbl.pack(anchor="w")
        except Exception as e:
            messagebox.showerror("Ошибка поиска", str(e))

if __name__ == "__main__":
    app = ImageManagerApp()
    app.mainloop()
