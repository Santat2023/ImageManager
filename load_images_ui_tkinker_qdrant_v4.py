import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import os
import uuid
import io

import torch
import clip
import boto3
import botocore

from PIL import Image, ImageTk

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)


# ============================================================
# CONFIG
# ============================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

S3_ENDPOINT = "http://localhost:9000"
S3_BUCKET = "images"
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"

CLIP_MODEL = "ViT-B/32"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

DEFAULT_RESOLUTION = 250

# Размер CLIP ViT-B/32
CLIP_DIM = 512


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[SYSTEM] Device: {device}")


# ============================================================
# QDRANT
# ============================================================

qdrant = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    check_compatibility=False
)


# ============================================================
# S3 / MINIO
# ============================================================

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)

try:
    s3.create_bucket(Bucket=S3_BUCKET)
    print(f"[S3] Bucket '{S3_BUCKET}' created")

except botocore.exceptions.ClientError as e:

    error_code = e.response["Error"]["Code"]

    if error_code in (
        "BucketAlreadyOwnedByYou",
        "BucketAlreadyExists"
    ):
        print(f"[S3] Bucket '{S3_BUCKET}' already exists")

    else:
        raise


# ============================================================
# CLIP
# ============================================================

print("[CLIP] Loading model...")

clip_model, clip_preprocess = clip.load(
    CLIP_MODEL,
    device=device
)

clip_model.eval()

print("[CLIP] Model loaded")


# ============================================================
# BLIP
# ============================================================

print("[BLIP] Loading model...")

blip_processor = BlipProcessor.from_pretrained(
    BLIP_MODEL
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL
)

blip_model = blip_model.to(device)
blip_model.eval()

print("[BLIP] Model loaded")


# ============================================================
# QDRANT COLLECTIONS
# ============================================================

def list_collections():
    """
    Возвращает список коллекций Qdrant.
    """

    collections = qdrant.get_collections().collections

    return [
        collection.name
        for collection in collections
    ]


def create_collection(name):
    """
    Создаёт новую коллекцию с двумя named vectors:

        auto   -> автоматическое описание BLIP
        manual -> ручная аннотация из TXT

    Внимание:
    старые коллекции с одним vector необходимо удалить.
    """

    if not name:
        raise ValueError("Имя коллекции не может быть пустым")

    if name in list_collections():
        return f"Коллекция '{name}' уже существует"

    qdrant.create_collection(
        collection_name=name,
        vectors_config={
            "auto": VectorParams(
                size=CLIP_DIM,
                distance=Distance.COSINE
            ),
            "manual": VectorParams(
                size=CLIP_DIM,
                distance=Distance.COSINE
            )
        }
    )

    return (
        f"Коллекция '{name}' создана.\n"
        f"Vectors: auto + manual"
    )


def delete_collection(name):

    qdrant.delete_collection(
        collection_name=name
    )

    return f"Коллекция '{name}' удалена"


# ============================================================
# IMAGE PROCESSING
# ============================================================

def resize_to_min_side(
    image_path: str,
    target_size: int = DEFAULT_RESOLUTION
):
    """
    Изменяет изображение так, чтобы минимальная сторона
    была target_size.
    """

    img = Image.open(image_path).convert("RGB")

    width, height = img.size

    if width <= height:

        new_width = target_size
        new_height = int(
            height * target_size / width
        )

    else:

        new_height = target_size
        new_width = int(
            width * target_size / height
        )

    img = img.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    return img


# ============================================================
# BLIP DESCRIPTION
# ============================================================

def describe_image(image: Image.Image) -> str:

    inputs = blip_processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = blip_model.generate(
            **inputs,
            max_length=50,
            min_length=5,
            num_beams=5,
            repetition_penalty=1.2
        )

    description = blip_processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return description.strip()


# ============================================================
# MANUAL DESCRIPTION
# ============================================================

def find_manual_annotation(
    image_path: str
):
    """
    Ищет TXT рядом с изображением.

    Например:

        woman.jpg
        woman.txt

    Если TXT отсутствует или пустой:
        return None
    """

    base_path = os.path.splitext(
        image_path
    )[0]

    txt_path = base_path + ".txt"

    if not os.path.exists(txt_path):
        return None

    try:

        with open(
            txt_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read().strip()

        if not text:
            return None

        return text

    except Exception as e:

        print(
            f"[WARNING] Cannot read annotation "
            f"{txt_path}: {e}"
        )

        return None


# ============================================================
# CLIP TEXT EMBEDDING
# ============================================================

def embed_text(text: str):

    if not text:
        return None

    # CLIP ViT-B/32 имеет context length 77.
    # clip.tokenize(..., truncate=True) позволяет
    # безопасно обработать слишком длинный текст.

    tokens = clip.tokenize(
        [text],
        truncate=True
    ).to(device)

    with torch.no_grad():

        embedding = clip_model.encode_text(
            tokens
        )

    embedding = embedding.float()

    # Нормализация нужна для cosine similarity.
    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    return embedding.cpu().numpy()[0]


# ============================================================
# S3
# ============================================================

def upload_to_s3(
    image: Image.Image,
    key: str
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95
    )

    buffer.seek(0)

    s3.upload_fileobj(
        buffer,
        S3_BUCKET,
        key
    )


def load_image_from_s3(
    key: str
):

    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=key
    )

    return Image.open(
        io.BytesIO(
            response["Body"].read()
        )
    ).convert("RGB")


# ============================================================
# IMAGE INDEXING
# ============================================================

def process_directory(
    directory,
    resolution,
    collection_name,
    progressbar,
    app
):

    files = [
        file
        for file in os.listdir(directory)
        if file.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp"
            )
        )
    ]

    total = len(files)

    if total == 0:
        raise ValueError(
            "В выбранной директории нет изображений"
        )

    progressbar["value"] = 0
    progressbar["maximum"] = total

    app.update_idletasks()

    print(
        f"[INDEX] Found {total} images"
    )

    for index, filename in enumerate(
        files,
        start=1
    ):

        image_path = os.path.join(
            directory,
            filename
        )

        print(
            f"\n[INDEX] "
            f"{index}/{total}: {filename}"
        )

        try:

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            image = resize_to_min_side(
                image_path,
                resolution
            )

            # ------------------------------------------------
            # AUTOMATIC DESCRIPTION
            # ------------------------------------------------

            auto_description = describe_image(
                image
            )

            print(
                "[BLIP]",
                auto_description
            )

            auto_vector = embed_text(
                auto_description
            )

            # ------------------------------------------------
            # MANUAL DESCRIPTION
            # ------------------------------------------------

            manual_description = find_manual_annotation(
                image_path
            )

            manual_vector = None

            if manual_description:

                print(
                    "[MANUAL]",
                    manual_description
                )

                manual_vector = embed_text(
                    manual_description
                )

            else:

                print(
                    "[MANUAL] No annotation"
                )

            # ------------------------------------------------
            # IMAGE ID
            # ------------------------------------------------

            image_id = str(
                uuid.uuid4()
            )

            s3_key = (
                f"{image_id}_{filename}"
            )

            # ------------------------------------------------
            # PAYLOAD
            # ------------------------------------------------

            payload = {

                "filename": filename,

                "s3_key": s3_key,

                "auto_description":
                    auto_description,

                "manual_description":
                    manual_description
                    if manual_description
                    else None,

                "has_manual":
                    manual_description is not None
            }

            # ------------------------------------------------
            # NAMED VECTORS
            # ------------------------------------------------

            vectors = {

                "auto":
                    auto_vector.tolist(),

            }

            # В Qdrant manual vector добавляем
            # только если существует ручная разметка.

            if manual_vector is not None:

                vectors["manual"] = (
                    manual_vector.tolist()
                )

            # ------------------------------------------------
            # QDRANT
            # ------------------------------------------------

            qdrant.upsert(

                collection_name=collection_name,

                points=[

                    PointStruct(

                        id=image_id,

                        vector=vectors,

                        payload=payload
                    )
                ]
            )

            # ------------------------------------------------
            # S3
            # ------------------------------------------------

            upload_to_s3(
                image,
                s3_key
            )

            print(
                "[INDEX] OK"
            )

        except Exception as e:

            print(
                f"[ERROR] {filename}: {e}"
            )

        progressbar["value"] = index

        app.update_idletasks()


# ============================================================
# SEARCH
# ============================================================

def search_images(
    query,
    collection_name,
    top_k=5,
    manual_weight=0.7,
    auto_weight=0.3
):
    """
    Взвешенный поиск.

    auto_weight:
        вес автоматического описания

    manual_weight:
        вес ручной разметки

    Для изображений без manual vector
    используется только auto score.

    Результаты объединяются по ID изображения.
    """

    if not query.strip():
        return []

    if collection_name not in list_collections():

        raise ValueError(
            f"Коллекция '{collection_name}' не существует"
        )

    # --------------------------------------------------------
    # Нормализация весов
    # --------------------------------------------------------

    total_weight = (
        manual_weight +
        auto_weight
    )

    if total_weight <= 0:

        raise ValueError(
            "Сумма весов должна быть > 0"
        )

    manual_weight /= total_weight
    auto_weight /= total_weight

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    query_vector = embed_text(
        query
    )

    # --------------------------------------------------------
    # AUTO SEARCH
    # --------------------------------------------------------

    auto_results = qdrant.query_points(

        collection_name=collection_name,

        query=query_vector.tolist(),

        using="auto",

        limit=max(
            top_k * 3,
            10
        ),

        with_payload=True
    ).points

    # --------------------------------------------------------
    # MANUAL SEARCH
    # --------------------------------------------------------

    manual_results = qdrant.query_points(

        collection_name=collection_name,

        query=query_vector.tolist(),

        using="manual",

        limit=max(
            top_k * 3,
            10
        ),

        with_payload=True
    ).points

    # --------------------------------------------------------
    # MERGE RESULTS
    # --------------------------------------------------------

    merged = {}

    # AUTO

    for result in auto_results:

        point_id = str(
            result.id
        )

        if point_id not in merged:

            merged[point_id] = {
                "id": result.id,
                "payload": result.payload,
                "auto_score": None,
                "manual_score": None
            }

        merged[point_id][
            "auto_score"
        ] = float(result.score)

    # MANUAL

    for result in manual_results:

        point_id = str(
            result.id
        )

        if point_id not in merged:

            merged[point_id] = {
                "id": result.id,
                "payload": result.payload,
                "auto_score": None,
                "manual_score": None
            }

        merged[point_id][
            "manual_score"
        ] = float(result.score)

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    results = []

    for item in merged.values():

        auto_score = item[
            "auto_score"
        ]

        manual_score = item[
            "manual_score"
        ]

        # ----------------------------------------------------
        # Есть оба вектора
        # ----------------------------------------------------

        if (
            auto_score is not None
            and
            manual_score is not None
        ):

            final_score = (
                auto_weight * auto_score
                +
                manual_weight * manual_score
            )

        # ----------------------------------------------------
        # Только AUTO
        # ----------------------------------------------------

        elif auto_score is not None:

            final_score = auto_score

        # ----------------------------------------------------
        # Теоретически только MANUAL
        # ----------------------------------------------------

        elif manual_score is not None:

            final_score = manual_score

        else:

            continue

        item[
            "final_score"
        ] = final_score

        results.append(item)

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# UI
# ============================================================

class ImageManagerApp(tk.Tk):

    def __init__(self):

        super().__init__()

        self.title(
            "Image Manager - Qdrant RAG v4"
        )

        self.geometry(
            "1000x800"
        )

        # ----------------------------------------------------
        # NOTEBOOK
        # ----------------------------------------------------

        self.notebook = ttk.Notebook(
            self
        )

        self.notebook.pack(
            fill="both",
            expand=True
        )

        self.tab_collections = ttk.Frame(
            self.notebook
        )

        self.tab_upload = ttk.Frame(
            self.notebook
        )

        self.tab_search = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            self.tab_collections,
            text="🗑 Collections"
        )

        self.notebook.add(
            self.tab_upload,
            text="⬆ Upload"
        )

        self.notebook.add(
            self.tab_search,
            text="🔎 Search"
        )

        # ----------------------------------------------------
        # INITIALIZE
        # ----------------------------------------------------

        self.init_collections_tab()

        self.init_upload_tab()

        self.init_search_tab()


    # ========================================================
    # COLLECTIONS
    # ========================================================

    def init_collections_tab(
        self
    ):

        frame = self.tab_collections

        ttk.Label(
            frame,
            text="Existing collections:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.collections_var = tk.StringVar(
            value=list_collections()
        )

        self.collections_listbox = tk.Listbox(
            frame,
            listvariable=self.collections_var,
            height=10
        )

        self.collections_listbox.pack(
            fill="x",
            padx=10
        )

        ttk.Label(
            frame,
            text="New collection name:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.new_collection_entry = ttk.Entry(
            frame
        )

        self.new_collection_entry.pack(
            fill="x",
            padx=10
        )

        ttk.Button(
            frame,
            text="Create collection",
            command=self.create_collection
        ).pack(
            pady=5,
            padx=10,
            anchor="w"
        )

        ttk.Button(
            frame,
            text="Delete selected",
            command=self.delete_collection
        ).pack(
            pady=5,
            padx=10,
            anchor="w"
        )

        ttk.Button(
            frame,
            text="Refresh",
            command=self.refresh_collections
        ).pack(
            pady=5,
            padx=10,
            anchor="w"
        )

        info = (
            "v4 collection structure:\n\n"
            "auto   = BLIP description → CLIP\n"
            "manual = TXT annotation → CLIP\n\n"
            "Images without TXT have only auto vector."
        )

        ttk.Label(
            frame,
            text=info,
            justify="left"
        ).pack(
            anchor="w",
            padx=10,
            pady=20
        )


    def refresh_collections(
        self
    ):

        collections = list_collections()

        self.collections_var.set(
            collections
        )

        self.upload_collection_combo[
            "values"
        ] = collections

        self.search_collection_combo[
            "values"
        ] = collections


    def create_collection(
        self
    ):

        name = (
            self.new_collection_entry
            .get()
            .strip()
        )

        if not name:

            messagebox.showerror(
                "Error",
                "Enter collection name"
            )

            return

        try:

            result = create_collection(
                name
            )

            messagebox.showinfo(
                "Collection",
                result
            )

            self.new_collection_entry.delete(
                0,
                tk.END
            )

            self.refresh_collections()

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


    def delete_collection(
        self
    ):

        selection = (
            self.collections_listbox
            .curselection()
        )

        if not selection:

            messagebox.showerror(
                "Error",
                "Select collection"
            )

            return

        name = (
            self.collections_listbox
            .get(selection[0])
        )

        answer = messagebox.askyesno(
            "Delete collection",
            f"Delete '{name}'?"
        )

        if not answer:
            return

        try:

            result = delete_collection(
                name
            )

            messagebox.showinfo(
                "Collection",
                result
            )

            self.refresh_collections()

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )


    # ========================================================
    # UPLOAD
    # ========================================================

    def init_upload_tab(
        self
    ):

        frame = self.tab_upload

        ttk.Label(
            frame,
            text="Image directory:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.dir_entry = ttk.Entry(
            frame
        )

        self.dir_entry.pack(
            fill="x",
            padx=10
        )

        ttk.Button(
            frame,
            text="Choose folder",
            command=self.choose_directory
        ).pack(
            pady=5,
            padx=10,
            anchor="w"
        )

        ttk.Label(
            frame,
            text="Minimum side (px):"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.resolution_spin = ttk.Spinbox(
            frame,
            from_=64,
            to=2048,
            increment=1
        )

        self.resolution_spin.set(
            DEFAULT_RESOLUTION
        )

        self.resolution_spin.pack(
            fill="x",
            padx=10
        )

        ttk.Label(
            frame,
            text="Collection:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.upload_collection_var = (
            tk.StringVar()
        )

        self.upload_collection_combo = (
            ttk.Combobox(
                frame,
                textvariable=
                    self.upload_collection_var,
                values=list_collections()
            )
        )

        self.upload_collection_combo.pack(
            fill="x",
            padx=10
        )

        ttk.Label(
            frame,
            text=(
                "Manual annotations:\n"
                "image.jpg → image.txt\n"
                "If TXT exists, a manual vector "
                "will be created."
            ),
            justify="left"
        ).pack(
            anchor="w",
            padx=10,
            pady=15
        )

        self.progressbar = (
            ttk.Progressbar(
                frame,
                length=400,
                mode="determinate"
            )
        )

        self.progressbar.pack(
            pady=10,
            padx=10
        )

        ttk.Button(
            frame,
            text="Upload / Index images",
            command=self.upload_images
        ).pack(
            pady=10,
            padx=10,
            anchor="w"
        )


    def choose_directory(
        self
    ):

        directory = filedialog.askdirectory()

        if directory:

            self.dir_entry.delete(
                0,
                tk.END
            )

            self.dir_entry.insert(
                0,
                directory
            )


    def upload_images(
        self
    ):

        directory = (
            self.dir_entry
            .get()
            .strip()
        )

        collection = (
            self.upload_collection_var
            .get()
            .strip()
        )

        try:

            resolution = int(
                self.resolution_spin.get()
            )

        except ValueError:

            messagebox.showerror(
                "Error",
                "Invalid resolution"
            )

            return

        if not collection:

            messagebox.showerror(
                "Error",
                "Select collection"
            )

            return

        if (
            not directory
            or
            not os.path.exists(directory)
        ):

            messagebox.showerror(
                "Error",
                "Invalid directory"
            )

            return

        try:

            process_directory(
                directory,
                resolution,
                collection,
                self.progressbar,
                self
            )

            messagebox.showinfo(
                "Success",
                "Images indexed successfully"
            )

        except Exception as e:

            messagebox.showerror(
                "Upload error",
                str(e)
            )


    # ========================================================
    # SEARCH
    # ========================================================

    def init_search_tab(
        self
    ):

        frame = self.tab_search

        ttk.Label(
            frame,
            text="Collection:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.search_collection_var = (
            tk.StringVar()
        )

        self.search_collection_combo = (
            ttk.Combobox(
                frame,
                textvariable=
                    self.search_collection_var,
                values=list_collections()
            )
        )

        self.search_collection_combo.pack(
            fill="x",
            padx=10
        )

        ttk.Label(
            frame,
            text="Search query:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.query_entry = ttk.Entry(
            frame
        )

        self.query_entry.pack(
            fill="x",
            padx=10
        )

        # ----------------------------------------------------
        # WEIGHTS
        # ----------------------------------------------------

        weights_frame = ttk.LabelFrame(
            frame,
            text="Search weights"
        )

        weights_frame.pack(
            fill="x",
            padx=10,
            pady=10
        )

        ttk.Label(
            weights_frame,
            text="Manual weight:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=5
        )

        self.manual_weight_spin = (
            ttk.Spinbox(
                weights_frame,
                from_=0.0,
                to=1.0,
                increment=0.1,
                width=10
            )
        )

        self.manual_weight_spin.set(
            "0.7"
        )

        self.manual_weight_spin.grid(
            row=0,
            column=1,
            padx=10,
            pady=5
        )

        ttk.Label(
            weights_frame,
            text="Automatic weight:"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=5
        )

        self.auto_weight_spin = (
            ttk.Spinbox(
                weights_frame,
                from_=0.0,
                to=1.0,
                increment=0.1,
                width=10
            )
        )

        self.auto_weight_spin.set(
            "0.3"
        )

        self.auto_weight_spin.grid(
            row=1,
            column=1,
            padx=10,
            pady=5
        )

        ttk.Label(
            weights_frame,
            text=(
                "Recommended starting point:\n"
                "Manual = 0.7\n"
                "Automatic = 0.3"
            ),
            justify="left"
        ).grid(
            row=0,
            column=2,
            rowspan=2,
            padx=20,
            pady=5
        )

        # ----------------------------------------------------
        # TOP K
        # ----------------------------------------------------

        ttk.Label(
            frame,
            text="Number of results:"
        ).pack(
            anchor="w",
            padx=10,
            pady=5
        )

        self.top_k_spin = ttk.Spinbox(
            frame,
            from_=1,
            to=20,
            increment=1,
            width=10
        )

        self.top_k_spin.set(
            "5"
        )

        self.top_k_spin.pack(
            anchor="w",
            padx=10
        )

        # ----------------------------------------------------
        # BUTTON
        # ----------------------------------------------------

        ttk.Button(
            frame,
            text="🔎 Search",
            command=self.search_images_ui
        ).pack(
            pady=10,
            padx=10,
            anchor="w"
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        self.results_frame = ttk.Frame(
            frame
        )

        self.results_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )


    def search_images_ui(
        self
    ):

        collection = (
            self.search_collection_var
            .get()
            .strip()
        )

        query = (
            self.query_entry
            .get()
            .strip()
        )

        if not collection:

            messagebox.showerror(
                "Error",
                "Select collection"
            )

            return

        if not query:

            messagebox.showerror(
                "Error",
                "Enter search query"
            )

            return

        try:

            manual_weight = float(
                self.manual_weight_spin.get()
            )

            auto_weight = float(
                self.auto_weight_spin.get()
            )

            top_k = int(
                self.top_k_spin.get()
            )

        except ValueError:

            messagebox.showerror(
                "Error",
                "Invalid search parameters"
            )

            return

        # Clear previous results

        for widget in (
            self.results_frame
            .winfo_children()
        ):

            widget.destroy()

        try:

            results = search_images(

                query,

                collection,

                top_k=top_k,

                manual_weight=
                    manual_weight,

                auto_weight=
                    auto_weight
            )

            if not results:

                ttk.Label(
                    self.results_frame,
                    text="Nothing found"
                ).pack(
                    anchor="w"
                )

                return

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            for index, result in enumerate(
                results,
                start=1
            ):

                payload = (
                    result["payload"]
                )

                final_score = (
                    result["final_score"]
                )

                auto_score = (
                    result["auto_score"]
                )

                manual_score = (
                    result["manual_score"]
                )

                filename = payload.get(
                    "filename",
                    "unknown"
                )

                s3_key = payload.get(
                    "s3_key"
                )

                # --------------------------------------------
                # RESULT FRAME
                # --------------------------------------------

                result_frame = ttk.Frame(
                    self.results_frame,
                    relief="solid",
                    borderwidth=1
                )

                result_frame.pack(
                    fill="x",
                    pady=8
                )

                # --------------------------------------------
                # TEXT
                # --------------------------------------------

                text = (
                    f"#{index}  {filename}\n"
                    f"Final score: "
                    f"{final_score:.4f}\n"
                    f"Auto score: "
                    f"{auto_score:.4f}"
                    if auto_score is not None
                    else
                    f"#{index}  {filename}\n"
                    f"Final score: "
                    f"{final_score:.4f}\n"
                    f"Auto score: N/A"
                )

                if manual_score is not None:

                    text += (
                        f"\nManual score: "
                        f"{manual_score:.4f}"
                    )

                else:

                    text += (
                        "\nManual score: N/A"
                    )

                ttk.Label(
                    result_frame,
                    text=text,
                    justify="left"
                ).pack(
                    anchor="w",
                    padx=10,
                    pady=5
                )

                # --------------------------------------------
                # DESCRIPTION
                # --------------------------------------------

                auto_description = (
                    payload.get(
                        "auto_description",
                        ""
                    )
                )

                manual_description = (
                    payload.get(
                        "manual_description"
                    )
                )

                ttk.Label(
                    result_frame,
                    text=(
                        "AUTO:\n"
                        + auto_description
                    ),
                    justify="left",
                    wraplength=900
                ).pack(
                    anchor="w",
                    padx=10,
                    pady=3
                )

                if manual_description:

                    ttk.Label(
                        result_frame,
                        text=(
                            "MANUAL:\n"
                            + manual_description
                        ),
                        justify="left",
                        wraplength=900
                    ).pack(
                        anchor="w",
                        padx=10,
                        pady=3
                    )

                # --------------------------------------------
                # IMAGE
                # --------------------------------------------

                if s3_key:

                    try:

                        image = (
                            load_image_from_s3(
                                s3_key
                            )
                        )

                        image.thumbnail(
                            (250, 250)
                        )

                        image_tk = (
                            ImageTk.PhotoImage(
                                image
                            )
                        )

                        image_label = ttk.Label(
                            result_frame,
                            image=image_tk
                        )

                        image_label.image = (
                            image_tk
                        )

                        image_label.pack(
                            anchor="w",
                            padx=10,
                            pady=5
                        )

                    except Exception as image_error:

                        ttk.Label(
                            result_frame,
                            text=(
                                "Image loading error: "
                                f"{image_error}"
                            )
                        ).pack(
                            anchor="w",
                            padx=10
                        )

        except Exception as e:

            messagebox.showerror(
                "Search error",
                str(e)
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = ImageManagerApp()

    app.mainloop()