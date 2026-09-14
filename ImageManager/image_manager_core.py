import os, io, uuid
import boto3, botocore, clip, torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

S3_ENDPOINT = "http://localhost:9000"
S3_BUCKET = "images"
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"

CLIP_MODEL = "ViT-B/32"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

DEFAULT_RESOLUTION = 250
CLIP_DIM = 512

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[SYSTEM] Device: {device}")

print("[CLIP] Загрузка модели...")
clip_model, clip_preprocess = clip.load(
    CLIP_MODEL,
    device=device
)
clip_model.eval()
print("[CLIP] Модель загружена")

print("[BLIP] Загрузка модели...")

blip_processor = BlipProcessor.from_pretrained(
    BLIP_MODEL
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL
).to(device)

blip_model.eval()

print("[BLIP] Модель загружена")


# ============================================================
# QDRANT
# ============================================================

qdrant = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    check_compatibility=False
)


# ============================================================
# MINIO / S3
# ============================================================

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

try:
    s3.create_bucket(
        Bucket=S3_BUCKET
    )

except botocore.exceptions.ClientError as e:

    error_code = e.response["Error"]["Code"]

    if error_code not in (
        "BucketAlreadyOwnedByYou",
        "BucketAlreadyExists"
    ):
        raise


# ============================================================
# COLLECTIONS
# ============================================================

def list_collections():

    collections = (
        qdrant
        .get_collections()
        .collections
    )

    return [
        collection.name
        for collection in collections
    ]


def create_collection(name):

    if not name:
        raise ValueError(
            "Имя коллекции не может быть пустым"
        )

    if name in list_collections():

        return (
            f"Коллекция '{name}' "
            f"уже существует"
        )

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
        f"Векторы: auto + manual"
    )


def delete_collection(name):

    qdrant.delete_collection(
        collection_name=name
    )

    return (
        f"Коллекция '{name}' удалена"
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def resize_to_min_side(
    image_path,
    target_size=DEFAULT_RESOLUTION
):

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    if width <= height:

        new_width = target_size

        new_height = int(
            height *
            target_size /
            width
        )

    else:

        new_height = target_size

        new_width = int(
            width *
            target_size /
            height
        )

    return image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


# ============================================================
# BLIP
# ============================================================

def describe_image(image):

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

    return blip_processor.decode(
        output[0],
        skip_special_tokens=True
    ).strip()


# ============================================================
# MANUAL DESCRIPTION
# ============================================================

def find_manual_annotation(
    image_path
):

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
            f"[WARNING] "
            f"Не удалось прочитать "
            f"{txt_path}: {e}"
        )

        return None


# ============================================================
# CLIP
# ============================================================

def embed_text(text):

    if not text:
        return None

    tokens = clip.tokenize(
        [text],
        truncate=True
    ).to(device)

    with torch.no_grad():

        embedding = (
            clip_model
            .encode_text(tokens)
            .float()
        )

    embedding = (
        embedding /
        embedding.norm(
            dim=-1,
            keepdim=True
        )
    )

    return embedding.cpu().numpy()[0]


# ============================================================
# S3
# ============================================================

def upload_to_s3(
    image,
    key
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
    key
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
# INDEXING
# ============================================================

def process_directory(
    directory,
    resolution,
    collection_name,
    progress_callback=None
):

    files = [
        file
        for file in os.listdir(directory)
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    if not files:

        raise ValueError(
            "В выбранной директории "
            "нет изображений"
        )

    total = len(files)

    print(
        f"[INDEX] "
        f"Найдено изображений: {total}"
    )

    for index, filename in enumerate(
        files,
        start=1
    ):

        try:

            image_path = os.path.join(
                directory,
                filename
            )

            image = resize_to_min_side(
                image_path,
                resolution
            )

            # ------------------------
            # AUTO
            # ------------------------

            auto_description = (
                describe_image(image)
            )

            print(
                "[BLIP]",
                auto_description
            )

            auto_vector = embed_text(
                auto_description
            )

            # ------------------------
            # MANUAL
            # ------------------------

            manual_description = (
                find_manual_annotation(
                    image_path
                )
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
                    "[MANUAL] "
                    "Разметка отсутствует"
                )

            # ------------------------
            # ID
            # ------------------------

            image_id = str(
                uuid.uuid4()
            )

            s3_key = (
                f"{image_id}_{filename}"
            )

            # ------------------------
            # PAYLOAD
            # ------------------------

            payload = {

                "filename":
                    filename,

                "s3_key":
                    s3_key,

                "auto_description":
                    auto_description,

                "manual_description":
                    manual_description,

                "has_manual":
                    manual_description
                    is not None
            }

            # ------------------------
            # VECTORS
            # ------------------------

            vectors = {

                "auto":
                    auto_vector.tolist()
            }

            if manual_vector is not None:

                vectors["manual"] = (
                    manual_vector.tolist()
                )

            # ------------------------
            # QDRANT
            # ------------------------

            qdrant.upsert(

                collection_name=
                    collection_name,

                points=[

                    PointStruct(

                        id=image_id,

                        vector=vectors,

                        payload=payload
                    )
                ]
            )

            # ------------------------
            # S3
            # ------------------------

            upload_to_s3(
                image,
                s3_key
            )

            print(
                "[INDEX] OK"
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{filename}: {e}"
            )

        if progress_callback:

            progress_callback(
                index,
                total
            )


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

    if not query.strip():
        return []

    if (
        collection_name
        not in list_collections()
    ):

        raise ValueError(
            f"Коллекция "
            f"'{collection_name}' "
            f"не существует"
        )

    total_weight = (
        manual_weight +
        auto_weight
    )

    if total_weight <= 0:

        raise ValueError(
            "Сумма весов "
            "должна быть больше 0"
        )

    manual_weight /= total_weight
    auto_weight /= total_weight

    query_vector = embed_text(
        query
    ).tolist()

    search_limit = max(
        top_k * 3,
        10
    )

    # ------------------------
    # AUTO SEARCH
    # ------------------------

    auto_results = qdrant.query_points(

        collection_name=
            collection_name,

        query=query_vector,

        using="auto",

        limit=search_limit,

        with_payload=True
    ).points

    # ------------------------
    # MANUAL SEARCH
    # ------------------------

    manual_results = qdrant.query_points(

        collection_name=
            collection_name,

        query=query_vector,

        using="manual",

        limit=search_limit,

        with_payload=True
    ).points

    # ------------------------
    # MERGE
    # ------------------------

    merged = {}

    for result in auto_results:

        point_id = str(
            result.id
        )

        if point_id not in merged:

            merged[point_id] = {

                "id":
                    result.id,

                "payload":
                    result.payload,

                "auto_score":
                    None,

                "manual_score":
                    None
            }

        merged[point_id][
            "auto_score"
        ] = float(
            result.score
        )

    for result in manual_results:

        point_id = str(
            result.id
        )

        if point_id not in merged:

            merged[point_id] = {

                "id":
                    result.id,

                "payload":
                    result.payload,

                "auto_score":
                    None,

                "manual_score":
                    None
            }

        merged[point_id][
            "manual_score"
        ] = float(
            result.score
        )

    # ------------------------
    # FINAL SCORE
    # ------------------------

    results = []

    for item in merged.values():

        auto_score = (
            item["auto_score"]
        )

        manual_score = (
            item["manual_score"]
        )

        if (
            auto_score is not None
            and
            manual_score is not None
        ):

            final_score = (
                auto_weight *
                auto_score
                +
                manual_weight *
                manual_score
            )

        elif auto_score is not None:

            final_score = auto_score

        elif manual_score is not None:

            final_score = manual_score

        else:

            continue

        item[
            "final_score"
        ] = final_score

        results.append(item)

    results.sort(
        key=lambda x:
            x["final_score"],
        reverse=True
    )

    return results[:top_k]