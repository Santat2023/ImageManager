import sys
import os
import subprocess

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chroma_data")

    # chroma запускается как subprocess
    subprocess.run([
        sys.executable, "-m", "chromadb",
        "run", "--host", "127.0.0.1", "--port", "8000",
        "--path", db_path
    ])
