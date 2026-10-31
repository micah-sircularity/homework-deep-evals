from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

MODEL_NAME = "accounts/fireworks/models/glm-5p3-flash"
MODEL = f"fireworks:{MODEL_NAME}"
