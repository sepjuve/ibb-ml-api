from starlette.config import Config

config = Config(".env")

DEFAULT_MODEL_PATH: str = config("DEFAULT_MODEL_PATH")