"""Controller-local Hugging Face credentials; public APIs expose presence only."""

import os
import re
import tempfile
from pathlib import Path

import httpx
from pydantic import BaseModel, SecretStr

from .config import Settings


class HuggingFaceTokenRequest(BaseModel):
    token: SecretStr


def token_path(settings: Settings) -> Path:
    return settings.setup_state_path / "huggingface-token"


def load_token(settings: Settings) -> None:
    path = token_path(settings)
    if path.is_file():
        settings.hf_token = path.read_text().strip() or settings.hf_token


def save_token(settings: Settings, token: str) -> None:
    path = token_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".hf-token-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(token)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


async def validate_token(token: str) -> None:
    if not re.fullmatch(r"hf_[A-Za-z0-9]{1,509}", token):
        raise ValueError("Enter a valid Hugging Face access token (hf_…).")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError:
        raise RuntimeError("Hugging Face could not validate the token. Try again later.") from None
    if response.status_code in {401, 403}:
        raise ValueError("Hugging Face rejected this token. Check its validity and permissions.")
    if response.status_code != 200:
        raise RuntimeError("Hugging Face could not validate the token. Try again later.")
