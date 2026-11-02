"""Task-specific endpoint smoke checks; no user content or benchmark claims."""

import io
import math
import wave


async def probe_recipe(client, recipe, headers, task=None):
    prefix = "/api/v1/pool/node/inference/"
    common = {"params": {"model_id": recipe.model.repository}, "headers": headers}
    arguments = recipe.runtime.arguments
    model = recipe.launch.served_model_name
    if "--convert" in arguments and arguments[arguments.index("--convert") + 1] == "embed":
        response = await client.post(
            prefix + "v1/embeddings",
            **common,
            json={"model": model, "input": ["Recipe endpoint check"]},
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        vector = data[0].get("embedding", []) if data else []
        if not vector or not all(isinstance(n, (float, int)) and math.isfinite(n) for n in vector):
            raise ValueError("Recipe endpoint returned no finite embedding vector")
    elif "--convert" in arguments and arguments[arguments.index("--convert") + 1] == "classify":
        response = await client.post(
            prefix + "classify", **common, json={"model": model, "input": ["Recipe endpoint check"]}
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        probabilities = data[0].get("probs", []) if data else []
        if not probabilities or not all(
            isinstance(n, (float, int)) and math.isfinite(n) for n in probabilities
        ):
            raise ValueError("Recipe endpoint returned no classification probabilities")
    elif task == "automatic-speech-recognition":
        audio = io.BytesIO()
        with wave.open(audio, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        response = await client.post(
            prefix + "v1/audio/transcriptions",
            **common,
            data={"model": model, "language": "en"},
            files={"file": ("silence.wav", audio.getvalue(), "audio/wav")},
        )
        response.raise_for_status()
        if not isinstance(response.json().get("text"), str):
            raise ValueError("Recipe endpoint returned no transcription result")
        # Silence may produce empty text. This checks the API, not transcription accuracy.
    elif task == "text-generation":
        response = await client.post(
            prefix + "v1/completions",
            **common,
            json={
                "model": model,
                "prompt": "The capital of France is",
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices or not choices[0].get("text"):
            raise ValueError("Recipe endpoint returned no completion text")
    else:
        response = await client.post(
            prefix + "v1/chat/completions",
            **common,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 128,
                "temperature": 0,
                "chat_template_kwargs": {"thinking": False},
            },
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices or not choices[0].get("message", {}).get("content"):
            raise ValueError("Recipe endpoint returned no generated text")
