import logging
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)


class OllamaEmbeddingGenerator:
    """Handles embedding generation using Ollama API."""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()

    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return False

    def ensure_model_available(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                if self.model in available_models:
                    return True
                else:
                    logger.warning(
                        f"Model {self.model} not found. Available models: {available_models}"
                    )
                    logger.info(f"Pulling model {self.model}...")
                    return self.pull_model()
            return False
        except Exception as e:
            logger.error(f"Error checking available models: {e}")
            return False

    def pull_model(self) -> bool:
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull", json={"name": self.model}, timeout=300
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error pulling model {self.model}: {e}")
            return False

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            payload = {"model": self.model, "prompt": text}
            response = self.session.post(
                f"{self.base_url}/api/embeddings", json=payload, timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding")
            else:
                logger.error(
                    f"Embedding generation failed: {response.status_code} - {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        embeddings = []
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding {i + 1}/{len(texts)}")
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings
