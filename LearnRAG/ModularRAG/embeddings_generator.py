import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:
    def __init__(self, model_name : str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name,device="cpu")
        print(f"{self.model_name} model loaded successfully with dimension: {self.model.get_sentence_embedding_dimension()}")
        print()

    def generate_embedding(self, text : list[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError(f"{self.model_name} model not loaded")
        print(f"Generating embeddings for {len(text)} texts...")
        print()
        embeddings = self.model.encode(
            text,
            batch_size=1,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        print()
        return embeddings