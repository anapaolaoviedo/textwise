'''
file to generate embeddings using sentence-transformers 
'''
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer #hugging face miniML
import pickle 

DEFAULT_MODEL = "all-MiniLM-L6-v2"

def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    print(f"loading model {model_name}")
    model = SentenceTransformer(model_name) #should i leave it this way or use -> ('sentence-transformers/all-MiniLM-L6-v2') ?
    return model 

def generate_embeddings(
    texts: list[str], 
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = True
    ) -> np.ndarray:
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    return embeddings

def save_embeddings(embeddings: np.ndarray, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    
    print(f"embeddings saved in: {path}")
    
def show_embeddings(path: str) -> np.ndarray:
    return np.load(path)

#should i do a funciton to see if an embedding exists?
def embedding_exists(path:str) -> bool:
    return Path(path).exists()

def generate_save_embeddings(lyrics: list[str],
    output_path: str = "embeddings/song_embeddings.npy",
    model_name: str = DEFAULT_MODEL 
    ):
    
    model = load_model(model_name)
    
    print(f"generating embeddings for {len(lyrics)} songs ! ")
    embeddings = generate_embeddings(lyrics, model)
    
    save_embeddings(embeddings, output_path)
    
    return embeddings
    



