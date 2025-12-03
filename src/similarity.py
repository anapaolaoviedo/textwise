'''
file to calculate similiarity between texts 
'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity #idk what happened but apparenlty requirement already satisfied 

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    #idk but i have to do some weird stuff
    return cosine_similarity(embeddings)

def find_most_similar(
    song_idx: int, 
    similarity_matrix: np.ndarray, 
    n: int = 10,
    exclude_self: bool = True
    ) -> list[tuple[int, float]]:
    similarities = similarity_matrix[song_idx]
    
    sorted_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in sorted_indices:
        if exclude_self and idx == song_idx:
            continue
        results.append((int(idx), float(similarities[idx])))
        if len(results) >= n:
            break
    
    return results


def calc_phrase_similarity(
    phrases1: list[str], 
    phrases2: list[str],
    model
    ) -> np.ndarray:
    if not phrases1 or not phrases2:
        return np.array([[]])
    
    emb1 = model.encode(phrases1, convert_to_numpy=True)
    emb2 = model.encode(phrases2, convert_to_numpy=True)
    
    return cosine_similarity(emb1, emb2)
    

def get_top_phrase_pairs(
    similarity_matrix: np.ndarray,
    phrases1: list[str],
    phrases2: list[str],
    n: int = 5
    ) -> list[tuple[str, str, float]]:

    flat_indices = np.argsort(similarity_matrix.flatten())[::-1][:n]
    
    results = []
    for flat_idx in flat_indices:
        i = flat_idx // similarity_matrix.shape[1]
        j = flat_idx % similarity_matrix.shape[1]
        score = similarity_matrix[i, j]
        results.append((phrases1[i], phrases2[j], float(score)))
    
    return results