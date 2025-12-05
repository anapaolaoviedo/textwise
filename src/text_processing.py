'''
load and process texts 
'''
import pandas as pd 
import re 
from pathlib import Path 

def load_songs(path: str = "data/songs.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    
    df["song_id"] = range(len(df)) #unique id for each song
    df["display_title"] =  df ["Artist"] + " - " + df["Title"]
    return df

def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    #pure regex implementation to clean up lyrics
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def split_into_sentences(text: str) -> list[str]:
    """
    Divide letras en frases/líneas.
    Como las letras del dataset no tienen saltos de línea,
    dividimos por patrones comunes: puntuación, palabras repetidas, etc.
    """
    if not isinstance(text, str):
        return []
    
    text = clean_lyrics(text)
    
    # Si hay saltos de línea reales, usar esos
    if '\n' in text:
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 3]
        if len(lines) > 3:
            return lines
    
    # Si no hay saltos de línea, dividir de forma inteligente
    # Dividir por patrones comunes en letras
    
    # Opción 1: Dividir cada ~50-80 caracteres en un espacio
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        current_line.append(word)
        current_length += len(word) + 1
        
        # Cortar en ~60 caracteres o después de puntuación
        if current_length >= 60 or word.endswith(('.', '!', '?', ',')):
            line = ' '.join(current_line).strip()
            if len(line) > 5:
                lines.append(line)
            current_line = []
            current_length = 0
    
    # Agregar última línea
    if current_line:
        line = ' '.join(current_line).strip()
        if len(line) > 5:
            lines.append(line)
    
    return lines

def get_song_by_id(df:pd.DataFrame, song_id: int) -> dict:
    row = df[df["song_id"] == song_id]. iloc[0]
    return {
        "song_id": song_id,
        "artist": row["Artist"],
        "title": row["Title"],
        "album": row["Album"],
        "year": row["Year"],
        "lyrics": row["Lyric"],
        "display_title": row["display_title"]
    }
    
def search_songs(df: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.lower()
    mask = (
        df["Title"].str.lower().str.contains(query, na=False) |
        df["Artist"].str.lower().str.contains(query)
    )
    return df[mask]