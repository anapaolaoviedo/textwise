

import sys
sys.path.insert(0, ".")

from src.text_processing import load_songs, clean_lyrics
from src.embeddings import generate_save_embeddings

def main():
    print("loading songies")
    df = load_songs("data/songs.csv")
    print(f"total of: {len(df)} songies")
    
    print("cleaning...")
    lyrics = [clean_lyrics(lyric) for lyric in df["Lyric"].tolist()]
    
    generate_save_embeddings(
        lyrics=lyrics,
        output_path="embeddings/song_embeddings.npy"
    )
    
    print("\nreaddyyy")

if __name__ == "__main__":
    main()