from .text_processing import (
    load_songs,
    clean_lyrics,
    split_into_sentences,
    get_song_by_id,
    search_songs
)

from .embeddings import (
    load_model,
    generate_embeddings,
    save_embeddings,
    show_embeddings,
    embedding_exists,
    generate_save_embeddings
)

from .similarity import (
    compute_similarity_matrix,
    find_most_similar,
    calc_phrase_similarity,
    get_top_phrase_pairs
)