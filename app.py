"""
Textwise - Comparación semántica de letras de canciones
Una sola página, diseño limpio, colores pastel
"""

import streamlit as st
import numpy as np
import plotly.express as px
from src import (
    load_songs,
    load_model,
    show_embeddings,
    embedding_exists,
    compute_similarity_matrix,
    find_most_similar,
    calc_phrase_similarity,
    split_into_sentences,
    clean_lyrics
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Textwise",
    page_icon="🎵",
    layout="wide"
)

# ============================================================
# ESTILOS PASTEL
# ============================================================

st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 50%, #ee9ca7 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #a8edea 0%, #fed6e3 100%);
    }
    
    /* Cards/contenedores */
    .song-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Títulos */
    h1 {
        color: #5a4a78 !important;
        text-align: center;
    }
    
    h2, h3 {
        color: #6b5b7a !important;
    }
    
    /* Resultado similar */
    .similar-item {
        background: linear-gradient(90deg, #d4fc79 0%, #96e6a1 100%);
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .similar-item:hover {
        transform: translateX(5px);
    }
    
    /* Frase box */
    .phrase-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .phrase-box-alt {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    /* Métrica de similitud */
    .sim-score {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 50%;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
    }
    
    /* Letra */
    .lyrics-container {
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 15px;
        max-height: 300px;
        overflow-y: auto;
        font-family: 'Georgia', serif;
        line-height: 1.8;
        white-space: pre-wrap;
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        color: #5a4a78;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data
def load_data():
    return load_songs("data/songs.csv")

@st.cache_data
def load_song_embeddings():
    return show_embeddings("embeddings/song_embeddings.npy")

@st.cache_data
def get_similarity_matrix(_embeddings):
    return compute_similarity_matrix(_embeddings)

@st.cache_resource
def get_model():
    return load_model()

# ============================================================
# INICIALIZACIÓN
# ============================================================

if not embedding_exists("embeddings/song_embeddings.npy"):
    st.error("⚠️ No se encontraron embeddings. Ejecuta: `python scripts/generate_embeddings.py`")
    st.stop()

df = load_data()
embeddings = load_song_embeddings()
similarity_matrix = get_similarity_matrix(embeddings)

# ============================================================
# HEADER
# ============================================================

st.markdown("<h1>🎵 Textwise</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b5b7a; font-size: 18px;'>Descubre qué canciones se parecen entre sí</p>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# PASO 1: SELECCIONAR CANCIÓN
# ============================================================

st.markdown("## 🎤 Paso 1: Elige una canción")

col1, col2 = st.columns([3, 1])

with col1:
    search = st.text_input("Buscar", placeholder="🔍 Escribe artista o título...", label_visibility="collapsed")

with col2:
    n_results = st.selectbox("Mostrar", [5, 10, 15, 20], index=1, label_visibility="collapsed")

# Filtrar
if search:
    mask = (
        df["Title"].str.lower().str.contains(search.lower(), na=False) |
        df["Artist"].str.lower().str.contains(search.lower(), na=False)
    )
    options = df[mask]["display_title"].tolist()
else:
    options = df["display_title"].tolist()

if not options:
    st.warning("No encontré canciones con ese nombre")
    st.stop()

selected = st.selectbox(
    "Canción seleccionada",
    options,
    label_visibility="collapsed"
)

# Info de la canción seleccionada
idx1 = df[df["display_title"] == selected].index[0]
song1 = df.iloc[idx1]

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div class="song-card">
        <h3 style="margin:0; color:#5a4a78;">🎵 {song1['Title']}</h3>
        <p style="margin:5px 0; color:#888;">por <strong>{song1['Artist']}</strong></p>
        <p style="margin:5px 0; color:#aaa; font-size:14px;">💿 {song1['Album']} • 📅 {song1['Year']}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    with st.expander("📜 Ver letra"):
        st.markdown(f"<div class='lyrics-container'>{clean_lyrics(song1['Lyric'])}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# PASO 2: VER CANCIONES SIMILARES
# ============================================================

st.markdown("## 🎯 Paso 2: Canciones similares")

similar = find_most_similar(idx1, similarity_matrix, n=50)

# Filtrar duplicados/remixes
filtered = []
seen = set()
for idx, score in similar:
    s = df.iloc[idx]
    key = f"{s['Artist'].lower()}_{s['Title'].split('(')[0].strip().lower()}"
    if key not in seen:
        filtered.append((idx, score))
        seen.add(key)

# Mostrar en grid
cols = st.columns(2)
for i, (idx, score) in enumerate(filtered[:n_results]):
    s = df.iloc[idx]
    pct = score * 100
    
    with cols[i % 2]:
        if st.button(
            f"🎵 {s['Artist']} - {s['Title']}  ({pct:.0f}%)",
            key=f"btn_{idx}",
            use_container_width=True
        ):
            st.session_state.song2_idx = idx

st.markdown("---")

# ============================================================
# PASO 3: COMPARACIÓN FRASE A FRASE
# ============================================================

st.markdown("## 🔬 Paso 3: Comparación frase a frase")

# Selector de segunda canción
if "song2_idx" not in st.session_state:
    st.session_state.song2_idx = filtered[0][0] if filtered else None

# Dropdown para seleccionar también
similar_options = [f"{df.iloc[idx]['display_title']} ({score*100:.0f}%)" for idx, score in filtered[:n_results]]
similar_indices = [idx for idx, _ in filtered[:n_results]]

selected_idx2 = st.selectbox(
    "O elige de la lista:",
    range(len(similar_options)),
    format_func=lambda x: similar_options[x],
    index=0
)

idx2 = similar_indices[selected_idx2]
song2 = df.iloc[idx2]

# Mostrar las dos canciones lado a lado
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown(f"""
    <div class="song-card" style="text-align:center;">
        <p style="color:#888; margin:0;">CANCIÓN 1</p>
        <h3 style="color:#5a4a78; margin:10px 0;">{song1['Title']}</h3>
        <p style="color:#6b5b7a;">{song1['Artist']}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    global_sim = similarity_matrix[idx1, idx2] * 100
    st.markdown(f"""
    <div style="text-align:center; padding:20px;">
        <div class="sim-score">{global_sim:.0f}%</div>
        <p style="color:#5a4a78; margin-top:10px;">Similitud</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="song-card" style="text-align:center;">
        <p style="color:#888; margin:0;">CANCIÓN 2</p>
        <h3 style="color:#5a4a78; margin:10px 0;">{song2['Title']}</h3>
        <p style="color:#6b5b7a;">{song2['Artist']}</p>
    </div>
    """, unsafe_allow_html=True)

# Calcular similitud frase a frase
phrases1 = split_into_sentences(clean_lyrics(song1['Lyric']))
phrases2 = split_into_sentences(clean_lyrics(song2['Lyric']))

if phrases1 and phrases2:
    model = get_model()
    phrase_matrix = calc_phrase_similarity(phrases1, phrases2, model)
    
    # Heatmap
    st.markdown("### 🗺️ Mapa de similitud")
    
    fig = px.imshow(
        phrase_matrix,
        x=[f"{i+1}" for i in range(len(phrases2))],
        y=[f"{i+1}" for i in range(len(phrases1))],
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    
    fig.update_layout(
        height=400,
        xaxis_title=f"Líneas de {song2['Title']}",
        yaxis_title=f"Líneas de {song1['Title']}",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Selector de frases
    st.markdown("### 🔍 Explorar frases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        line1 = st.selectbox(
            f"Línea de {song1['Title']}",
            range(len(phrases1)),
            format_func=lambda x: f"Línea {x+1}"
        )
    
    with col2:
        line2 = st.selectbox(
            f"Línea de {song2['Title']}",
            range(len(phrases2)),
            format_func=lambda x: f"Línea {x+1}"
        )
    
    # Mostrar frases seleccionadas
    sim = phrase_matrix[line1, line2] * 100
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"<div class='phrase-box'>{phrases1[line1]}</div>", unsafe_allow_html=True)
    
    with col2:
        color = "#28a745" if sim >= 70 else "#ffc107" if sim >= 40 else "#dc3545"
        st.markdown(f"""
        <div style="text-align:center; padding:20px;">
            <span style="font-size:32px; color:{color}; font-weight:bold;">{sim:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<div class='phrase-box-alt'>{phrases2[line2]}</div>", unsafe_allow_html=True)
    
    # Top frases más similares
    st.markdown("### 🏆 Frases más similares")
    
    top_indices = np.argsort(phrase_matrix.flatten())[::-1][:5]
    
    for rank, flat_idx in enumerate(top_indices, 1):
        i = flat_idx // phrase_matrix.shape[1]
        j = flat_idx % phrase_matrix.shape[1]
        score = phrase_matrix[i, j] * 100
        
        with st.expander(f"#{rank} — {score:.0f}% similitud"):
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**{song1['Title']}** (línea {i+1}):\n\n{phrases1[i]}")
            with c2:
                st.success(f"**{song2['Title']}** (línea {j+1}):\n\n{phrases2[j]}")

else:
    st.warning("Una de las canciones no tiene letra disponible")

# ============================================================
# MATRIZ GLOBAL (colapsada)
# ============================================================

st.markdown("---")

with st.expander("🗺️ Ver matriz global de similitud"):
    st.markdown("Selecciona artistas para comparar todas sus canciones:")
    
    artists = sorted(df["Artist"].unique().tolist())
    selected_artists = st.multiselect("Artistas", artists, default=artists[:2])
    
    if len(selected_artists) >= 2:
        # Tomar canciones balanceadas de cada artista
        songs_per_artist = 15 // len(selected_artists)
        if songs_per_artist < 3:
            songs_per_artist = 3
        
        indices = []
        titles = []
        
        for artist in selected_artists:
            artist_df = df[df["Artist"] == artist].head(songs_per_artist)
            indices.extend(artist_df.index.tolist())
            titles.extend(artist_df["display_title"].tolist())
        
        # Limitar a 40 máximo
        indices = indices[:40]
        titles = titles[:40]
        
        sub_matrix = similarity_matrix[np.ix_(indices, indices)]
        
        fig = px.imshow(
            sub_matrix,
            x=titles,
            y=titles,
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align:center; color:#888;'>📊 {len(df):,} canciones de {df['Artist'].nunique()} artistas</p>", unsafe_allow_html=True)