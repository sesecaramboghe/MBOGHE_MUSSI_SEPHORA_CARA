# =============================================================================
#  DÉTECTION DE FRAUDE BANCAIRE — Application Streamlit
#  Auteur  : MBOGHE Mussi
#  Modèles : Decision Tree (élagué) + XGBoost
#  Méthode : Analyse → Nettoyage → Modélisation → Évaluation → Prédiction
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

import xgboost as xgb

# ── Palette & style globaux ────────────────────────────────────────────────────
BLEU_FONCE  = "#0D1B2A"
BLEU_MOYEN  = "#1B3A5C"
BLEU_CLAIR  = "#2E7DAF"
ACCENT      = "#E8C547"        # jaune doré
ROUGE       = "#E05C4B"
VERT        = "#4CAF50"
BLANC       = "#F5F5F0"

st.set_page_config(
    page_title="Fraude Bancaire — MBOGHE",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Sora', sans-serif;
      background-color: {BLEU_FONCE};
      color: {BLANC};
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {BLEU_FONCE} 0%, {BLEU_MOYEN} 100%);
      border-right: 2px solid {BLEU_CLAIR};
  }}
  [data-testid="stSidebar"] * {{ color: {BLANC} !important; }}

  /* Titres */
  h1 {{ color: {ACCENT} !important; font-weight: 700; letter-spacing: -1px; }}
  h2 {{ color: {BLEU_CLAIR} !important; font-weight: 600; }}
  h3 {{ color: {BLANC} !important; font-weight: 400; }}

  /* Cartes métriques */
  [data-testid="stMetric"] {{
      background: {BLEU_MOYEN};
      border: 1px solid {BLEU_CLAIR};
      border-radius: 10px;
      padding: 12px;
  }}
  [data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-family: 'IBM Plex Mono'; font-size: 28px !important; }}
  [data-testid="stMetricLabel"] {{ color: {BLANC} !important; }}

  /* Boîtes d'info */
  .info-box {{
      background: {BLEU_MOYEN};
      border-left: 4px solid {ACCENT};
      border-radius: 0 8px 8px 0;
      padding: 14px 18px;
      margin: 10px 0;
      font-size: 0.9rem;
      line-height: 1.6;
  }}

  /* Boutons */
  .stButton > button {{
      background: linear-gradient(135deg, {BLEU_CLAIR}, {ACCENT});
      color: {BLEU_FONCE};
      font-weight: 700;
      border: none;
      border-radius: 8px;
      padding: 10px 28px;
      font-family: 'Sora', sans-serif;
      transition: transform .15s;
  }}
  .stButton > button:hover {{ transform: scale(1.03); }}

  /* Tableaux */
  [data-testid="stDataFrame"] {{ border: 1px solid {BLEU_CLAIR}; border-radius: 8px; }}

  /* Séparateur */
  hr {{ border: 1px solid {BLEU_CLAIR}; }}

  /* Fond des plots */
  .plot-bg {{ background: {BLEU_MOYEN}; border-radius: 10px; padding: 10px; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ÉTAT DE SESSION — stocke les données entre les pages
# ══════════════════════════════════════════════════════════════════════════════
def init_session():
    """Initialise les variables persistantes dans st.session_state."""
    defaults = {
        "df_brut"      : None,   # DataFrame original
        "df_propre"    : None,   # DataFrame après nettoyage
        "modele_dt"    : None,   # Decision Tree élagué
        "modele_xgb"   : None,   # XGBoost
        "X_test"       : None,
        "y_test"       : None,
        "features"     : None,   # liste des colonnes features
        "target"       : None,   # nom de la colonne cible
        "encodeurs"    : {},     # LabelEncoders par colonne
        "entrainement_fait": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 20px 0 10px'>
        <div style='font-size:2.5rem'>🏦</div>
        <div style='font-size:1.1rem; font-weight:700; color:{ACCENT}'>Détection Fraude</div>
        <div style='font-size:0.75rem; color:#aaa; margin-top:4px'>MBOGHE Mussi · L3 S2 ML</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Accueil",
            "📊  Analyse Descriptive",
            "🧹  Nettoyage des Données",
            "🌳  Modélisation",
            "🔮  Prédiction",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Indicateurs de progression
    st.markdown("**Progression**")
    etapes = {
        "Données chargées"   : st.session_state.df_brut is not None,
        "Données nettoyées"  : st.session_state.df_propre is not None,
        "Modèle entraîné"    : st.session_state.entrainement_fait,
    }
    for label, fait in etapes.items():
        icon = "✅" if fait else "⬜"
        st.markdown(f"{icon} {label}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Accueil":
    st.title("Détection de Fraude Bancaire")
    st.markdown(f"<div style='color:{BLEU_CLAIR}; font-size:1.1rem; margin-bottom:24px'>Projet Machine Learning · Decision Tree + XGBoost</div>", unsafe_allow_html=True)

    # ── Présentation du projet ─────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"""
        <div class="info-box">
        <b>🎯 Objectif</b><br>
        Construire un modèle capable de distinguer automatiquement une transaction bancaire
        <b>légitime</b> d'une transaction <b>frauduleuse</b>, en suivant une méthodologie
        rigoureuse de Machine Learning.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="border-left-color:{BLEU_CLAIR}">
        <b>📌 Méthodologie suivie</b><br>
        1. <b>Analyse Descriptive</b> — comprendre les données brutes<br>
        2. <b>Nettoyage</b> — traiter les valeurs manquantes, outliers, encodage<br>
        3. <b>Modélisation</b> — Decision Tree élagué vs XGBoost<br>
        4. <b>Évaluation</b> — matrice de confusion, ROC, cross-validation<br>
        5. <b>Prédiction</b> — prédire sur un nouvel exemple
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box" style="border-left-color:{VERT}">
        <b>🧠 Pourquoi deux modèles ?</b><br>
        Le <b>Decision Tree non contraint</b> souffre de sur-apprentissage : il mémorise
        les données d'entraînement mais échoue sur de nouvelles données.<br><br>
        On le corrige de deux façons :<br>
        • <b>Élagage (pruning)</b> : on limite la profondeur et la taille de l'arbre<br>
        • <b>XGBoost</b> : un ensemble de petits arbres qui se corrigent mutuellement
        (boosting), naturellement résistant au sur-apprentissage.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**📂 Charger votre dataset**")
        st.markdown(f"<div style='color:#aaa; font-size:0.85rem; margin-bottom:8px'>Format attendu : CSV avec une colonne cible binaire (0 = légitime, 1 = fraude)</div>", unsafe_allow_html=True)

        fichier = st.file_uploader("Glissez votre fichier CSV ici", type=["csv"])

        if fichier:
            try:
                df = pd.read_csv(fichier)
                st.session_state.df_brut = df
                st.success(f"✅ Fichier chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")

        if st.session_state.df_brut is not None:
            df = st.session_state.df_brut
            st.markdown("**Aperçu (5 premières lignes)**")
            st.dataframe(df.head(), use_container_width=True)

            # Sélection de la colonne cible
            st.markdown("**🎯 Colonne cible (fraude)**")
            st.markdown(f"<div style='color:#aaa; font-size:0.82rem'>Choisissez la colonne qui indique si la transaction est frauduleuse (généralement 0/1)</div>", unsafe_allow_html=True)
            cible = st.selectbox("Colonne cible", df.columns.tolist(), index=len(df.columns)-1)
            st.session_state.target = cible
            st.session_state.features = [c for c in df.columns if c != cible]

    # ── Colonnes détectées ─────────────────────────────────────────────────
    if st.session_state.df_brut is not None:
        st.markdown("---")
        st.markdown("**Colonnes détectées**")
        df = st.session_state.df_brut
        cols = st.columns(4)
        for i, col_name in enumerate(df.columns):
            dtype = str(df[col_name].dtype)
            with cols[i % 4]:
                couleur = ACCENT if col_name == st.session_state.target else BLEU_CLAIR
                st.markdown(f"<span style='color:{couleur}'>◆</span> **{col_name}** <span style='color:#888;font-size:0.8rem'>({dtype})</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYSE DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analyse Descriptive":
    st.title("Analyse Descriptive")

    if st.session_state.df_brut is None:
        st.warning("⚠️ Aucune donnée chargée. Allez d'abord sur la page **Accueil** pour importer votre CSV.")
        st.stop()

    df     = st.session_state.df_brut
    target = st.session_state.target

    # ── Statistiques globales ──────────────────────────────────────────────
    st.markdown("## Vue d'ensemble du dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes",         f"{df.shape[0]:,}")
    c2.metric("Colonnes",       df.shape[1])
    c3.metric("Valeurs manq.",  f"{df.isnull().sum().sum():,}")
    if target and target in df.columns:
        nb_fraude = int(df[target].sum())
        c4.metric("Transactions fraude", f"{nb_fraude:,} ({nb_fraude/len(df)*100:.1f}%)")

    # ── Distribution de la cible ───────────────────────────────────────────
    if target and target in df.columns:
        st.markdown("---")
        st.markdown("## Distribution de la variable cible")

        st.markdown(f"""
        <div class="info-box">
        <b>💡 Pourquoi regarder la distribution ?</b><br>
        En détection de fraude, les cas frauduleux sont <b>très rares</b> (souvent &lt; 1 %).
        C'est ce qu'on appelle un <b>déséquilibre de classes</b>. Si on ignore ce problème,
        le modèle apprendra à tout prédire "non-fraude" et sera quand même précis à 99 % —
        mais complètement inutile ! On devra donc <b>rééquilibrer les classes</b>.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            counts = df[target].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4), facecolor=BLEU_MOYEN)
            ax.set_facecolor(BLEU_MOYEN)
            bars = ax.bar(
                ["Légitime (0)", "Fraude (1)"],
                counts.values,
                color=[BLEU_CLAIR, ROUGE],
                width=0.5, edgecolor="white", linewidth=0.5
            )
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f"{val:,}", ha="center", color=BLANC, fontsize=11, fontweight="bold")
            ax.set_title("Répartition des classes", color=ACCENT, fontweight="bold")
            ax.tick_params(colors=BLANC)
            ax.spines[:].set_visible(False)
            ax.yaxis.set_visible(False)
            st.pyplot(fig)

        with col_b:
            st.markdown(f"<br>", unsafe_allow_html=True)
            for label, count in counts.items():
                pct = count / len(df) * 100
                st.markdown(f"""
                <div style="background:{BLEU_FONCE}; border-radius:8px; padding:12px; margin:8px 0">
                    <div style="color:{ROUGE if label==1 else BLEU_CLAIR}; font-weight:700">
                        {'🔴 Fraude' if label==1 else '🟢 Légitime'}
                    </div>
                    <div style="font-family:'IBM Plex Mono'; font-size:1.4rem; color:{ACCENT}">{count:,}</div>
                    <div style="color:#aaa">{pct:.2f} % du dataset</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Statistiques descriptives ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Statistiques descriptives (colonnes numériques)")
    st.markdown(f"""
    <div class="info-box">
    <b>💡 Que lire ici ?</b><br>
    • <b>mean</b> = moyenne &nbsp;|&nbsp; <b>std</b> = écart-type (dispersion)<br>
    • <b>min/max</b> = bornes &nbsp;|&nbsp; <b>50%</b> = médiane<br>
    Un grand écart entre la <b>moyenne</b> et la <b>médiane</b> indique des valeurs aberrantes (outliers).
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

    # ── Valeurs manquantes ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Valeurs manquantes par colonne")
    manquantes = df.isnull().sum()
    manquantes = manquantes[manquantes > 0]
    if manquantes.empty:
        st.success("✅ Aucune valeur manquante détectée !")
    else:
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=BLEU_MOYEN)
        ax.set_facecolor(BLEU_MOYEN)
        manquantes.plot(kind="barh", ax=ax, color=ROUGE, edgecolor="white")
        ax.set_title("Valeurs manquantes", color=ACCENT, fontweight="bold")
        ax.tick_params(colors=BLANC)
        ax.spines[:].set_visible(False)
        st.pyplot(fig)

    # ── Distributions numériques ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Distributions des variables numériques")
    numeriques = df.select_dtypes(include=np.number).columns.tolist()
    if target in numeriques:
        numeriques.remove(target)

    if numeriques:
        n_cols = min(3, len(numeriques))
        rows   = (len(numeriques) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(rows, n_cols, figsize=(5*n_cols, 3*rows), facecolor=BLEU_FONCE)
        axes = np.array(axes).flatten()

        for i, col_name in enumerate(numeriques):
            ax = axes[i]
            ax.set_facecolor(BLEU_MOYEN)
            ax.hist(df[col_name].dropna(), bins=40, color=BLEU_CLAIR, edgecolor="white", linewidth=0.3)
            ax.set_title(col_name, color=ACCENT, fontsize=9)
            ax.tick_params(colors=BLANC, labelsize=7)
            ax.spines[:].set_visible(False)

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig)

    # ── Matrice de corrélation ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Matrice de corrélation")
    st.markdown(f"""
    <div class="info-box">
    <b>💡 Corrélation</b> : mesure entre -1 et 1 le lien linéaire entre deux variables.
    Une corrélation proche de <b>1</b> (rouge foncé) signifie que les deux variables augmentent ensemble.
    On cherche surtout les variables <b>fortement corrélées à la cible</b> (colonne/ligne <i>{target}</i>).
    </div>
    """, unsafe_allow_html=True)

    num_df = df.select_dtypes(include=np.number)
    if len(num_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=BLEU_FONCE)
        ax.set_facecolor(BLEU_FONCE)
        sns.heatmap(
            num_df.corr(), ax=ax,
            cmap="coolwarm", annot=len(num_df.columns) <= 10,
            fmt=".2f", linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Corrélations", color=ACCENT, fontweight="bold")
        ax.tick_params(colors=BLANC, labelsize=8)
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — NETTOYAGE DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧹  Nettoyage des Données":
    st.title("Nettoyage des Données")

    if st.session_state.df_brut is None:
        st.warning("⚠️ Importez d'abord votre CSV sur la page **Accueil**.")
        st.stop()

    df     = st.session_state.df_brut.copy()
    target = st.session_state.target

    # ── Options de nettoyage ───────────────────────────────────────────────
    st.markdown("## Options de nettoyage")
    col1, col2 = st.columns(2)

    with col1:
        suppr_doublons = st.checkbox("Supprimer les doublons", value=True)
        traiter_manquants = st.selectbox(
            "Valeurs manquantes",
            ["Supprimer les lignes", "Remplacer par la médiane/mode"],
            help="La médiane est plus robuste que la moyenne en présence d'outliers"
        )

    with col2:
        traiter_outliers = st.checkbox("Supprimer les outliers (méthode IQR)", value=True)
        rééquilibrer = st.checkbox("Rééquilibrer les classes (oversampling)", value=True)

    st.markdown(f"""
    <div class="info-box">
    <b>🔍 Méthode IQR (Interquartile Range)</b><br>
    On calcule Q1 (25e percentile) et Q3 (75e percentile). L'IQR = Q3 - Q1.<br>
    Toute valeur en dehors de <b>[Q1 - 1.5×IQR, Q3 + 1.5×IQR]</b> est considérée comme aberrante.<br><br>
    <b>📈 Oversampling (SMOTE simplifié)</b><br>
    On <b>duplique aléatoirement</b> les exemples de la classe minoritaire (fraude)
    jusqu'à atteindre le même nombre que la classe majoritaire. Cela évite que le modèle
    ignore la fraude par déséquilibre.
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ Lancer le nettoyage"):
        n_initial = len(df)
        log = []

        # 1. Doublons
        if suppr_doublons:
            n_avant = len(df)
            df = df.drop_duplicates()
            supprimés = n_avant - len(df)
            log.append(f"✅ Doublons supprimés : {supprimés:,}")

        # 2. Valeurs manquantes
        if traiter_manquants == "Supprimer les lignes":
            n_avant = len(df)
            df = df.dropna()
            log.append(f"✅ Lignes avec NaN supprimées : {n_avant - len(df):,}")
        else:
            # Médiane pour numériques, mode pour catégorielles
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            log.append("✅ Valeurs manquantes remplacées (médiane/mode)")

        # 3. Encodage des colonnes catégorielles
        encodeurs = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col != target:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encodeurs[col] = le
        if encodeurs:
            log.append(f"✅ Colonnes encodées (LabelEncoder) : {list(encodeurs.keys())}")
        st.session_state.encodeurs = encodeurs

        # 4. Outliers IQR (uniquement sur features numériques)
        if traiter_outliers:
            features = [c for c in df.select_dtypes(include=np.number).columns if c != target]
            n_avant  = len(df)
            for col in features:
                Q1  = df[col].quantile(0.25)
                Q3  = df[col].quantile(0.75)
                IQR = Q3 - Q1
                masque = (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)
                df = df[masque]
            log.append(f"✅ Outliers supprimés (IQR) : {n_avant - len(df):,} lignes")

        # 5. Rééquilibrage des classes
        if rééquilibrer and target in df.columns:
            majeur  = df[df[target] == df[target].value_counts().idxmax()]
            mineur  = df[df[target] == df[target].value_counts().idxmin()]
            mineur_up = resample(mineur, replace=True, n_samples=len(majeur), random_state=42)
            df = pd.concat([majeur, mineur_up]).sample(frac=1, random_state=42).reset_index(drop=True)
            log.append(f"✅ Classes rééquilibrées — total : {len(df):,} lignes")

        st.session_state.df_propre = df
        st.session_state.features  = [c for c in df.columns if c != target]

        # ── Rapport de nettoyage ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Rapport de nettoyage")
        for msg in log:
            st.markdown(f"<div class='info-box' style='border-left-color:{VERT}'>{msg}</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes initiales",  f"{n_initial:,}")
        c2.metric("Lignes finales",    f"{len(df):,}")
        c3.metric("Lignes supprimées", f"{n_initial - len(df):,}" if n_initial >= len(df) else "N/A (oversampling)")

        # Aperçu du dataset propre
        st.markdown("**Dataset après nettoyage (5 premières lignes)**")
        st.dataframe(df.head(), use_container_width=True)

        # Nouvelle distribution des classes
        if target in df.columns:
            fig, ax = plt.subplots(figsize=(4, 3), facecolor=BLEU_MOYEN)
            ax.set_facecolor(BLEU_MOYEN)
            df[target].value_counts().plot(kind="bar", ax=ax, color=[BLEU_CLAIR, ROUGE], edgecolor="white")
            ax.set_title("Classes après nettoyage", color=ACCENT, fontweight="bold")
            ax.tick_params(colors=BLANC, rotation=0)
            ax.spines[:].set_visible(False)
            st.pyplot(fig)

    elif st.session_state.df_propre is not None:
        st.info("✅ Données déjà nettoyées. Vous pouvez passer à la **Modélisation**.")
        st.dataframe(st.session_state.df_propre.head(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODÉLISATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌳  Modélisation":
    st.title("Modélisation & Évaluation")

    if st.session_state.df_propre is None:
        st.warning("⚠️ Nettoyez d'abord les données sur la page **Nettoyage**.")
        st.stop()

    df       = st.session_state.df_propre
    target   = st.session_state.target
    features = st.session_state.features

    # ── Hyperparamètres Decision Tree ──────────────────────────────────────
    st.markdown("## Hyperparamètres")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌳 Decision Tree (élagué)**")
        profondeur  = st.slider("Profondeur max (max_depth)", 2, 20, 5,
                                help="Limite la profondeur de l'arbre → réduit le sur-apprentissage")
        min_feuille = st.slider("Min échantillons par feuille (min_samples_leaf)", 1, 50, 10,
                                help="Une feuille avec peu d'exemples → sur-apprentissage")
        min_split   = st.slider("Min échantillons pour découper (min_samples_split)", 2, 100, 20,
                                help="Exige plus d'exemples pour couper un nœud → arbre plus conservateur")

    with col2:
        st.markdown("**🚀 XGBoost**")
        xgb_arbres  = st.slider("Nombre d'arbres (n_estimators)", 50, 500, 100,
                                help="Plus d'arbres = meilleur mais plus lent")
        xgb_depth   = st.slider("Profondeur max XGB (max_depth)", 2, 10, 4,
                                help="Les arbres XGBoost sont volontairement petits")
        xgb_lr      = st.slider("Taux d'apprentissage (learning_rate)", 0.01, 0.5, 0.1, step=0.01,
                                help="Taux faible = plus prudent, moins de sur-apprentissage")

    test_ratio = st.slider("Proportion du jeu de test", 0.1, 0.4, 0.2, step=0.05,
                           help="20 % des données serviront à évaluer le modèle (jamais vues à l'entraînement)")

    st.markdown(f"""
    <div class="info-box">
    <b>📐 Élagage (Pruning)</b><br>
    Sans contrainte, un Decision Tree peut avoir des centaines de niveaux et mémoriser chaque
    exemple d'entraînement. On l'élague en fixant <b>max_depth</b>, <b>min_samples_leaf</b>
    et <b>min_samples_split</b> : l'arbre devient plus simple, plus général, et moins sur-ajusté.<br><br>
    <b>🔁 Cross-validation (CV 5-fold)</b><br>
    On divise les données en 5 parties. On entraîne 5 fois en changeant chaque fois la partie
    de test. La moyenne des 5 scores donne une estimation fiable de la performance réelle.
    </div>
    """, unsafe_allow_html=True)

    # ── Entraînement ───────────────────────────────────────────────────────
    if st.button("🚀 Entraîner les modèles"):

        X = df[features].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42, stratify=y
        )

        # ── Decision Tree élagué ───────────────────────────────────────────
        with st.spinner("Entraînement du Decision Tree..."):
            dt = DecisionTreeClassifier(
                max_depth=profondeur,
                min_samples_leaf=min_feuille,
                min_samples_split=min_split,
                random_state=42
            )
            dt.fit(X_train, y_train)

        # ── XGBoost ───────────────────────────────────────────────────────
        with st.spinner("Entraînement XGBoost..."):
            xgb_model = xgb.XGBClassifier(
                n_estimators=xgb_arbres,
                max_depth=xgb_depth,
                learning_rate=xgb_lr,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)

        # Sauvegarde dans session
        st.session_state.modele_dt  = dt
        st.session_state.modele_xgb = xgb_model
        st.session_state.X_test     = X_test
        st.session_state.y_test     = y_test
        st.session_state.entrainement_fait = True

        # ── Prédictions ────────────────────────────────────────────────────
        y_pred_dt  = dt.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)

        y_prob_dt  = dt.predict_proba(X_test)[:, 1]
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # ── Métriques globales ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Résultats comparatifs")

        acc_dt  = accuracy_score(y_test, y_pred_dt)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        auc_dt  = roc_auc_score(y_test, y_prob_dt)
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)

        # Cross-validation Decision Tree
        cv_dt  = cross_val_score(dt,  X, y, cv=5, scoring="roc_auc")
        cv_xgb = cross_val_score(xgb_model, X, y, cv=5, scoring="roc_auc")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy DT",   f"{acc_dt:.3f}")
        col2.metric("AUC-ROC DT",    f"{auc_dt:.3f}")
        col3.metric("Accuracy XGB",  f"{acc_xgb:.3f}")
        col4.metric("AUC-ROC XGB",   f"{auc_xgb:.3f}")

        # ── Matrices de confusion ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Matrices de confusion")
        st.markdown(f"""
        <div class="info-box">
        <b>📊 Lire une matrice de confusion</b><br>
        • <b>Vrais Positifs (VP)</b> : fraudes correctement détectées ✅<br>
        • <b>Vrais Négatifs (VN)</b> : légitimes correctement identifiés ✅<br>
        • <b>Faux Positifs (FP)</b> : légitimes classés "fraude" — alarme inutile ⚠️<br>
        • <b>Faux Négatifs (FN)</b> : fraudes non détectées — le cas le plus dangereux 🚨
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        for ax_col, y_pred, titre in [
            (col_a, y_pred_dt,  "Decision Tree élagué"),
            (col_b, y_pred_xgb, "XGBoost"),
        ]:
            with ax_col:
                cm  = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4.5, 4), facecolor=BLEU_MOYEN)
                ax.set_facecolor(BLEU_MOYEN)
                sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                            cmap="YlOrRd",
                            xticklabels=["Légitime", "Fraude"],
                            yticklabels=["Légitime", "Fraude"],
                            linewidths=1, linecolor=BLEU_FONCE)
                ax.set_xlabel("Prédit", color=BLANC)
                ax.set_ylabel("Réel",   color=BLANC)
                ax.set_title(titre, color=ACCENT, fontweight="bold")
                ax.tick_params(colors=BLANC)
                st.pyplot(fig)

        # ── Courbe ROC ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Courbe ROC")
        st.markdown(f"""
        <div class="info-box">
        <b>📈 Courbe ROC (Receiver Operating Characteristic)</b><br>
        Elle montre le compromis entre le <b>taux de détection (TPR)</b> et le <b>taux de fausses alarmes (FPR)</b>.
        Une courbe proche du coin supérieur gauche = excellent modèle.<br>
        L'<b>AUC</b> (aire sous la courbe) résume la qualité : 1.0 = parfait, 0.5 = aléatoire.
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BLEU_MOYEN)
        ax.set_facecolor(BLEU_MOYEN)

        for y_prob, label, color in [
            (y_prob_dt,  f"Decision Tree (AUC={auc_dt:.3f})",  BLEU_CLAIR),
            (y_prob_xgb, f"XGBoost (AUC={auc_xgb:.3f})",       ACCENT),
        ]:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, color=color, linewidth=2.5, label=label)

        ax.plot([0,1], [0,1], "--", color="#666", linewidth=1.5, label="Aléatoire")
        ax.set_xlabel("Taux Faux Positifs (FPR)", color=BLANC)
        ax.set_ylabel("Taux Vrais Positifs (TPR)", color=BLANC)
        ax.set_title("Courbe ROC", color=ACCENT, fontweight="bold")
        ax.tick_params(colors=BLANC)
        ax.legend(facecolor=BLEU_FONCE, labelcolor=BLANC, framealpha=0.8)
        ax.spines[:].set_color(BLEU_CLAIR)
        st.pyplot(fig)

        # ── Cross-validation ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Cross-validation (5-fold)")

        col1, col2 = st.columns(2)
        for c, scores, titre in [(col1, cv_dt, "Decision Tree"), (col2, cv_xgb, "XGBoost")]:
            with c:
                st.markdown(f"**{titre}**")
                st.markdown(f"""
                <div class="info-box">
                AUC moyen : <b style='color:{ACCENT}'>{scores.mean():.4f}</b><br>
                Écart-type : <b>{scores.std():.4f}</b><br>
                Scores : {' · '.join([f'{s:.3f}' for s in scores])}
                </div>
                """, unsafe_allow_html=True)

        # ── Importance des features ────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Importance des variables (XGBoost)")
        st.markdown(f"""
        <div class="info-box">
        L'importance mesure combien chaque variable contribue aux décisions de l'ensemble des arbres.
        Les variables avec une importance proche de 0 pourraient être retirées sans perte de performance.
        </div>
        """, unsafe_allow_html=True)

        importances = xgb_model.feature_importances_
        idx         = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(3, len(features)*0.35)), facecolor=BLEU_MOYEN)
        ax.set_facecolor(BLEU_MOYEN)
        ax.barh(
            [features[i] for i in idx[::-1]],
            importances[idx[::-1]],
            color=ACCENT, edgecolor="white", linewidth=0.3
        )
        ax.set_title("Importance des variables (XGBoost)", color=ACCENT, fontweight="bold")
        ax.tick_params(colors=BLANC, labelsize=9)
        ax.spines[:].set_visible(False)
        st.pyplot(fig)

        # ── Rapport de classification ──────────────────────────────────────
        st.markdown("---")
        st.markdown("## Rapport de classification détaillé")

        for y_pred, nom in [(y_pred_dt, "Decision Tree élagué"), (y_pred_xgb, "XGBoost")]:
            st.markdown(f"**{nom}**")
            rapport = classification_report(y_test, y_pred,
                        target_names=["Légitime", "Fraude"], output_dict=True)
            st.dataframe(pd.DataFrame(rapport).T.round(3), use_container_width=True)

        st.success("✅ Entraînement terminé ! Rendez-vous sur la page **Prédiction** pour tester le modèle.")

    elif st.session_state.entrainement_fait:
        st.info("✅ Modèles déjà entraînés. Modifiez les paramètres et relancez si besoin.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prédiction":
    st.title("Prédiction sur un nouvel exemple")

    if not st.session_state.entrainement_fait:
        st.warning("⚠️ Entraînez d'abord les modèles sur la page **Modélisation**.")
        st.stop()

    df       = st.session_state.df_propre
    features = st.session_state.features
    target   = st.session_state.target
    dt       = st.session_state.modele_dt
    xgb_model = st.session_state.modele_xgb

    st.markdown(f"""
    <div class="info-box">
    Renseignez les valeurs d'une transaction pour obtenir une prédiction en temps réel.
    Les deux modèles (Decision Tree élagué et XGBoost) donneront chacun leur verdict.
    </div>
    """, unsafe_allow_html=True)

    # ── Formulaire dynamique ───────────────────────────────────────────────
    st.markdown("## Saisir les valeurs de la transaction")
    valeurs = {}

    cols = st.columns(3)
    for i, feat in enumerate(features):
        with cols[i % 3]:
            # Déterminer si numérique ou catégoriel
            if df[feat].dtype in [np.float64, np.float32]:
                val = st.number_input(
                    feat,
                    value=float(df[feat].median()),
                    format="%.4f",
                    key=f"input_{feat}"
                )
            else:
                val = st.number_input(
                    feat,
                    value=int(df[feat].median()),
                    step=1,
                    key=f"input_{feat}"
                )
            valeurs[feat] = val

    # ── Prédiction ─────────────────────────────────────────────────────────
    if st.button("🔍 Prédire"):
        X_nouveau = np.array([[valeurs[f] for f in features]])

        pred_dt  = dt.predict(X_nouveau)[0]
        prob_dt  = dt.predict_proba(X_nouveau)[0][1]

        pred_xgb = xgb_model.predict(X_nouveau)[0]
        prob_xgb = xgb_model.predict_proba(X_nouveau)[0][1]

        st.markdown("---")
        st.markdown("## Résultat de la prédiction")

        col1, col2 = st.columns(2)

        for col, pred, prob, nom in [
            (col1, pred_dt,  prob_dt,  "Decision Tree élagué"),
            (col2, pred_xgb, prob_xgb, "XGBoost"),
        ]:
            with col:
                est_fraude = bool(pred == 1)
                couleur    = ROUGE if est_fraude else VERT
                verdict    = "🔴 FRAUDE DÉTECTÉE" if est_fraude else "🟢 TRANSACTION LÉGITIME"

                st.markdown(f"""
                <div style="
                    background: {BLEU_MOYEN};
                    border: 2px solid {couleur};
                    border-radius: 12px;
                    padding: 24px;
                    text-align: center;
                ">
                    <div style="color:#aaa; font-size:0.9rem">{nom}</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{couleur}; margin:12px 0">{verdict}</div>
                    <div style="color:{BLANC}; font-family:'IBM Plex Mono'; font-size:1.2rem">
                        Probabilité fraude : <b style="color:{ACCENT}">{prob:.1%}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Jauge de probabilité
                fig, ax = plt.subplots(figsize=(4, 0.6), facecolor=BLEU_MOYEN)
                ax.set_facecolor(BLEU_MOYEN)
                ax.barh(0, 1,     height=0.4, color="#333")
                ax.barh(0, prob,  height=0.4, color=couleur)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(["0%", "50%", "100%"], color=BLANC, fontsize=8)
                ax.spines[:].set_visible(False)
                st.pyplot(fig)

        # ── Accord des modèles ─────────────────────────────────────────────
        st.markdown("---")
        if pred_dt == pred_xgb:
            st.markdown(f"""
            <div class="info-box" style="border-left-color:{VERT}">
            ✅ <b>Les deux modèles sont d'accord</b> — La prédiction est fiable.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box" style="border-left-color:{ROUGE}">
            ⚠️ <b>Les modèles divergent</b> — Résultat incertain. On privilégie la prédiction XGBoost
            (généralement plus précise), mais une investigation manuelle est recommandée.
            </div>
            """, unsafe_allow_html=True)

        # ── Valeurs saisies (rappel) ───────────────────────────────────────
        with st.expander("Voir les valeurs saisies"):
            st.dataframe(pd.DataFrame([valeurs]), use_container_width=True)
