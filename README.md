import nltk
import re
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer

# ============================================================
# 1. SENTIMENT LEXICONS
# ============================================================

POSITIVE_STRONG = [
    "excellent", "super", "parfait", "genial", "magnifique",
    "extra", "fantastique", "incroyable"
]

POSITIVE_WEAK = [
    "bien", "rapide", "aimable", "correct", "satisfait",
    "sympa", "agreable", "bon", "efficace"
]

NEGATIVE_STRONG = [
    "horrible", "inadmissible", "impossible", "insupportable",
    "degoutant", "catastrophique", "affreux"
]

NEGATIVE_WEAK = [
    "lent", "retard", "cher", "bug", "probleme",
    "crash", "mauvais", "dommage", "mediocre"
]

NEGATIONS = {"pas", "jamais", "aucun", "sans", "ni"}
INTENSIFIERS = {"tres": 1.5, "vraiment": 1.4, "trop": 1.6, "hyper": 1.7}

stopwords_fr = set(stopwords.words("french"))
stemmer = FrenchStemmer()


# ============================================================
# 2. ASSURANCE PROBLEM CATEGORIES (TA VERSION OPTIMISÉE)
# ============================================================

CATEGORIES = {
    # 1. DELAIS / RETARDS
    "delai_traitement": [
        "retard", "delai", "attente", "long", "lent",
        "tard", "bloque", "pas de nouvelle", "dossier bloque",
        "prise en charge lente", "trop long"
    ],

    # 2. INDEMNISATION / REMBOURSEMENT / MONTANT
    "indemnisation": [
        "indemnisation", "remboursement", "montant", 
        "dedommagement", "compensation", "trop bas",
        "pas rembourse", "franchise", "indemnite",
        "refus indemnis", "non pris en charge"
    ],

    # 3. REFUS DE PRISE EN CHARGE
    "refus_prise_en_charge": [
        "refus", "pas pris en charge", "non conforme", 
        "dossier refuse", "preuve insuffisante", "rejet",
        "preuve non acceptee", "non couvert", "garantie"
    ],

    # 4. TELEPHONE REMPLACE
    "remplacement_telephone": [
        "telephone change", "modele change", "different",
        "pas le meme", "downgrade", "inferieur", "reconditionne",
        "pas conforme", "valeur inferieure", "remplace"
    ],

    # 5. PERTE DE DONNEES
    "donnees_stockage": [
        "donnees", "stockage", "memoire", "photos perdues",
        "fichiers perdus", "efface", "data", "sauvegarde",
        "perte donnees"
    ],

    # 6. PROBLEMES TECHNIQUES
    "problemes_techniques": [
        "bug", "batterie", "chauffe", "tactile", "camera",
        "wifi", "micro", "haut parleur", "face id",
        "fingerprint", "ne marche pas", "ne fonctionne pas",
        "redemarre", "defaut", "probleme technique"
    ],

    # 7. QUALITE PIECES / REPARATION
    "qualite_piece": [
        "piece", "defectueuse", "qualite", "non originale",
        "bas de gamme", "ecran", "batterie", "pas conforme",
        "reparation mauvaise", "reparation mal faite"
    ],

    # 8. SERVICE CLIENT
    "service_client": [
        "communication", "pas informe", "pas de reponse",
        "impoli", "mauvais", "aucune info", "mensonge",
        "mauvaise experience", "promesse non tenue",
        "contact difficile"
    ],

    # 9. ADMINISTRATIF
    "administratif": [
        "facture", "imei", "garantie", "document",
        "papiers", "erreur", "contrat", "dossier incomplet",
        "information manquante"
    ]
}


# ============================================================
# 3. CLEANING
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unidecode(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation
    text = re.sub(r"\d+", " ", text)       # digits
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 4. TOKENIZE + STEM
# ============================================================

def preprocess(text):
    cleaned = clean_text(text)
    tokens = [
        t for t in nltk.word_tokenize(cleaned)
        if len(t) > 2 and t not in stopwords_fr
    ]
    stems = [stemmer.stem(t) for t in tokens]
    return tokens, stems


# ============================================================
# 5. SENTIMENT SCORING
# ============================================================

def score_sentiment(tokens):
    score = 0
    prev = ""

    for w in tokens:

        multiplier = INTENSIFIERS.get(prev, 1.0)

        if w in POSITIVE_STRONG:
            score += 2 * multiplier
        elif w in POSITIVE_WEAK:
            score += 1 * multiplier
        elif w in NEGATIVE_STRONG:
            score -= 2 * multiplier
        elif w in NEGATIVE_WEAK:
            score -= 1 * multiplier

        if prev in NEGATIONS:
            score *= -1

        prev = w

    return score


def classify_sentiment(score):
    if score >= 1:
        return "promoteur"
    elif score <= -1:
        return "detracteur"
    return "neutre"


# ============================================================
# 6. DETECT PROBLEMS
# ============================================================

def detect_problems(tokens):
    text = " ".join(tokens)
    found = []

    for category, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in tokens or kw in text:
                found.append(category)
                break

    return found if found else ["aucun"]


# ============================================================
# 7. ANALYZE ONE COMMENT
# ============================================================

def analyze_comment(text):
    tokens, stems = preprocess(text)
    score = score_sentiment(tokens)
    sentiment = classify_sentiment(score)
    problems = detect_problems(tokens)

    return {
        "clean": " ".join(tokens),
        "tokens": tokens,
        "sentiment_score": score,
        "sentiment": sentiment,
        "problems": problems
    }


# ============================================================
# 8. APPLY ON DATAFRAME → RETURNS result_comment
# ============================================================

def apply_pipeline(df, col="comment"):
    df["analysis"] = df[col].apply(analyze_comment)

    df["clean"] = df["analysis"].apply(lambda x: x["clean"])
    df["tokens"] = df["analysis"].apply(lambda x: x["tokens"])
    df["sentiment_score"] = df["analysis"].apply(lambda x: x["sentiment_score"])
    df["sentiment"] = df["analysis"].apply(lambda x: x["sentiment"])
    df["problems"] = df["analysis"].apply(lambda x: x["problems"])

    result_comment = df.drop(columns=["analysis"]).copy()
    result_comment = result_comment.rename(columns={col: "comment_original"})

    return result_comment




how to use 


df = pd.DataFrame({
    "comment": [
        "Délai trop long, aucune information du service client.",
        "J'ai perdu mes données et reçu un téléphone inférieur.",
        "Très bon service, rapide et efficace."
    ]
})


run pipeline 
result_comment = apply_pipeline(df, "comment")
print(result_comment)