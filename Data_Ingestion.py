from pymongo import MongoClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_keys(doc):
    """Supprime les espaces autour des clés d'un document."""
    return {k.strip(): v for k, v in doc.items()}

def clean_text(text):
    """Nettoyer le texte : supprime les nouvelles lignes et les espaces multiples."""
    if text is None:
        return "Inconnu"
    return ' '.join(str(text).strip().split())

def format_float(value):
    """Formatte un nombre à 2 chiffres après la virgule, sinon retourne 'N/A'."""
    try:
        num = float(value)
        return f"{num:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def load_data_from_mongodb(split_chunks=True):
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["achat_db"]
        print("✅ Connexion à MongoDB réussie !")
    except Exception as e:
        print(f"❌ Erreur de connexion à MongoDB : {e}")
        return []

    collection_data = db["data"]
    documents_data = collection_data.find()
    document_count = collection_data.count_documents({})
    print(f"Nombre de documents dans la collection 'data' : {document_count}")

    if document_count == 0:
        print("❌ La collection 'data' est vide.")
        return []

    raw_docs = []

    for doc in documents_data:
        doc = clean_keys(doc)

        fournisseur = clean_text(doc.get("Nom Fournisseur"))
        article = clean_text(doc.get("Article"))
        taux_conformite = format_float(doc.get("Taux de conformité (%)"))
        taux_respect = format_float(doc.get("Taux de Respect (%)"))
        score = format_float(doc.get("Score"))
        categorie = clean_text(doc.get("Catégorie", "Inconnu"))
        cout_unitaire = format_float(doc.get("Coût unitaire"))

        if fournisseur == "Inconnu" or article == "Inconnu":
            print(f"❌ Document manquant des champs nécessaires : {doc}")
            continue

        text = f"""
        📦 Fournisseur : {fournisseur}
        🏷️ Article : {article}
        ✅ Taux de conformité : {taux_conformite}%
        ⏱️ Taux de respect des délais : {taux_respect}%
        🧮 Score global : {score}
        🗂️ Catégorie : {categorie}
        💸 Coût unitaire moyen : {cout_unitaire} DT
        """

        raw_docs.append(Document(page_content=text.strip(), metadata={"source": "MongoDB"}))

    print(f"✅ Nombre de documents ajoutés à raw_docs : {len(raw_docs)}")

    if not split_chunks:
        return raw_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(raw_docs)

    return split_docs

# Appel
if __name__ == "__main__":
    documents = load_data_from_mongodb()
    print(f"Documents chargés : {len(documents)}")
    if documents:
        print("Exemple de document chargé :")
        print(documents[0].page_content)



















