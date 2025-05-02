from pymongo import MongoClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_keys(doc):
    """Supprime les espaces autour des clés d'un document."""
    return {k.strip(): v for k, v in doc.items()}

def clean_text(text):
    """Supprime les nouvelles lignes et les espaces multiples dans un texte."""
    if text is None:
        return "Inconnu"
    return ' '.join(text.strip().split())

def load_data_from_mongodb(split_chunks=True):
    # Connexion à MongoDB
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["achat_db"]
        print("✅ Connexion à MongoDB réussie !")
    except Exception as e:
        print(f"❌ Erreur de connexion à MongoDB : {e}")
        return []

    collection_data = db["data"]
    collection_model = db["models"]

    # Compter les documents dans la collection models
    model_count = collection_model.count_documents({})
    print(f"Nombre de documents dans la collection 'models' : {model_count}")

    documents_data = collection_data.find()
    document_count = collection_data.count_documents({})

    if document_count == 0:
        print("❌ La collection 'data' est vide.")
        return []

    raw_docs = []
    problematic_docs = []

    for i, doc in enumerate(documents_data):
        doc = clean_keys(doc)  # Nettoyer les clés

        fournisseur = clean_text(doc.get("Nom Fournisseur", None))
        article = clean_text(doc.get("Article", None))
        taux_conformite = doc.get("Taux de conformité (%)", "N/A")
        taux_respect = doc.get("Taux de Respect (%)", "N/A")
        score = doc.get("Score", "N/A")
        categorie = doc.get("Catégorie", "Inconnu")
        cout_unitaire = doc.get("Coût unitaire", "N/A")

        # 🆕 Ajout des métadonnées essentielles
        metadata = {
            "source": "MongoDB",
            "doc_id": str(i),
            "Article": article,  # 📌 Champ critique pour RAGAS
            "Catégorie": categorie,  # 📌 Champ critique pour RAGAS
            "Fournisseur": fournisseur  # 📌 Champ critique pour RAGAS
        }

        if fournisseur == "Inconnu" or article == "Inconnu":
            problematic_docs.append({
                "doc_index": i,
                "doc": doc,
                "error": "Missing Nom Fournisseur or Article"
            })
            continue

        article_fournisseur_key = f"{article} - {fournisseur}"
        prediction_doc = collection_model.find_one({"Article_Fournisseur": article_fournisseur_key})
        quantite_predite = prediction_doc.get("Quantité_Prédite", "Inconnue") if prediction_doc else "Inconnue"

        text = f"""
📦 Fournisseur : {fournisseur}
🏷️ Article : {article}
✅ Taux de conformité : {taux_conformite}
⏱️ Taux de respect des délais : {taux_respect}
🧮 Score global : {score}
🗂️ Catégorie : {categorie}
💸 Coût unitaire moyen : {cout_unitaire} DT
📊 Quantité Prédite : {quantite_predite}
"""

        raw_docs.append(Document(
            page_content=text.strip(),
            metadata=metadata  # 📦 Maintenant avec métadonnées essentielles
        ))

    print(f"✅ Nombre de documents ajoutés à raw_docs : {len(raw_docs)}")
    print(f"⚠️ Nombre de documents problématiques : {len(problematic_docs)}")

    # Save problematic documents
    with open("problematic_ingestion_docs.txt", "w", encoding="utf-8") as f:
        for prob in problematic_docs:
            f.write(f"Document {prob['doc_index']}:\n")
            f.write(f"Content: {prob['doc']}\n")
            f.write(f"Error: {prob['error']}\n")
            f.write("-" * 50 + "\n")

    if not split_chunks:
        return raw_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(raw_docs)
    return split_docs

if __name__ == "__main__":
    documents = load_data_from_mongodb()
    print(f"Documents chargés : {len(documents)}")
    if documents:
        print("Exemple de document chargé :")
        print(documents[0].page_content)
















