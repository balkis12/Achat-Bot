from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from Data_Ingestion import load_data_from_mongodb

def build_faiss_index(persist_path="faiss_index"):
    print("📥 Chargement des documents depuis MongoDB...")
    documents = load_data_from_mongodb(split_chunks=True)
    
    if not documents:
        print("❌ Aucun document trouvé !")
        return
    
    print(f"✅ {len(documents)} documents chargés.")
    
    # 🔎 Génération des embeddings avec LaBSE 
    print("🔎 Génération des embeddings avec LaBSE...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    
    print("📦 Construction de l'index FAISS...")
    db = FAISS.from_documents(documents, embeddings)
    
    # 💾 Sauvegarde de l'index FAISS
    print("💾 Sauvegarde de l'index FAISS...")
    db.save_local(persist_path)
    
    print(f"✅ FAISS index sauvegardé localement dans : {persist_path}")

def load_faiss_index(persist_path="faiss_index"):
    print("🔄 Chargement de l'index FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    db = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    return db

if __name__ == "__main__":
    build_faiss_index()


