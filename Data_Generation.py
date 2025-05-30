from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM as Ollama  # ✅ Mise à jour
from Data_Indexing_Storage import load_faiss_index
import json
import os


# ⛓️ Création de la chaîne RAG avec FAISS et Ollama
def get_rag_chain():
    vectorstore = load_faiss_index("faiss_index")
    retriever = vectorstore.as_retriever()

    llm = Ollama(model="llama3.2:1b")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain, llm  # ✅ On retourne deux objets



# 💬 Génération d'une réponse enrichie à partir du contexte + question
def generate_answer_with_context(llm, context, question):

    full_prompt = f"""
Tu es un assistant spécialisé en optimisation des achats. En te basant sur les données ci-dessous, réponds clairement à la question de l'utilisateur.

Données :
{context}

Question :
{question}

📝 Inclue dans ta réponse si possible :
- Score global
- Taux de conformité
- Taux de respect des délais
- Catégorie (fiable, moyen, risqué)
- Coût unitaire
"""
    response = llm.invoke(full_prompt)
    return response



# 🟩 1. Initialise liste vide fi début mta3 __main__ # Save pour RAGAS
data = []

if __name__ == "__main__":
    # ✅ Correction ici : on sépare les deux objets
    qa_chain, llm = get_rag_chain()

    user_question = "Quel est le meilleur fournisseur le plus fiable pour les articles de type Matériel ?"
    result = qa_chain.invoke({"query": user_question})

    # ✅ On extrait les documents sources
    context = "\n\n".join([doc.page_content for doc in result['source_documents']])

    # ✅ Réutilisation du LLM pour le prompt enrichi
    final_answer = generate_answer_with_context(llm, context, user_question)

    print("💬 Réponse enrichie :\n", final_answer)

 # 🟩 2. Ajoute les données à la liste # Save pour RAGAS
    data.append({
        "question": user_question,
        "context": context,
        "answer": final_answer
    })

    print("Données ajoutées à la liste:", data)  # Debug pour vérifier les données ajoutées

    # 🟩 3. Vérifie si le fichier existe déjà # Save pour RAGAS
    if os.path.exists("evaluation_data.json"):
        print("Le fichier 'evaluation_data.json' existe déjà.")
    else:
        print("Le fichier 'evaluation_data.json' n'existe pas encore.")

    # 🟩 4. Écris les résultats dans un fichier JSON # Save pour RAGAS
    try:
        with open("evaluation_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("Fichier 'evaluation_data.json' sauvegardé avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'écriture dans le fichier: {e}")