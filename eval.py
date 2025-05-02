from Data_Generation import get_rag_chain, generate_answer_with_context
from sentence_transformers import SentenceTransformer, util

# Initialiser SentenceTransformer
semantic_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Charger la chaîne RAG et le modèle LLM
qa_chain, llm = get_rag_chain()

# Liste des questions à tester
test_questions = [
    "Donner le meilleur Fournisseur pour les articles de type Matériel ?",   
    "Quel Fournisseur devrait- j'éviter pour les commandes urgentes ?",
]

# Fonction : Similarité sémantique
def compute_semantic_similarity(a, b):
    embedding_a = semantic_model.encode(a, convert_to_tensor=True)
    embedding_b = semantic_model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(embedding_a, embedding_b))

# Fonction : Précision
def compute_precision(final_answer, expected_answer):
    final_set = set(final_answer.lower().split())
    expected_set = set(expected_answer.lower().split())
    true_positives = final_set.intersection(expected_set)
    return len(true_positives) / len(final_set) if final_set else 0

# Fonction : Recall
def compute_recall(final_answer, expected_answer):
    final_set = set(final_answer.lower().split())
    expected_set = set(expected_answer.lower().split())
    true_positives = final_set.intersection(expected_set)
    return len(true_positives) / len(expected_set) if expected_set else 0

# Fonction : F1-score
def compute_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Générer réponse attendue à partir des documents
def generate_expected_answer_from_sources(source_docs):
    fournisseurs_info = []
    for doc in source_docs:
        lines = doc.page_content.split("\n")
        info = {}
        for line in lines:
            if "Fournisseur" in line:
                info["fournisseur"] = line.split(":")[-1].strip()
            elif "Article" in line:
                info["article"] = line.split(":")[-1].strip()
            elif "Taux de conformité" in line:
                try:
                    info["taux_conformite"] = float(line.split(":")[-1].replace('%', '').strip())
                except:
                    info["taux_conformite"] = "N/A"
            elif "Taux de respect des délais" in line:
                try:
                    info["taux_respect"] = float(line.split(":")[-1].replace('%', '').strip())
                except:
                    info["taux_respect"] = "N/A"
            elif "Score global" in line:
                try:
                    info["score"] = float(line.split(":")[-1].strip())
                except:
                    info["score"] = "N/A"
            elif "Catégorie" in line:
                info["categorie"] = line.split(":")[-1].strip()
            elif "Coût unitaire" in line:
                try:
                    info["cout_unitaire"] = float(line.split(":")[-1].replace('DT', '').strip())
                except:
                    info["cout_unitaire"] = "N/A"
            elif "Quantité prédite" in line:
                try:
                    info["quantite_predite"] = float(line.split(":")[-1].strip())
                except:
                    info["quantite_predite"] = "N/A"
        if info:
            fournisseurs_info.append(info)

    reponse = ""
    for f in fournisseurs_info:
        reponse += (
            f"Fournisseur : {f.get('fournisseur', 'Inconnu')}\n"
            f"🏷️ Article : {f.get('article', 'N/A')}\n"
            f"✅ Taux de conformité : {f.get('taux_conformite', 'N/A')}%\n"
            f"⏱️ Taux de respect des délais : {f.get('taux_respect', 'N/A')}%\n"
            f"🧮 Score global : {f.get('score', 'N/A')}\n"
            f"🗂️ Catégorie : {f.get('categorie', 'N/A')}\n"
            f"💸 Coût unitaire moyen : {f.get('cout_unitaire', 'N/A')} DT\n"
        )

    return reponse.strip()

# Boucle de test
for idx, question in enumerate(test_questions, start=1):
    print(f"\n🧪 Test {idx} : {question}")
    
    result = qa_chain.invoke({"query": question})
    source_docs = result.get("source_documents", [])
    context = "\n\n".join(doc.page_content for doc in source_docs)

    final_answer = generate_answer_with_context(llm, context, question)
    expected_answer = generate_expected_answer_from_sources(source_docs)

    # 🔹 Calcul des métriques
    sem_sim = compute_semantic_similarity(final_answer, expected_answer)
    precision = compute_precision(final_answer, expected_answer)
    recall = compute_recall(final_answer, expected_answer)
    f1_score = compute_f1(precision, recall)

    # 🔹 Affichage
    print("\n💬 Réponse générée :\n", final_answer)
    print("\n📄 Réponse attendue :\n", expected_answer)

    print("\n📊 Évaluation des performances :")
    print(f"🧠 Similarité sémantique : {sem_sim:.2f} {'✅' if sem_sim > 0.7 else '❌'}")
    print(f"📏 Précision              : {precision:.2f}")
    print(f"🔍 Rappel (Recall)        : {recall:.2f}")
    print(f"🎯 F1-score               : {f1_score:.2f}")

    print("-" * 90)
