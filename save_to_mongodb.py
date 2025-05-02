import pandas as pd
from pymongo import MongoClient
import pickle

# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["achat_db"]
collection_data = db["data"]
collection_model = db["models"]

# Charger les données depuis le CSV pour la collection "data"
df_data = pd.read_csv("dataframe_achats.csv")

# Insérer les données dans la collection "data"
collection_data.insert_many(df_data.to_dict("records"))
print("✅ Données insérées dans la collection 'data'.")

# Charger les données de prédiction depuis le CSV pour la collection "models"
df_model = pd.read_csv("resultats_catboost.csv")

# Insérer les données de prédiction dans la collection "models"
collection_model.insert_many(df_model.to_dict("records"))
print("✅ Modèle de prédiction inséré dans la collection 'models'.")

