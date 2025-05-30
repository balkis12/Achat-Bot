import pandas as pd
from pymongo import MongoClient


# Connexion à MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["achat_db"]
collection_data = db["data"]


# Charger les données depuis le CSV pour la collection "data"
df_data = pd.read_csv("dataframe_achats.csv")

# Insérer les données dans la collection "data"
collection_data.insert_many(df_data.to_dict("records"))
print("✅ Données insérées dans la collection 'data'.")


