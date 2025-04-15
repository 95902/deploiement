# 🔌 Prédiction de Consommation Électrique - Streamlit + FastAPI + LSTM

Ce projet permet de prédire la **consommation électrique** sur 24 heures (48 pas de temps de 30 minutes), en utilisant un modèle LSTM avec attention, déployé via **FastAPI** et visualisé via **Streamlit**.

---

## 📦 Technologies utilisées

- Python 3.9+
- TensorFlow / Keras (LSTM avec attention)
- FastAPI (API REST)
- Streamlit (Interface web)
- Scikit-learn (scaling des données)
- Plotly (visualisation interactive)
- SciPy (détection de pics/creux)

---

## 🚀 Structure du projet

├── app/ │ 
├── main.py # API FastAPI │ 
├── model.py # Chargement du modèle │ 
├── preprocess.py # Prétraitement et mise à jour des séquences 
│ └── ... ├── model/ 
│ ├── modele_consommation.keras 
│ ├── scaler.joblib 
│ ├── features.joblib 
│ └── last_sequences.joblib 
├── streamlit_app.py # Interface utilisateur 
├── requirements.txt 
└── README.md



---

## ⚙️ Installation

```bash
# Cloner le projet
git clone https://github.com/votre-utilisateur/nom-du-repo.git
cd nom-du-repo

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt


uvicorn app.main:app --reload


streamlit run streamlit_app.py
