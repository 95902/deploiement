from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import logging
from app.model import load_model, make_prediction
from app.preprocess import update_sequence

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de prédiction de consommation électrique",
    description="API pour prédire la consommation électrique future",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes de Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines en développement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle au démarrage
try:
    logger.info("Chargement du modèle...")
    model, scaler, features, target_index = load_model()
    logger.info("Modèle chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    raise

class PredictionRequest(BaseModel):
    start_date: str
    n_steps: int

@app.get("/")
async def root():
    return {"message": "API de prédiction de consommation électrique"}

@app.post("/predict/")
async def get_prediction(request: PredictionRequest):
    try:
        logger.info(f"Nouvelle requête: date={request.start_date}, n_steps={request.n_steps}")
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d %H:%M:%S")
        n_steps = request.n_steps
        
        # Obtenir les prédictions
        future_denorm = make_prediction(start_date, n_steps, model, scaler, features, target_index)
        
        logger.info(f"Prédiction réussie: {len(future_denorm)} valeurs générées")
        return {"predictions": future_denorm}
    
    except ValueError as e:
        logger.error(f"Erreur de format: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Format de date invalide: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))