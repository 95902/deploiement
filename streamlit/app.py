import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go

st.set_page_config(page_title="Prédiction de consommation électrique", layout="wide")

st.title("Prédiction de consommation électrique")

# Date et heure d'entrée
col1, _ = st.columns(2)  # juste une colonne visible

with col1:
    date_part = st.date_input("Date de début :", value=datetime.now().date())
    time_part = st.time_input("Heure de début :", value="00:00")

# Combiner date et heure
start_datetime = datetime.combine(date_part, time_part)
start_date = start_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Nombre d'étapes fixé
n_steps = 48  # 24 heures

# Bouton prédire
if st.button("Prédire", type="primary"):
    with st.spinner("Calcul des prédictions en cours..."):
        try:
            # Appel à l'API
            response = requests.post("http://localhost:8000/predict/", 
                                     json={"start_date": start_date, "n_steps": n_steps})
            
            if response.status_code == 200:
                predictions = response.json()
                
                # Générer les timestamps
                timestamps = [
                    (start_datetime + timedelta(minutes=30 * i)).strftime("%Y-%m-%d %H:%M")
                    for i in range(n_steps)
                ]
                
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'consommation': predictions['predictions']
                })

                st.success("Prédictions calculées avec succès!")

                

            # Graphique 1 

                # --- Premier graphique type "pointillés + valeurs" toutes les 4h ---

                st.subheader("Prévision simple - Consommation toutes les 30 min")

                # Créer une colonne d'étiquettes vides sauf toutes les 4 heures
                df['labels'] = ""
                for i in range(0, len(df), 4):
                    df.loc[i, 'labels'] = f"{df['consommation'].iloc[i]:.0f}"

                fig1 = px.line(df, x='timestamp', y='consommation', markers=True,
                            title=f"Prévisions de consommation électrique - 24h à partir du {date_part.strftime('%d/%m/%Y')}",
                            labels={"timestamp": "Date et Heure", "consommation": "Consommation (kWh)"})

                # Ajouter les annotations à certains points (toutes les 4 heures)
                for i in range(0, len(df), 4):
                    fig1.add_annotation(
                        x=df['timestamp'].iloc[i],
                        y=df['consommation'].iloc[i],
                        text=f"{df['consommation'].iloc[i]:.0f}",
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        yshift=10
                    )

                fig1.update_traces(line=dict(color='green'), marker=dict(size=6))
                fig1.update_layout(height=800, xaxis_tickangle=-45)

                st.plotly_chart(fig1, use_container_width=True)

            #Ajout des colonnes pour l'IC à 95%
                confidence_factor = 1.0  # ou 1.28 pour 80%-90% IC
                std_error = np.std(df['consommation'])  # ou utiliser erreur réelle si connue
                df['upper'] = df['consommation'] + confidence_factor * std_error
                df['lower'] = df['consommation'] - confidence_factor * std_error
            # --- Détection des pics et creux ---
                peaks, _ = find_peaks(df['consommation'], distance=3)
                troughs, _ = find_peaks(-df['consommation'], distance=3)

            # --- Graphique avec IC + pics/creux ---
                st.subheader("Prévision avec intervalle de confiance, pics et creux")

                fig2 = go.Figure()

                # Ligne de prédiction
                fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['consommation'], mode='lines', name='Prédiction', line=dict(color='green')))

                # Intervalle de confiance
                fig2.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['upper'],
                    mode='lines', name='Borne supérieure',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig2.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['lower'],
                    mode='lines', name='Intervalle de confiance',
                    fill='tonexty', fillcolor='rgba(0,255,0,0.1)',
                    line=dict(width=0), showlegend=True
                ))

                # Pics
                fig2.add_trace(go.Scatter(
                    x=df['timestamp'].iloc[peaks], y=df['consommation'].iloc[peaks],
                    mode='markers+text', name='Pics',
                    marker=dict(color='red', size=10),
                    text=[f"{y:.1f}" for y in df['consommation'].iloc[peaks]],
                    textposition="top center"
                ))

                # Creux
                fig2.add_trace(go.Scatter(
                    x=df['timestamp'].iloc[troughs], y=df['consommation'].iloc[troughs],
                    mode='markers+text', name='Creux',
                    marker=dict(color='blue', size=10),
                    text=[f"{y:.1f}" for y in df['consommation'].iloc[troughs]],
                    textposition="bottom center"
                ))

                # Mise en forme
                fig2.update_layout(
                    height=800,
                    title="Prédictions avec Intervalle de Confiance et Détection des Pics/Creux",
                    xaxis_title="Date et Heure",
                    yaxis_title="Consommation (kWh)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=60, b=40)
                )

                st.plotly_chart(fig2, use_container_width=True)

            # Affichage csv
                st.subheader("Données de prédiction")
                st.dataframe(df)

            # csv
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Télécharger les données (CSV)",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


            else:
                st.error(f"Erreur lors de la prédiction: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Erreur de connexion à l'API: {str(e)}")
