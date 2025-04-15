import numpy as np

def update_sequence(last_seq, current_time, pred, features, target_index):
    new_row = last_seq[-1].copy()
    new_row[target_index] = pred
    
    hour = current_time.hour + current_time.minute / 60
    weekday = current_time.weekday()
    
    hour_sin_index = features.index('heure_sin')
    hour_cos_index = features.index('heure_cos')
    jour_sin_index = features.index('jour_semaine_sin')
    jour_cos_index = features.index('jour_semaine_cos')
    
    new_row[hour_sin_index] = np.sin(2 * np.pi * hour / 24)
    new_row[hour_cos_index] = np.cos(2 * np.pi * hour / 24)
    new_row[jour_sin_index] = np.sin(2 * np.pi * weekday / 7)
    new_row[jour_cos_index] = np.cos(2 * np.pi * weekday / 7)
    
    return new_row