import cv2
import easyocr
import re
import os
import sys
import requests
from fuzzywuzzy import fuzz

# --- INITIALISATION DE L'IA ---
print("⏳ Chargement du modèle IA (EasyOCR)...")
reader = easyocr.Reader(['ar', 'en'], gpu=False) 

# ===================================================================
# API DE TRANSLITTÉRATION PHONÉTIQUE
# ===================================================================
def translitteration_franco_arabe(mot):
    if re.search(r'[\u0600-\u06FF]', mot) or mot.isdigit():
        return mot
    url = f"https://inputtools.google.com/request?text={mot}&itc=ar-t-i0-und&num=3"
    try:
        reponse = requests.get(url).json()
        if reponse[0] == "SUCCESS":
            return reponse[1][0][1][0]
    except Exception as e:
        pass
    return mot

# ===================================================================
# NORMALISATEUR ARABE
# ===================================================================
def normaliser_arabe(texte):
    if not texte: return ""
    texte = texte.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    texte = texte.replace("ة", "ه")
    texte = re.sub(r'\bال', '', texte)
    return texte

# ===================================================================
# TÂCHE 1 : FILTRE DE QUALITÉ
# ===================================================================
def detecter_flou(image, seuil=30.0):
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score_nettete = cv2.Laplacian(gris, cv2.CV_64F).var()
    return score_nettete >= seuil, score_nettete

# ===================================================================
# TÂCHE 2 : EXTRACTION INTELLIGENTE (AVEC ZOOM PHYSIQUE 🔍)
# ===================================================================
def extraire_texte_intelligent(image):
    image_agrandie = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    h, w = image_agrandie.shape[:2]
    gris = cv2.cvtColor(image_agrandie, cv2.COLOR_BGR2GRAY)
    
    if h > w:
        angles_a_tester = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
        noms_angles = ["90° (Tournée à droite)", "270° (Tournée à gauche)"]
    else:
        angles_a_tester = [None, cv2.ROTATE_180]
        noms_angles = ["0° (Normale)", "180° (À l'envers)"]
        
    meilleur_score = -1
    meilleur_texte = ""
    angle_choisi = ""
    
    for i, angle in enumerate(angles_a_tester):
        img_test = gris if angle is None else cv2.rotate(gris, angle)
            
        try:
            resultats = reader.readtext(img_test)
        except Exception as e:
            return None, str(e)
            
        score_actuel = sum([prob for (bbox, text, prob) in resultats if prob > 0.10])
        
        if score_actuel > meilleur_score:
            meilleur_score = score_actuel
            angle_choisi = noms_angles[i]
            
            texte_complet = [text for (bbox, text, prob) in resultats if prob > 0.10]
            meilleur_texte = " ".join(texte_complet).lower()
            meilleur_texte = re.sub(r'\s+', ' ', meilleur_texte).strip()
            
    print(f"🔄 [Vision] Orientation trouvée par l'IA : {angle_choisi}")
    return meilleur_texte, None

# ===================================================================
# TÂCHE 3 : MOTEUR DE CROISEMENT (NOUVEAU SYSTÈME DE SCORING ⚖️)
# ===================================================================
def verify_data(texte_propre, infos_candidat):
    anomalies = []
    score = 100.0 # On utilise des décimales pour un calcul précis
    texte_propre_norm = normaliser_arabe(texte_propre)
    
    infos_utiles = {k: v for k, v in infos_candidat.items() if v and str(v).strip() != ""}
    if not infos_utiles: return 0, ["Aucune information."]
        
    # ✨ LE NOUVEAU CALCUL DES POINTS
    champs_cin = [k for k in infos_utiles.keys() if k.lower() in ['cin', 'id', 'identifiant', 'passeport']]
    champs_autres = [k for k in infos_utiles.keys() if k.lower() not in ['cin', 'id', 'identifiant', 'passeport']]
    
    # 50 points pour le CIN, et 50 points répartis sur les autres champs
    penalty_cin = 50.0 / len(champs_cin) if champs_cin else 0
    penalty_autres = 50.0 / len(champs_autres) if champs_autres else 0

    # Sécurité : si on n'a que des CIN ou que du texte, on remet sur 100
    if not champs_cin: penalty_autres = 100.0 / len(champs_autres)
    if not champs_autres: penalty_cin = 100.0 / len(champs_cin)
    
    for key, expected_value in infos_utiles.items():
        if key in champs_cin:
            clean_expected = re.sub(r'\s+', '', str(expected_value))
            numeros_trouves = re.findall(r'\d{7,9}', re.sub(r'\s+', '', texte_propre))
            
            if clean_expected not in numeros_trouves and clean_expected not in texte_propre.replace(" ", ""):
                anomalies.append(f"CIN '{expected_value}' introuvable.")
                score -= penalty_cin
        else:
            valeur_phonetique_arabe = translitteration_franco_arabe(str(expected_value).lower())
            valeur_phonetique_norm = normaliser_arabe(valeur_phonetique_arabe)
            
            print(f"🌐 [Phonétique] '{expected_value}' -> transformé en: '{valeur_phonetique_arabe}' -> IA cherche: '{valeur_phonetique_norm}'")

            match_score_1 = fuzz.token_set_ratio(valeur_phonetique_norm, texte_propre_norm)
            
            meilleur_score_mot = 0
            for mot_lu in texte_propre_norm.split():
                score_mot = fuzz.ratio(valeur_phonetique_norm, mot_lu)
                if score_mot > meilleur_score_mot:
                    meilleur_score_mot = score_mot
                    
            match_score = max(match_score_1, meilleur_score_mot)
            
            # Seuil dynamique : 45 pour les petits mots, 60 pour les grands
            seuil_exige = 45 if len(valeur_phonetique_norm) <= 4 else 60
            
            if match_score < seuil_exige: 
                anomalies.append(f"{key.capitalize()} '{expected_value}' (cherché comme: {valeur_phonetique_arabe}) introuvable ou illisible.")
                score -= penalty_autres

    # On arrondit le score à la fin pour avoir un entier propre (ex: 83 au lieu de 83.3333)
    return max(0, int(round(score))), anomalies

# ===================================================================
# ORCHESTRATEUR
# ===================================================================
def analyser_document_ia(image_path, infos_candidat):
    result = {"score": 0, "anomalies": [], "status": "Échec", "message": "", "texte_brut": "", "score_nettete": 0.0}
    
    image = cv2.imread(image_path)
    if image is None:
        result["message"] = "Impossible de lire le fichier image."
        return result

    est_nette, score_nettete = detecter_flou(image, seuil=30.0)
    result["score_nettete"] = round(score_nettete, 2)
    if not est_nette:
        result["status"] = "Rejeté (Flou)"
        return result

    print("\n--- ÉTAPE 2 : Redressement & Lecture du texte ---")
    texte_propre, erreur = extraire_texte_intelligent(image)
    if erreur:
        result["message"] = f"Erreur OCR : {erreur}"
        return result
    result["texte_brut"] = texte_propre

    print("\n--- ÉTAPE 3 : Croisement Phonétique ---")
    score, anomalies = verify_data(texte_propre, infos_candidat)
    
    result["score"] = score
    result["anomalies"] = anomalies
    
    # Nouveau barème de statuts !
    if score == 100: result["status"] = "Conforme"
    elif score >= 50: result["status"] = "Partiellement Conforme"
    else: result["status"] = "Non Conforme"
        
    return result

# ===================================================================
# SCRIPT D'ENTRÉE (ZONE DE TEST)
# ===================================================================
if __name__ == "__main__":
    print("=========================================")
    print("🤖 ANALYSEUR IA MULTILINGUE & AUTO-ROTATIF (PURE EASYOCR)")
    print("=========================================\n")

    chemin_image = "ma_carte2.jpg" 
    
    donnees_candidat = {
        "cin": "14373619",   
        "nom": "haboubi",   
        "prenom": "syrine",
        "lieu_naissance": "tunis" 
    }

    if not os.path.exists(chemin_image):
        print(f"❌ Erreur : L'image '{chemin_image}' est introuvable.")
        sys.exit(1)

    resultat = analyser_document_ia(chemin_image, donnees_candidat)

    print("\n=== BILAN DE CONFORMITÉ FINALE ===")
    print(f"Vision (Netteté) : {resultat['score_nettete']}")
    print(f"Score Global     : {resultat['score']}/100")
    print(f"Statut           : {resultat['status']}")
    
    if resultat['anomalies']:
        print("\n⚠️ Anomalies détectées :")
        for ano in resultat['anomalies']:
            print(f"  - {ano}")
    elif resultat['status'] == "Conforme":
        print("\n✅ Dossier 100% conforme ! ")
        
    print(f"\n[DEBUG] Texte vu par l'IA : {resultat['texte_brut']}")
    print("=========================================")