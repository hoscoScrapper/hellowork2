import time
import random
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime
from selenium import webdriver
from time import sleep
import random
import os
from selenium.webdriver.chrome.service import Service
#from webdriver_manager.chrome import ChromeDriverManager
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import unicodedata
import re
import asyncio
import httpx
import nest_asyncio
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import json
import os
from openai import OpenAI
import instructor
from enum import Enum
from pydantic import BaseModel, Field, field_validator, constr, ValidationError, model_validator
from typing import List, Optional, Any
import math


# Définir les paramètres de recherche
url_base = f"https://www.hellowork.com/fr-fr/emploi/recherche.html?k=Restauration&k_autocomplete=&l=&l_autocomplete=&st=date&d=all"


# Configurer Selenium avec undetected_chromedriver
options = uc.ChromeOptions()
options.add_argument('--headless=new')  # Optionnel : exécuter sans ouvrir le navigateur
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.7390.61 Safari/537.36')

# Initialiser le driver avec undetected_chromedriver
driver = uc.Chrome(options=options)

def extraire_offres(limit=10):
    offres_totales = []
    date_scraping = datetime.datetime.now().strftime("%Y-%m-%d")
    start = 1

    try:
        while len(offres_totales) < limit:
            url = f"{url_base}&p={start}"
            print(f"Scraping page {start} from URL: {url}")
            driver.get(url)
            sleep(random.uniform(2, 4))  # pause légère avant cookie popup

            # 👉 Try clicking "Tout accepter" cookie button
            try:
                wait = WebDriverWait(driver, 5)
                accepter_btn = wait.until(
                    EC.element_to_be_clickable((By.ID, "hw-cc-notice-accept-btn"))
                )
                accepter_btn.click()
                print("Cookies popup accepted.")
                sleep(1)
            except Exception:
                print("No cookie popup.")
                pass

            sleep(random.uniform(3, 5))  # laisser charger la page après popup

            wait = WebDriverWait(driver, 10)

            # Wait for offers to load
            offres = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li[data-id-storage-item-id]")))
            
            print("Nombre d'offres trouvées:", len(offres))

            for offre in offres:
                try:
                    link = offre.find_element(By.XPATH, ".//a[contains(@href, '/fr-fr/emplois/')]")
                    url_offre = link.get_attribute("href")
                except:
                    url_offre = "N/A"

                offres_totales.append({'url': url_offre})


            if len(offres_totales) < limit:
                start += 1
                sleep(random.uniform(1, 2))

    finally:
        driver.quit()

    return offres_totales


resultats_part1 = extraire_offres(limit=500)
resultats_part1 = pd.DataFrame(resultats_part1)
job_urls = resultats_part1.url.tolist()


#------------------------CHECK DUPLICATES URL DANS BIGQUERY--------------------------------------------------

from google.cloud import bigquery
import pandas as pd
from google.oauth2 import service_account


# Load JSON from GitHub secret
key_json = json.loads(os.environ["BIGQUERY"])

# Create credentials from dict
credentials = service_account.Credentials.from_service_account_info(key_json)

# Initialize BigQuery client
client = bigquery.Client(
    credentials=credentials,
    project=key_json["project_id"]
)

# Query existing URLs from your BigQuery table
query = """
    SELECT url
    FROM `databasealfred.jobListings.hellowork`
    WHERE url IS NOT NULL
"""
query_job = client.query(query)

# Convert results to a set for fast lookup
existing_urls = {row.url for row in query_job}

print(f"Loaded {len(existing_urls)} URLs from BigQuery")

# Filter job_urls
job_urls = [url for url in job_urls if url not in existing_urls]

print(f"✅ Remaining job URLs to scrape: {len(job_urls)}")

#------------------------ FIN CHECK DUPLICATES URL DANS BIGQUERY--------------------------------------------------


# Setup undetected Chrome driver
options = uc.ChromeOptions()
options.add_argument('--headless')
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.7390.61 Safari/537.36')


# Launch the driver
driver = uc.Chrome(options=options)

# Function to extract text safely
def get_text(selector, multiple=False):
    try:
        if multiple:
            return [elem.text.strip() for elem in driver.find_elements(By.CSS_SELECTOR, selector)]
        return driver.find_element(By.CSS_SELECTOR, selector).text.strip()
    except NoSuchElementException:
        return "" if not multiple else []

# Initialize list to store job data
job_data = []

for i, job_url in enumerate(job_urls):
    driver.get(job_url)
    time.sleep(random.uniform(3, 4))  # Random delay for human-like behavior

    #entreprise = get_text("div.css-1x55bdz a")
    title = get_text("h1#main-content [data-cy='jobTitle']")
    entreprise = get_text("h1#main-content span a")
    #details = get_text('span.tw-inline-flex.tw-typo-m.tw-text-grey-500', multiple=True)
    #location = details[0] if len(details) > 0 else ""
    details = get_text("ul[data-cy='tags-resume'] li:first-child", multiple=True)
    location = details[0] if details else ""
    type_contrat = details[1] if len(details) > 1 else ""
    #type_contrat = details[1] if len(details) > 1 else ""
    temps_plein = details[2] if len(details) > 2 else ""    

    
    # Get all <li> texts
    items = get_text("ul.tw-flex.tw-flex-wrap.tw-gap-3 li", multiple=True)

    salaire = ""
    tags = []

    for item in items:
        if not salaire and "€" in item:
            salaire = item
        else:
            tags.append(item)

    # Optional: clean salary (e.g. remove non-breaking spaces)
    salaire = salaire.replace("\u202f", " ").strip()

    description = driver.find_element(By.CSS_SELECTOR,'div[data-truncate-text-target="content"] p').text.strip()
    
    # Get complementary info (second section)
    complementary_info = driver.find_elements(By.XPATH, '//span[text()="Les avantages"]/ancestor::details//ul/li')
    complementary_info = [a.text.strip() for a in complementary_info]
    complementary_info = "\n".join(complementary_info)


    # Append with a line break if complementary info exists
    if complementary_info:
        description += "\n\n" + complementary_info
        
    date_scraping = datetime.datetime.now().strftime("%Y-%m-%d")

    # Extract job tags (contract type, etc.)
    tags = [t.replace("\u202f", " ").strip() for t in tags]
    if temps_plein.strip() != "":
        tags.append(temps_plein)
    tags = ', '.join([t for t in tags if t])

    # Append extracted data to list
    job_data.append({
        "titre": title,
        "localisation": location,
        "entreprise": entreprise,
        "salaire": salaire,
        "description": description,
        "date_scraping": date_scraping,
        "Tags": tags,
        "type_contrat": type_contrat,
        
        
    })

driver.quit()

# Convert list to Pandas DataFrame
resultats_part2 = pd.DataFrame(job_data)

#Concat axis 0 resultats_part1 and resultats_part2
df_jobs = pd.concat([resultats_part1, resultats_part2], axis=1)

print(df_jobs.description)

# Convert scraped results into a DataFrame
new_data = df_jobs

# Apply nest_asyncio to fix event loop issue in Jupyter
#nest_asyncio.apply()

# Data Gouv API URL
API_URL = "https://api-adresse.data.gouv.fr/search"

# Function to call API asynchronously with retries
async def get_geodata(client, address, retries=3):
    params = {"q": address, "limit": 1}

    for attempt in range(retries):
        try:
            response = await client.get(API_URL, params=params, timeout=5)

            if response.status_code == 503:  # Server overloaded
                print(f"503 Error - Retrying {address} (Attempt {attempt+1})...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

            response.raise_for_status()  # Raise error if response is bad
            data = response.json()

            if data["features"]:
                props = data["features"][0]["properties"]
                geo = data["features"][0]["geometry"]["coordinates"]

                ville = props.get("city", "")
                code_postal = props.get("postcode", "")
                longitude = geo[0] if geo else None
                latitude = geo[1] if geo else None
                contexte = props.get("context", "")

                # Extract region name (after second comma)
                region = contexte.split(", ")[-1] if contexte.count(",") >= 2 else ""

                return ville, code_postal, longitude, latitude, region
        
        except Exception as e:
            print(f"Error fetching data for {address} (Attempt {attempt+1}): {e}")
        
        await asyncio.sleep(2 ** attempt)  # Exponential backoff for retries

    return None, None, None, None, None  # Return empty values if all retries fail

# Async function to process all addresses with rate limiting
async def process_addresses(address_list, delay_between_requests=0.017):  # 1/60 = ~0.017s
    results = []
    async with httpx.AsyncClient() as client:
        for i, address in enumerate(address_list):
            result = await get_geodata(client, address)
            results.append(result)
            
            print(f"Processed {i + 1} / {len(address_list)}")

            # Respect 60 requests per second limit
            await asyncio.sleep(delay_between_requests)  

    return results

# Run API calls asynchronously
addresses = new_data["localisation"].tolist()
geodata_results = asyncio.run(process_addresses(addresses))

# Assign the results to the DataFrame
new_data[["Ville", "Code Postal", "Longitude", "Latitude", "Region"]] = pd.DataFrame(geodata_results)

# Add "France Travail" column
new_data["Source"] = "hellowork"

print(f"Post geo new data Check length {len(new_data)}")

# -------- DEBUT CHATGPT DATA ENRICHMENT --------------------------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_ai = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class Loge(str, Enum):
    LOGE = "Logé"
    NON_LOGE = "Non Logé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class TypeContrat(str, Enum):
    CDD = "CDD"
    CDI = "CDI"
    STAGE = "Stage"
    APPRENTISSAGE = "Apprentissage"
    INTERIM = "Interim"
    EXTRA = "Extra"
    SAISONNIER = "Saisonnier"
    ALTERNANCE = "Alternance"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CoupleAccepte(str, Enum):
    ACCEPTE = "Couple accepté"
    NON_ACCEPTE = "Couple non accepté"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieEtablissement(str, Enum):
    GASTRONOMIQUE = "Gastronomique"
    BRASSERIE = "Brasserie"
    BAR = "Bar"
    RAPIDE = "Restauration rapide"
    COLLECTIVE = "Restauration collective"
    RESTAURANT = "Restaurant"
    HOTEL_LUXE = "Hôtel luxe"
    HOTEL = "Hôtel"
    CAMPING = "Camping"
    CAFE = "Café/Salon de thé"
    BOULANGERIE = "Boulangerie/Patisserie"
    ETOILE = "Etoile Michelin"
    PALACE = "Palace"
    TRAITEUR = "Traiteur/Événementiel/Banquet"
    SPA = "Spa"
    LABORATOIRE = "Laboratoire"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob1(str, Enum):
    RESTAURATION = "Restauration"
    HOTELLERIE = "Hôtellerie"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"
    GARBAGE = "Support & Back-office"


class CategorieJob2(str, Enum):
    SALLE = "Salle & Service"
    DIRECTION = "Direction & Management"
    SUPPORT = "Support & Back-office"
    CUISINE = "Cuisine"
    SPA = "Spa & Bien-être"
    ETAGES = "Étages & Housekeeping"
    BAR = "Bar & Sommellerie"
    RECEPTION = "Réception & Hébergement"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob3(str, Enum):
    CHEF_EXECUTIF = "Chef exécutif"
    CHEF_CUISINE = "Chef de cuisine"
    SOUS_CHEF = "Sous-chef"
    CHEF_PARTIE = "Chef de partie"
    COMMIS_CUISINE = "Commis de cuisine"
    PATISSIER = "Pâtissier"
    BOULANGER = "Boulanger"
    PIZZAIOLO = "Pizzaiolo"
    TRAITEUR = "Traiteur"
    MANAGER = "Manager / Responsable"
    EMPLOYE = "Employé polyvalent"
    PLONGEUR = "Plongeur"
    STEWARD = "Steward"
    DIRECTEUR = "Directeur"
    RESPONSABLE_SALLE = "Responsable de salle"
    MAITRE_HOTEL = "Maître d’hôtel"
    CHEF_RANG = "Chef de rang"
    COMMIS_SALLE = "Commis de salle / Runner"
    SERVEUR = "Serveur"
    SOMMELIER = "Sommelier"
    BARMAN = "Barman"
    BARISTA = "Barista"
    RECEPTIONNISTE = "Réceptionniste / Hôte d’accueil"
    CONCIERGE = "Concierge"
    BAGAGISTE = "Bagagiste / Voiturier"
    VALET = "Valet / Femme de chambre"
    MARKETING = "Marketing / Communication"
    AGENT_RESERVATIONS = "Agent de réservations"
    REVENUE_MANAGER = "Revenue manager"
    GOUVERNANT = "Gouvernant(e)"
    SPA_PRATICIEN = "Spa praticien(ne) / Ésthéticien(ne)"
    COACH = "Coach sportif"
    MAITRE_NAGEUR = "Maître-nageur"
    ANIMATION = "Animation / Événementiel"
    COMMERCIAL = "Commercial"
    RH = "RH / Paie"
    COMPTABILITE = "Comptabilité / Contrôle de gestion"
    TECHNICIEN = "Technicien / Maintenance"
    IT = "IT / Data"
    HACCP = "HACCP manager"
    CUISINIER = "Cuisinier"
    LIMONADIER = "Limonadier"
    ALLOTISSEUR = "Allotisseur"
    APPROVISIONNEUR = "Approvisionneur / Économe"
    AGENT_SECURITE = "Agent de sécurité"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Urgent(str, Enum):
    URGENT = "Urgent"
    NON_URGENT = "Non Urgent"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class Environnement(str, Enum):
    CENTRE_VILLE = "Centre ville"
    BORD_MER = "Bord de mer"
    MONTAGNE = "Montagne"
    BANLIEUE = "Banlieue"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class ChaineIndependant(str, Enum):
    CHAINE = "Chaine"
    INDEPENDANT = "Indépendant"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class TempsTravail(str, Enum):
    PLEIN_TEMPS = "Plein temps"
    TEMPS_PARTIEL = "Temps partiel"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class HorairesTravail(str, Enum):
    JOUR = "Jour"
    NUIT = "Nuit"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Experience(str, Enum):
    DEBUTANT = "Débutant"
    CONFIRME = "Confirmé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class DureeModel(BaseModel):
    value: str


class HeuresParSemaineModel(BaseModel):
    heures: Optional[int] = None

    # v2 field validator
    @field_validator("heures", mode="before")
    def parse_heures(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            match = re.search(r"\d+", v)
            if match:
                return int(match.group())
        return None

class DateDebutModel(BaseModel):
    value: str

class SalaireModel(BaseModel):
    value: str

# --- Base model that ties everything together ---
class JobClassification(BaseModel):
    IA_Logé: Loge
    IA_Type_de_contrat: TypeContrat
    IA_Salaire: SalaireModel
    IA_Couple_accepté: CoupleAccepte
    IA_Catégorie_établissement: CategorieEtablissement
    IA_Catégorie_Job_1: CategorieJob1
    IA_Catégorie_Job_2: CategorieJob2
    IA_Catégorie_Job_3: CategorieJob3
    IA_Urgent: Urgent
    IA_Date_de_début: DateDebutModel
    IA_Durée: DureeModel
    IA_Type_environnement: Environnement
    IA_Chaine_Indépendant: ChaineIndependant
    IA_Temps_de_travail: TempsTravail
    IA_Horaires_de_travail: HorairesTravail
    IA_Heures_par_semaine: HeuresParSemaineModel
    IA_Éxpérience: Experience

SYSTEM_PROMPT = """You are a classifier for job listings in the hospitality industry in France. You are an expert and absolutely have to respect the 
instructions. Each category can ONLY take one the value that are specified for it.
The success of my business depends on you so double check!!
    "IA_Logé": when accomodation or help with accomodation is provided "Logé" else "Non logé",
        "IA_Type_de_contrat": it MUST BE one of ["CDD", "CDI", "Stage", "Apprentissage", "Interim", "Extra", "Saisonnier", "Alternance"],
        "IA_Salaire": the highest salary offered in format "X€/heure" or "X€/mois" or "X€/an", or "" if not specified,
        "IA_Couple_accepté": "Couple accepté" or "",
    	"IA_Catégorie_établissement": it MUST BE one of the following and CANNOT be empty ["Gastronomique","Brasserie","Bar","Restauration rapide","Restauration collective","Restaurant","Hôtel luxe","Hôtel","Camping","Café/Salon de thé”,”Boulangerie/Patisserie”,”Etoile Michelin","Palace”, “Traiteur/Événementiel/Banquet”,“Spa”, “Laboratoire”],
    	"IA_Catégorie_Job_1":  it MUST BE one of the following and it cannot be empty [“Restauration”, “Hôtellerie”],
    	“IA_Catégorie_Job_2”:  it MUST BE one of and the most relevant, it cannot be empty [“Salle & Service”, “Direction & Management”, “Support & Back-office”, “Cuisine”, “Spa & Bien-être”, “Étages & Housekeeping”, “Bar & Sommellerie”, “Réception & Hébergement”],
        “IA_Catégorie_Job_3”: it has to be one of the following and the most relevant, it cannot be empty ["Chef exécutif","Chef de cuisine","Sous-chef","Chef de partie","Commis de cuisine","Pâtissier","Boulanger","Pizzaiolo","Traiteur","Manager / Responsable","Employé polyvalent","Plongeur","Steward","Directeur","Responsable de salle","Maître d’hôtel","Chef de rang","Commis de salle / Runner","Serveur","Sommelier","Barman","Barista","Réceptionniste / Hôte d’accueil","Concierge","Bagagiste / Voiturier","Valet / Femme de chambre","Marketing / Communication","Agent de réservations","Revenue manager","Gouvernant(e)","Spa praticien(ne) / Ésthéticien(ne)","Coach sportif","Maître-nageur","Animation / Événementiel","Commercial","RH / Paie","Comptabilité / Contrôle de gestion","Technicien / Maintenance","IT / Data","HACCP manager","Cuisinier","Limonadier","Allotisseur","Approvisionneur / Économe","Agent de sécurité"],
    	"IA_Urgent": "Urgent" or "", it takes "Urgent" only when the starting date is within 2 weeks of the date_scraping or when it is explicitly mentioned in the description
        "IA_Date_de_début": starting date in format YYYY-MM-DD if present, else "",
        "IA_Durée": contract duration like "N days", "N weeks", "N months", or "Indéfini",
        "IA_Type_environnement”: one of ["Centre ville","Bord de mer","Montagne","Banlieue"],
    	“IA_Chaine_Indépendant”: when the company posting the job listing is part of a group or bigger company "Chaine", else ""
        "IA_Temps_de_travail": "Plein temps" or "Temps partiel",
        "IA_Horaires_de_travail": "Jour" or "Nuit",
        "IA_Heures_par_semaine": return a number not a string ! the number of hours worked per week if available, when the contract is less than a week just put how many hours it , else “”,
    	“IA_Éxpérience” one the following [“Débutant”, “Confirmé”]

    Strictly output without explanations."""


def classify_job_listing(ticket_text: str) -> JobClassification:
    response = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            max_retries=3,
            response_model=JobClassification,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ticket_text}
            ],
            temperature=0
        )
    return response

# Convert each row into a single string with "col":"value" format
new_data["row_as_string"] = new_data.apply(
    lambda row: ", ".join([f'"{col}":"{row[col]}"' for col in new_data.columns]),
    axis=1
)

# Apply your classify_job_listing function to each row
result = new_data["row_as_string"].apply(classify_job_listing)

# If you want, convert the results (list of dicts) into a DataFrame
classified_df = pd.DataFrame(result.tolist())

base_model_columns = list(JobClassification.model_fields.keys())

def get_value(cell, column_name=None):
    if isinstance(cell, tuple) and len(cell) == 2:
        val = cell[1]

        # Special case for IA_Heures_par_semaine
        if column_name == "IA_Heures_par_semaine" and hasattr(val, "heures"):
            return val.heures  # directly the int

        # Other enums / objects
        if hasattr(val, "value"):
            return val.value
        return str(val)
    elif hasattr(cell, "value"):
        return cell.value
    return str(cell)

classified_df = pd.DataFrame([
    [get_value(cell, col) for cell, col in zip(row, base_model_columns)]
    for row in classified_df.values
], columns=base_model_columns)

new_data = new_data.drop(columns=["row_as_string"])

# -------- FIN CHATGPT DATA ENRICHMENT ----------------------------------------------------------------------------------------------

# Merge with original sample
new_data = pd.concat([new_data.reset_index(drop=True), classified_df], axis=1)

# -------- DEBUT EMBEDING OPENAI LARGE ----------------------------------------------------------------------------------------------

from tqdm import tqdm
import time

# --- Function to build a natural-language sentence per row ---
#def build_sentence(row):
#    return (
#        f"Le titre de ce job est {row.get('titre', '')} et est à {row.get('Ville', '')} "
#        f"dans le département {row.get('Code Postal', '')} de la région {row.get('Region', '')}. "
#        f"Ce poste est {row.get('IA_Logé', '')} et le contrat est en {row.get('IA_Type_de_contrat', '')}. "
#        f"Le salaire est de {row.get('IA_Salaire', '')}. Particularité : {row.get('IA_Couple_accepté', '')} "
#        f"et {row.get('IA_Urgent', '')}. Le job est dans un {row.get('IA_Catégorie_établissement', '')} "
#        f"et dans le secteur de {row.get('IA_Catégorie_Job_1', '')}. "
#        f"Détails du job : {row.get('IA_Catégorie_Job_2', '')} {row.get('IA_Catégorie_Job_3', '')}. "
#        f"Il commence le {row.get('IA_Date_de_début', '')} et dure {row.get('IA_Durée', '')}. "
#        f"L'établissement est situé {row.get('IA_Type_environnement', '')} et est {row.get('IA_Chaine_Indépendant', '')}. "
#        f"C'est un contrat à {row.get('IA_Temps_de_travail', '')} et travail de {row.get('IA_Horaires_de_travail', '')} "
#        f"pendant {row.get('IA_Heures_par_semaine', '')} heures par semaine. "
#        f"Le poste est pour les {row.get('IA_Éxpérience', '')}."
#    )

# --- Create the combined natural-language field ---
#new_data["combined_text"] = new_data.apply(build_sentence, axis=1)

# --- Define batching function ---
#def embed_in_batches(texts, batch_size=1500):
#    """
#    Embed a list of texts in batches to avoid API rate limits and memory issues.
#    """
#    embeddings = []
#    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
#        batch = texts[i:i + batch_size]
#        try:
#            response = client_ai.embeddings.create(
#                model="text-embedding-3-small",
#                input=batch
#            )
#            batch_embeddings = [item.embedding for item in response.data]
#            embeddings.extend(batch_embeddings)
#        except Exception as e:
#            print(f"Error at batch {i}: {e}")
#            time.sleep(5)  # Wait a bit and retry
#            continue
#    return embeddings

# --- Generate embeddings ---
#new_data["OpenAIEmbeddedSmall"] = embed_in_batches(new_data["combined_text"].tolist(), batch_size=1500)
new_data["OpenAIEmbeddedSmall"] = ""

# --- Optional: Clean up ---
#new_data.drop(columns=["combined_text"], inplace=True)

# -------- FIN EMBEDING OPENAI LARGE ----------------------------------------------------------------------------------------------


# -------- DEBUT DATA VALIDATION EMPTY VALUES OPENAI ----------------------------------------------------------------------------------------------

# Select columns starting with "IA_"
ia_cols = [col for col in new_data.columns if col.startswith("IA_")]

# Replace "" with "Non spécifié" in those columns only
new_data[ia_cols] = new_data[ia_cols].replace("", "Non spécifié")

new_data["IA_Catégorie_Job_1"] = new_data["IA_Catégorie_Job_1"].replace("Support & Back-office","Non spécifié")

# -------- FIN DATA VALIDATION EMPTY VALUES OPENAI ----------------------------------------------------------------------------------------------

print(f"Post concat Check combined_data length {len(new_data)}")

# Debug: Print the number of rows to append
rows_to_append = new_data.shape[0]
print(f"Rows to append: {rows_to_append}")

# Handle NaN, infinity values before sending to Google Sheets
# Replace NaN values with 0 or another placeholder (you can customize this)
new_data = new_data.fillna(0)

# Replace infinite values with 0 or another placeholder
new_data.replace([float('inf'), float('-inf')], 0, inplace=True)

# Optional: Ensure all float types are valid (e.g., replace any invalid float with 0)
new_data = new_data.applymap(lambda x: 0 if isinstance(x, float) and (x == float('inf') or x == float('-inf') or x != x) else x)

# Optional: Ensuring no invalid values (like lists or dicts) in any column
def clean_value(value):
    if isinstance(value, (list, dict)):
        return str(value)  # Convert lists or dicts to string
    return value

new_data = new_data.applymap(clean_value)

#add column titre de annonce sans accents ni special characters
def remove_accents_and_special(text):
    # Normalize the text to separate characters from their accents.
    normalized = unicodedata.normalize('NFD', text)
    # Remove the combining diacritical marks.
    without_accents = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Replace special characters (-, ') with a space.
    cleaned = re.sub(r"[-']", " ", without_accents)
    # Remove other special characters (retain letters, digits, and whitespace).
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", cleaned)
    return cleaned

# Create the new column "Titre annonce sans accent" by applying the function on "intitule".
new_data["TitreAnnonceSansAccents"] = new_data["titre"].apply(
    lambda x: remove_accents_and_special(x) if isinstance(x, str) else x
)

print(f"Post concat Check combined_data length {len(new_data)}")


#---------UPLOAD TO BIGQUERY-------------------------------------------------------------------------------------------------------------

from google.cloud import bigquery
from google.oauth2 import service_account

# Load JSON from GitHub secret
key_json = json.loads(os.environ["BIGQUERY"])

# Create credentials from dict
credentials = service_account.Credentials.from_service_account_info(key_json)

# Initialize BigQuery client
client = bigquery.Client(
    credentials=credentials,
    project=key_json["project_id"]
)

table_id = "databasealfred.jobListings.hellowork"

# CONFIG WITHOUT PYARROW
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
)

# Convert DataFrame → list of dict rows (JSON compatible)
rows = new_data.to_dict(orient="records")

# Upload
job = client.load_table_from_json(
    rows,
    table_id,
    job_config=job_config
)

job.result()

print("✅ Data successfully loaded into BigQuery (JSON mode, no PyArrow needed)")
