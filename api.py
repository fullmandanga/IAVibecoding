import spacy
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Cargar el modelo de lenguaje en español
nlp = spacy.load("es_core_news_md")

def obtener_palabras_clave(texto: str) -> List[str]:
    """Extrae las palabras clave de un texto utilizando SpaCy."""
    doc = nlp(texto)
    palabras_clave = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']]
    return palabras_clave

def generar_consulta_busqueda(palabras_clave: List[str]) -> str:
    """Genera una consulta de búsqueda a partir de las palabras clave."""
    return " ".join(palabras_clave)

def realizar_busqueda(consulta: str, api_key: str = None, engine_id: str = None) -> List[str]:
    """Realiza una búsqueda en Google y devuelve los resultados."""
    try:
        if api_key and engine_id:
            from googleapiclient.discovery import build
            service = build("customsearch", "v1", developerKey=api_key)
            results = service.cse().list(q=consulta, cx=engine_id, num=3).execute()
            return [item["link"] for item in results.get("items", [])]
        else:
            # Búsqueda web simple (sin API)
            url = f"https://www.google.com/search?q={consulta}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.find_all("a", href=True)
            links = [link["href"] for link in search_results if link["href"].startswith("http") and "google" not in link["href"]]
            return links[:3]  # Return the first 3 results

    except requests.exceptions.RequestException as e:
        print(f"Error en la búsqueda: {e}")
        return []
    except Exception as e:
        print(f"Error al procesar los resultados: {e}")
        return []

def obtener_respuestas(pregunta: str, api_key: str = None, engine_id: str = None) -> List[str]:
    """Procesa la pregunta, realiza la búsqueda y devuelve las respuestas."""
    palabras_clave = obtener_palabras_clave(pregunta)
    consulta = generar_consulta_busqueda(palabras_clave)
    resultados = realizar_busqueda(consulta, api_key, engine_id)
    return resultados

app = FastAPI()

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/api/search")
async def search(pregunta: str, api_key: str = None, engine_id: str = None):
    """Endpoint para realizar la búsqueda."""
    if not pregunta:
        raise HTTPException(status_code=400, detail="La pregunta es requerida")

    try:
        respuestas = obtener_respuestas(pregunta, api_key, engine_id)
        return {"respuestas": respuestas}
    except Exception as e:
        print(f"Error en la API: {e}")
    raise HTTPException(status_code=500, detail=str(e))

from agi.agi_model import analyze_code

@app.post("/api/analyze_code")
async def analyze_code_endpoint(code_snippet: str, analysis_type: str):
    """Endpoint to analyze code."""
    if not code_snippet or not analysis_type:
        raise HTTPException(status_code=400, detail="code_snippet and analysis_type are required")

    try:
        analysis_result = analyze_code(code_snippet, analysis_type)
        return {"analysis_type": analysis_type, "analysis_result": analysis_result}
    except Exception as e:
        print(f"Error in /api/analyze_code: {e}")
        raise HTTPException(status_code=500, detail=str(e))
