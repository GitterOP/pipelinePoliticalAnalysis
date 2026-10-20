import sys
import os
import requests
import json
import time
import argparse
import pandas as pd
import re
import subprocess
import unicodedata
import difflib

# Intentar importar tiktoken para conteo preciso
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Configuración de LM Studio
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODELS_URL = "http://localhost:1234/api/v0/models"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"

def get_lm_studio_model_info(default_model):
    """Detecta el modelo cargado y su ventana de contexto."""
    try:
        response = requests.get(LM_STUDIO_MODELS_URL, timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                # Tomar el primero que esté cargado
                loaded = [m for m in data if m.get("state") == "loaded"]
                model = loaded[0] if loaded else data[0]
                model_id = model.get("id")
                max_ctx = model.get("loaded_context_length") or model.get("max_context_length", 32768)
                print(f"=> LM Studio: Detectado '{model_id}' con {max_ctx} tokens de contexto.")
                return model_id, int(max_ctx)
    except Exception:
        pass
    print(f"=> LM Studio: No se detectó modelo via API. Usando default {default_model}.")
    return default_model, 32768

def manage_model_loading(target_model: str, context_length: int = 16000):
    """ Gestiona la carga del modelo usando la CLI 'lms' de LM Studio. """
    print(f"🔧 Configurando modelo: {target_model} (ctx={context_length})...")
    try:
        print("   Descargando modelos previos...")
        subprocess.run(["lms", "unload", "--all"], check=False, capture_output=True)
        print(f"   Cargando {target_model}...")
        cmd = ["lms", "load", target_model, "--context-length", str(context_length), "--gpu", "max"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Modelo cargado: {target_model}")
            import time
            time.sleep(2)
        else:
            print(f"   ⚠️ Error cargando modelo via CLI: {result.stderr}")
    except Exception as e:
        print(f"   ⚠️ Error en gestión de modelos: {e}")


def merge_diarization(json_data):
    """
    FASE 1: Ingesta y Fusión Determinista.
    Agrupa segmentos contiguos del mismo SPEAKER.
    """
    segments = json_data.get("segments", [])
    if not segments:
        return pd.DataFrame()

    df_raw = pd.DataFrame(segments)
    
    # Lógica de fusión
    merged_data = []
    if not df_raw.empty:
        current_segment = {
            "Start": df_raw.iloc[0]["start"],
            "End": df_raw.iloc[0]["end"],
            "Speaker_Tag": df_raw.iloc[0].get("speaker", "UNKNOWN"),
            "Text": df_raw.iloc[0]["text"].strip()
        }
        
        for i in range(1, len(df_raw)):
            row = df_raw.iloc[i]
            speaker = row.get("speaker", "UNKNOWN")
            
            if speaker == current_segment["Speaker_Tag"]:
                # Misma persona hablando, extendemos tiempo y texto
                current_segment["End"] = row["end"]
                current_segment["Text"] += " " + row["text"].strip()
            else:
                # Cambio de hablante, guardamos el anterior e iniciamos nuevo
                merged_data.append(current_segment)
                current_segment = {
                    "Start": row["start"],
                    "End": row["end"],
                    "Speaker_Tag": speaker,
                    "Text": row["text"].strip()
                }
        merged_data.append(current_segment)
    
    return pd.DataFrame(merged_data)

def analyze_identities_with_llm(context_text, model_id, candidato_foco):
    """
    FASE 2: Inferencia Dinámica de Identidades (Llamada al LLM).
    Solicita un JSON para el mapeo.
    """
    system_prompt = f"""
[SYSTEM INSTRUCTION]
Eres un sistema experto en extracción de metadatos de transcripciones políticas.
Analiza la transcripción y deduce la identidad real de las etiquetas genéricas (SPEAKER_XX).
Sujeto de Interés: {candidato_foco}

[SOLICITUD]
Devuelve ÚNICAMENTE un objeto JSON. NO escribas análisis previo, ni pensamientos, ni explicaciones.
Usa EXACTAMENTE las etiquetas (ej. SPEAKER_02, SPEAKER_03) que aparezcan en el texto que se te envía. 
Estructura de ejemplo:
{{
  "contexto_principal": "...",
  "mapping": {{
    "ETIQUETA_1": "Nombre Real (Cargo)",
    "ETIQUETA_2": "Nombre Real (Cargo)"
  }}
}}

[CONSTRAINT]
- Respond ONLY with raw JSON.
- Identify at least {candidato_foco}.
- DEBES incluir TODAS las etiquetas de hablantes que encuentres en el texto.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcripción Ensamblada:\n\n{context_text}"}
    ]

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 64000 # Aumentado para evitar truncamiento
    }

    content = ""
    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=300)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Limpieza de tags <think>
            content = re.sub(r'<(think|thought)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)

            # Buscador de bloques JSON recursivo
            # Intentamos encontrar todos los bloques que empiecen con { y terminen con }
            matches = []
            
            # Buscamos todos los posibles cierres de llaves
            stack = 0
            start = -1
            for i, char in enumerate(content):
                if char == '{':
                    if stack == 0: start = i
                    stack += 1
                elif char == '}':
                    stack -= 1
                    if stack == 0 and start != -1:
                        matches.append(content[start:i+1])
            
            # Intentar parsear cada bloque encontrado, de atrás hacia adelante (suele ser el último)
            for block in reversed(matches):
                try:
                    data = json.loads(block)
                    if "mapping" in data:
                        return data
                except:
                    continue

            # Si falla, intentar limpiar y parsear el más grande
            if matches:
                largest = max(matches, key=len)
                try:
                    # Limpieza básica de carácteres no válidos
                    clean_str = re.sub(r'[^\x20-\x7E\s]', '', largest)
                    return json.loads(clean_str)
                except: pass

            print(f"Warning: No valid JSON block found in LLM response.")
    except Exception as e:
        print(f"Error parsing LLM JSON: {e}")
        print(f"Bruto (clean): {content[:200]}...")
    return None

def procesar_archivo(archivo, candidato, target_model):
    print(f"\n--- Procesando: {os.path.basename(archivo)} ---")
    print(f"Sujeto de Interés detectado: {candidato}")
    with open(archivo, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df_merged = merge_diarization(data)
    print(f"Segmentos originales: {len(data.get('segments', []))}")
    print(f"Segmentos fusionados: {len(df_merged)}")

    if df_merged.empty:
        print("Error: No se detectaron segmentos en el archivo.")
        return None

    # Preparar texto para el LLM (Primeros 60 bloques suelen cubrir presentaciones)
    text_for_llm = ""
    for _, row in df_merged.head(60).iterrows():
        text_for_llm += f"[{row['Speaker_Tag']}]: {row['Text']}\n"

    print(f"Enviando contexto al LLM para identificación de hablantes...")
    
    metadata = analyze_identities_with_llm(text_for_llm, target_model, candidato)
    
    if not metadata:
        print("Error: No se pudo obtener el mapeo del LLM. Usando etiquetas genéricas.")
        mapping = {}
        contexto_global = "Sin contexto detectado"
    else:
        mapping = metadata.get("mapping", {})
        contexto_global = metadata.get("contexto_principal", "Contexto detectado por LLM")
        print(f"Mapeo detectado: {mapping}")

    # Aplicar mapeo al DataFrame
    df_merged["Speaker_Name"] = df_merged["Speaker_Tag"].map(lambda x: mapping.get(x, x))
    df_merged["Contexto_Evento"] = contexto_global
    
    # Reordenar columnas para el CSV final
    df_final = df_merged[["Start", "End", "Speaker_Tag", "Speaker_Name", "Contexto_Evento", "Text"]]
    df_final.index.name = "ID_Intervencion"
    
    return df_final

def procesar_carpeta(input_folder, output_folder="output_procesado", links_excel="Discursos/Links.xlsx", target_model=DEFAULT_MODEL, context_length=16000):
    """
    Procesa todos los archivos JSON en `input_folder`, determinando el candidato
    desde `links_excel` utilizando el ID en el nombre del archivo (formato ID_CID_nombreF.json).
    Guarda los resultados como archivos CSV en `output_folder`.
    """
    manage_model_loading(target_model, context_length)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    try:
        df_links = pd.read_excel(links_excel)
    except Exception as e:
        print(f"Error al cargar el archivo de enlaces ({links_excel}): {e}")
        return
        
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            parts = filename.replace('.json', '').split('_')
            
            try:
                vid_id = int(parts[0])
            except ValueError:
                print(f"\nSaltando '{filename}': no se pudo extraer un ID numérico válido al inicio.")
                continue
                
            # Buscar el candidato usando el ID
            fila_candidato = df_links[df_links['ID'] == vid_id]
            if fila_candidato.empty:
                print(f"\nSaltando '{filename}': no se encontró el ID {vid_id} en la hoja de enlaces.")
                continue
                
            candidato = fila_candidato.iloc[0]['Candidato']
            
            archivo_path = os.path.join(input_folder, filename)
            df_final = procesar_archivo(archivo_path, candidato, target_model)
            
            if df_final is not None:
                output_csv = os.path.join(output_folder, filename.replace('.json', '.csv'))
                df_final.to_csv(output_csv, encoding='utf-8-sig') 
                print(f"--> Archivo guardado con éxito: {output_csv}")

def normalize_text(text: str) -> str:
    """Elimina acentos, convierte a minúsculas y elimina signos de puntuación (ej. guiones)."""
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8').lower()
    # Reemplazamos cualquier cosa que no sea texto/número por espacio
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Colapsamos espacios adicionales
    return re.sub(r'\s+', ' ', text).strip()

def is_similar_word(w1: str, w2: str, threshold: float = 0.75) -> bool:
    """Compara si dos palabras son similares permitiendo pequeños errores (ej. chau vs chao)"""
    if len(w1) <= 2 or len(w2) <= 2:
        return w1 == w2
    return difflib.SequenceMatcher(None, w1, w2).ratio() >= threshold

def normalizar_nombres_candidatos_csv(csv_path: str, candidato_foco: str):
    """
    Lee un CSV, busca los nombres únicos en Speaker_Name y añade
    una columna 'Es_Candidato' (1 si coincide con candidato_foco, 0 si no).
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            print(f"Error leyendo {csv_path}: {e}")
            return

    if "Speaker_Name" not in df.columns:
        return

    norm_candidato = normalize_text(candidato_foco)
    partes_candidato = norm_candidato.split()
    nombres_unicos = df["Speaker_Name"].dropna().unique()
    
    nombres_validos = set()
    for nombre in nombres_unicos:
        norm_speaker = normalize_text(nombre)
        partes_speaker = norm_speaker.split()
        
        # 1. Matching exacto o como substring directo
        if (norm_candidato in norm_speaker) or (len(norm_speaker) > 4 and norm_speaker in norm_candidato):
            nombres_validos.add(nombre)
            continue
            
        # 2. Matching inteligente por tokens (palabras) con tolerancia a errores
        coincidencias = 0
        for pc in partes_candidato:
            # Buscamos si la parte del nombre 'pc' se parece a alguna palabra del LLM
            if any(is_similar_word(pc, ps) for ps in partes_speaker):
                coincidencias += 1
                
        # Exigimos coincidir en al menos 2 palabras si el nombre buscado tiene >= 2
        umbral = 2 if len(partes_candidato) >= 2 else 1
        
        if coincidencias >= umbral:
            nombres_validos.add(nombre)

    # Añadimos la columna inicializada en 0
    df["Es_Candidato"] = 0
    
    if nombres_validos:
        # Asignamos 1 a las filas donde Speaker_Name esté en nombres_validos
        df.loc[df["Speaker_Name"].isin(nombres_validos), "Es_Candidato"] = 1
        print(f"    -> Candidato '{candidato_foco}' detectado bajo las etiquetas: {list(nombres_validos)}")
    else:
        print(f"    ⚠️ AVISO: No se detectó al candidato '{candidato_foco}' en el archivo {os.path.basename(csv_path)}. Se marcaron todas las filas con 0.")

    # Sobreescribimos el CSV con la nueva columna
    df.to_csv(csv_path, encoding='utf-8-sig', index=False)

def normalizar_candidatos_carpeta(output_folder: str = "output_procesado", links_excel: str = "Discursos/Links.xlsx"):
    """
    Paso intermedio que evalúa la presencia del candidato en la columna Speaker_Name
    y añade la columna indicador 'Es_Candidato' en todos los CSV procesados.
    """
    print(f"\nIniciando identificación y etiquetado de candidatos (Es_Candidato) en carpeta: {output_folder}")
    if not os.path.exists(output_folder):
        print(f"La carpeta {output_folder} no existe.")
        return
        
    try:
        df_links = pd.read_excel(links_excel)
    except Exception as e:
        print(f"Error al cargar el archivo de enlaces ({links_excel}): {e}")
        return
        
    for filename in os.listdir(output_folder):
        if filename.endswith(".csv"):
            parts = filename.replace('.csv', '').split('_')
            try:
                vid_id = int(parts[0])
            except ValueError:
                continue
                
            fila_candidato = df_links[df_links['ID'] == vid_id]
            if fila_candidato.empty:
                continue
                
            candidato = str(fila_candidato.iloc[0]['Candidato']).strip()
            
            archivo_path = os.path.join(output_folder, filename)
            normalizar_nombres_candidatos_csv(archivo_path, candidato)
    print("Normalización completada.")

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Procesamiento de Transcripciones Lote (Fase 1-3)")
    parser.add_argument("--input_folder", required=True, help="Carpeta con los archivos .json de WhisperX")
    parser.add_argument("--output_folder", default="output_estructurada", help="Carpeta de salida para los CSV")
    parser.add_argument("--links_excel", default="Discursos/Links.xlsx", help="Ruta al archivo Excel de metadatos")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nombre del modelo para LM Studio")
    parser.add_argument("--ctx", type=int, default=16000, help="Ventana de contexto")
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: La carpeta de entrada '{args.input_folder}' no existe.")
        sys.exit(1)

    procesar_carpeta(args.input_folder, args.output_folder, args.links_excel, args.model, args.ctx)
    # Paso extra: normalizar los nombres en los csv recién generados
    normalizar_candidatos_carpeta(args.output_folder, args.links_excel)
    print("\nProcesamiento por lotes completado.")

if __name__ == "__main__":
    main()
