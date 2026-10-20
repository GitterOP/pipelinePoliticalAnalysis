import json
import re
import requests
import difflib
from typing import List, Optional, Any, Tuple
from pydantic import BaseModel, ValidationError, field_validator

VALID_TOPICS = {
    "SEGURIDAD_CIUDADANA", "ECONOMIA_Y_EMPLEO", "JUSTICIA_Y_ANTICORRUPCION", "SALUD_Y_BIENESTAR", 
    "EDUCACION_Y_CIENCIA", "INFRAESTRUCTURA_Y_TERRITORIO", "MINERIA_Y_AMBIENTE", "REFORMA_ESTADO"
}

VALID_TONES = {
    "TECNICO_INFORMATIVO", "INSTITUCIONAL_FORMAL", "PROPOSITIVO_ASPIRACIONAL", 
    "CONFRONTACIONAL_CRITICO", "REIVINDICATIVO_POPULISTA", "DISCIPLINARIO_MANO_DURA"
}

class Content(BaseModel):
    raw_text: str
    normalized_text: str

class Tags(BaseModel):
    topics: List[str]
    keywords: List[str]
    sentiment_tone: str

    @field_validator('topics', mode='before')
    @classmethod
    def clean_topics(cls, v: Any):
        if not isinstance(v, list): return v
        cleaned_topics = []
        for topic in v:
            t = str(topic).strip().upper()
            if t in VALID_TOPICS: cleaned_topics.append(t)
            else: cleaned_topics.append(f"otros: {t}")
        return cleaned_topics

    @field_validator('sentiment_tone', mode='before')
    @classmethod
    def clean_tone(cls, v: Any):
        t = str(v).strip().upper()
        if t in VALID_TONES: return t
        return f"otros: {t}"

class EntrevistaPropuesta(BaseModel):
    chunk_id: str = ""
    content: Content
    tags: Tags

class RespuestaLLM(BaseModel):
    es_propuesta_valida: bool
    analisis_previo_exhaustivo: str
    propuestas_extraidas: List[EntrevistaPropuesta] = []

import subprocess

def manage_model_loading(target_model: str, context_length: int = 16000):
    print(f"🔧 Configurando modelo: {target_model} (ctx={context_length})...")
    try:
        subprocess.run(["lms", "unload", "--all"], check=False, capture_output=True)
        cmd = ["lms", "load", target_model, "--context-length", str(context_length), "--gpu", "max"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode == 0:
            print(f"   ✅ Modelo cargado: {target_model}")
            import time
            time.sleep(2)
        else:
            print(f"   ⚠️ Error cargando modelo via CLI: {result.stderr}")
    except Exception as e:
        print(f"   ⚠️ Error en gestión de modelos: {e}")

class AnalizadorEntrevista:
    def __init__(self, api_url: str = "http://127.0.0.1:1234/v1", modelo: str = "gpt-oss-20b", debug_mode: bool = False):
        self.api_url = f"{api_url}/chat/completions" if not api_url.endswith("/chat/completions") else api_url
        self.api_base = api_url.replace("/chat/completions", "")
        self.modelo = modelo
        self.modelo_exacto = modelo
        self.debug_mode = debug_mode

    def procesar_intervencion(self, texto_bloque: str, contexto_fragmento: str, base_chunk_id: str) -> Tuple[List[dict], Optional[dict]]:
        system_prompt = """Eres un analista de datos experto en PLN. Tu objetivo es transformar lenguaje oral en unidades lógicas de información atómica.

**REGLAS ESTRICTAS DE EXTRACCIÓN (FILTRO DE RUIDO):**
1. **Definición de Propuesta:** Una propuesta DEBE ser una acción explícita del Estado, una inversión, una reforma institucional o la creación de leyes/programas. 
2. **Descarte Absoluto:** Si el texto es una anécdota, un saludo ("Muchas gracias"), una queja del pasado, una justificación personal ("Porque trabajo"), o un ataque a un rival, ES RUIDO. Debes establecer "es_propuesta_valida": false y dejar "propuestas_extraidas": [].
3. **Aislamiento Atómico:** Si hay 2 ideas válidas distintas, genera 2 objetos JSON separados dentro del array. NO cortes una misma idea por la mitad.
4. **Fidelidad:** El 'raw_text' DEBE ser una cita literal exacta del texto fuente.
5. **Normalización:** El 'normalized_text' debe redactarse en tercera persona, en tono formal y programático.
6. **Taxonomía:** Clasifica la propuesta obligatoriamente usando solo estos valores:
Tópicos Válidos: SEGURIDAD_CIUDADANA, ECONOMIA_Y_EMPLEO, JUSTICIA_Y_ANTICORRUPCION, SALUD_Y_BIENESTAR, EDUCACION_Y_CIENCIA, INFRAESTRUCTURA_Y_TERRITORIO, MINERIA_Y_AMBIENTE, REFORMA_ESTADO.
Tonos Válidos: TECNICO_INFORMATIVO, INSTITUCIONAL_FORMAL, PROPOSITIVO_ASPIRACIONAL, CONFRONTACIONAL_CRITICO, REIVINDICATIVO_POPULISTA, DISCIPLINARIO_MANO_DURA.

Retorna ÚNICAMENTE un JSON con esta estructura exacta, sin alterar llaves:
{
    "es_propuesta_valida": true,
    "analisis_previo_exhaustivo": "Paso 1: Evalúo si hay propuestas. Encuentro X e Y. Son acciones del estado. Paso 2: Las separo.",
    "propuestas_extraidas": [
        {
            "content": {
                "raw_text": "Cita textual del fragmento",
                "normalized_text": "Propuesta en tercera persona y técnica"
            },
            "tags": {
                "topics": ["TÓPICO_VÁLIDO_1"],
                "keywords": ["palabra1", "palabra2"],
                "sentiment_tone": "TONO_VÁLIDO"
            }
        }
    ]
}"""

        user_prompt = f"El texto a analizar es el siguiente:\n\n{texto_bloque}"
        if contexto_fragmento and contexto_fragmento.strip():
            user_prompt = f"Contexto y resolución explícita de referencias: {contexto_fragmento.strip()}\n\n" + user_prompt
            
        # Resolución dinámica de ID exacto del modelo cargado
        if self.modelo_exacto == self.modelo:
            try:
                res = requests.get(f"{self.api_base}/models", timeout=5)
                if res.status_code == 200:
                    for m in res.json().get("data", []):
                        uid = m.get("id", "")
                        if self.modelo.lower() in uid.lower():
                            self.modelo_exacto = uid
                            break
            except Exception:
                pass

        payload = {
            "model": self.modelo_exacto,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "stream": False
        }

        try:
            if self.debug_mode:
                print(f"[{base_chunk_id}] Analizando bloque atómico de {len(texto_bloque.split())} palabras...")
            response = requests.post(self.api_url, json=payload, timeout=300)
            
            if response.status_code != 200:
                print(f"[{base_chunk_id}] HTTP {response.status_code}")
                return [], {"chunk_id": base_chunk_id, "motivo": f"HTTP Error {response.status_code}", "analisis": ""}
                
            res_content = response.json()['choices'][0]['message']['content']
            clean_json = re.sub(r'<think>.*?</think>', '', res_content, flags=re.DOTALL).strip()
            
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_json, re.DOTALL)
            if json_match:
                clean_json = json_match.group(1).strip()
            else:
                start_dict = clean_json.find('{')
                end_dict = clean_json.rfind('}')
                if start_dict != -1 and end_dict != -1:
                    clean_json = clean_json[start_dict:end_dict+1]
                    
            data = json.loads(clean_json)
            
            # Validación mediante Pydantic
            respuesta_validada = RespuestaLLM(**data)
            
        except ValidationError as ve:
            print(f"[{base_chunk_id}] Error validando esquema: {ve}")
            return [], {"chunk_id": base_chunk_id, "motivo": "Excepción de validación (Pydantic)", "analisis": ""}
        except Exception as e:
            print(f"[{base_chunk_id}] Error inesperado: {e}")
            return [], {"chunk_id": base_chunk_id, "motivo": str(e), "analisis": ""}
            
        # Condición de Ruido Político
        if not respuesta_validada.es_propuesta_valida or not respuesta_validada.propuestas_extraidas:
            return [], {
                "chunk_id": base_chunk_id, 
                "motivo": "Ruido político descartado", 
                "analisis": respuesta_validada.analisis_previo_exhaustivo
            }
            
        propuestas_validadas = []
        for i, prop in enumerate(respuesta_validada.propuestas_extraidas):
            # 1. Asignar ID Único
            prop.chunk_id = f"{base_chunk_id}_prop_{i+1}"
            
            # 2. Capa de Verificación Heurística (Fuzzy Matching) implementada como dictada
            texto_llm = prop.content.raw_text
            texto_fuente = texto_bloque # El texto original ingresado al método
            
            # Si la extracción no es una subcadena exacta, forzamos la alineación
            if texto_llm not in texto_fuente:
                matcher = difflib.SequenceMatcher(None, texto_llm, texto_fuente)
                match = matcher.find_longest_match(0, len(texto_llm), 0, len(texto_fuente))
                
                # Si encuentra un nivel de similitud aceptable, sobrescribe con el texto real de la fuente
                if match.size > 10: 
                    # Expandimos ligeramente para capturar la palabra completa si se cortó
                    inicio = max(0, match.b - 15)
                    fin = min(len(texto_fuente), match.b + match.size + 15)
                    texto_real_recuperado = texto_fuente[inicio:fin].strip()
                    prop.content.raw_text = texto_real_recuperado
                    
                    if self.debug_mode:
                        print(f"   [Fidelidad] Re-alineado:\n   Original LLM: '{texto_llm[:50]}...'\n   Sobrescrito:  '{texto_real_recuperado[:50]}...'")

            propuestas_validadas.append(prop.model_dump())
            
        return propuestas_validadas, None

import os
import pandas as pd

def procesar_carpeta_atomico(
    input_folder: str,
    output_validados_folder: str = "output_atomico/validados",
    output_rechazados_folder: str = "output_atomico/rechazados",
    links_excel: str = "Discursos/Links.xlsx",
    modelo: str = "gpt-oss-20b",
    api_url: str = "http://127.0.0.1:1234/v1",
    debug_mode: bool = False,
    context_length: int = 16000
):
    """
    Procesa batch de todos los JSON generados por el fragmentador en `input_folder`.
    Extrae propuestas usando el motor atómico y mapea los fragmentos de la intervención.
    """
    if not os.path.exists(output_validados_folder):
        os.makedirs(output_validados_folder)
    if not os.path.exists(output_rechazados_folder):
        os.makedirs(output_rechazados_folder)
        
    try:
        df_links = pd.read_excel(links_excel)
    except Exception as e:
        print(f"Error al cargar el archivo de enlaces ({links_excel}): {e}")
        return
        
    # 1. Cargar el modelo en bloque
    manage_model_loading(target_model=modelo, context_length=context_length)
    # 2. Instanciar analizador
    analizador = AnalizadorEntrevista(api_url=api_url, modelo=modelo, debug_mode=debug_mode)

    print(f"\n[PIPELINE - FASE 3] Extracción Atómica en carpeta: {input_folder}")
    
    for filename in os.listdir(input_folder):
        # Asegurarnos de agarrar los archivos json correctos
        if filename.endswith(".json") and "rechazad" not in filename:
            parts = filename.replace('.json', '').replace('_fragmentos', '').split('_')
            try:
                vid_id = int(parts[0])
                fila_candidato = df_links[df_links['ID'] == vid_id]
                candidato_foco = str(fila_candidato.iloc[0]['Candidato']).strip() if not fila_candidato.empty else "CAND"
            except ValueError:
                candidato_foco = "CAND"
                
            # Crear el acrónimo para el ID (ej: RLA de "Rafael Lopez Aliaga")
            candidato_abbr = "".join([palabra[0] for palabra in candidato_foco.split() if palabra]).upper()
            if not candidato_abbr: candidato_abbr = "CAND"
            
            archivo_path = os.path.join(input_folder, filename)
            base_name = filename.replace('_fragmentos.json', '').replace('.json', '')
            output_json_path = os.path.join(output_validados_folder, f"{base_name}_atomico.json")
            output_rechazos_path = os.path.join(output_rechazados_folder, f"{base_name}_deriva.json")
            
            print(f"\n>>> INICIANDO EXTRACCIÓN ATÓMICA: {filename}")
            print(f">>> Candidato Identificado: {candidato_foco} (Acrónimo: {candidato_abbr})")
            
            try:
                with open(archivo_path, "r", encoding="utf-8") as f:
                    intervenciones = json.load(f)
            except Exception as e:
                print(f"Error leyendo {archivo_path}: {e}")
                continue
                
            salida_conjunta = []
            rechazos_totales = []
            
            for intervencion in intervenciones:
                id_intervencion = intervencion.get("id_intervencion", 0)
                t_start = intervencion.get("timestamp_start", "00:00:00")
                t_end = intervencion.get("timestamp_end", "00:00:00")
                fragmentos = intervencion.get("fragmentos", [])
                
                for frag in fragmentos:
                    # Formando el chunk base para trazabilidad
                    base_chunk_id = f"ENT_{candidato_abbr}_{id_intervencion}_{frag.get('id', 'frag')}"
                    frag_text = frag.get('texto', '')
                    frag_context = frag.get('contexto', '')
                    
                    if not frag_text.strip(): continue
                    
                    # Llamada atómica
                    resultados, rechazo = analizador.procesar_intervencion(frag_text, frag_context, base_chunk_id)
                    
                    if resultados:
                        for r in resultados:
                            # Inyección de metadatos trazables
                            r["meta_intervencion"] = {
                                "id_intervencion": id_intervencion,
                                "timestamp_start": t_start,
                                "timestamp_end": t_end,
                                "fragmento_origen": frag.get('id', 'frag'),
                                "texto_original_fragmento": frag_text
                            }
                            salida_conjunta.append(r)
                    else:
                        if rechazo:
                            rechazos_totales.append(rechazo)
                            
            if output_json_path and salida_conjunta:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(salida_conjunta, f, indent=4, ensure_ascii=False)
            
            if output_rechazos_path and rechazos_totales:
                with open(output_rechazos_path, "w", encoding="utf-8") as f:
                    json.dump(rechazos_totales, f, indent=4, ensure_ascii=False)
                    
            print(f">>> Archivo completado: {len(salida_conjunta)} promesas extraídas, {len(rechazos_totales)} ruidos/anécdotas filtradas.")
            
    print("\n✅ Procesamiento Atómico por lote Finalizado Exitosamente.")

def procesar_json_fragmentos_v2(
    input_json_path: str = "fragmentos_intermedios_v2.json",
    candidato_foco_abbr: str = "CAND",
    output_json_path: Optional[str] = "entrevistas_procesadas_v2_final.json",
    output_rechazos_path: Optional[str] = "entrevistas_rechazadas_v2.json",
    modelo: str = "gpt-oss-20b",
    api_url: str = "http://127.0.0.1:1234/v1",
    debug_mode: bool = False,
):
    """ Función orquestadora para procesamiento batch del JSON producido por FragmentadorOraciones """
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            intervenciones = json.load(f)
    except Exception as e:
        print(f"Error leyendo {input_json_path}: {e}")
        return []
    
    manage_model_loading(target_model=modelo, context_length=16000)
    analizador = AnalizadorEntrevista(api_url=api_url, modelo=modelo, debug_mode=debug_mode)
    
    salida_conjunta = []
    rechazos_totales = []
    
    for intervencion in intervenciones:
        id_intervencion = intervencion.get("id_intervencion", 0)
        t_start = intervencion.get("timestamp_start", "00:00:00")
        t_end = intervencion.get("timestamp_end", "00:00:00")
        fragmentos = intervencion.get("fragmentos", [])
        
        print(f"\n==========================================================================================")
        print(f"ANALIZANDO INTERVENCIÓN {id_intervencion} [{t_start} - {t_end}] ({len(fragmentos)} bloques lógicos)")
        print(f"==========================================================================================")
        
        for idx, frag in enumerate(fragmentos):
            base_chunk_id = f"ENT_{candidato_foco_abbr}_{id_intervencion}_{frag['id']}"
            frag_text = frag['texto']
            frag_context = frag.get('contexto', '')
            
            resultados, rechazo = analizador.procesar_intervencion(frag_text, frag_context, base_chunk_id)
            
            if resultados:
                print(f"   → Extraídas {len(resultados)} proposiciones lógicas.")
                for r in resultados:
                    r["meta_intervencion"] = {
                        "id_intervencion": id_intervencion,
                        "timestamp_start": t_start,
                        "timestamp_end": t_end,
                        "fragmento_origen": frag['id'],
                        "texto_original_fragmento": frag_text
                    }
                    salida_conjunta.append(r)
            else:
                motivo = rechazo.get("motivo", "") if rechazo else ""
                print(f"   → Descartado: {motivo}")
                if rechazo:
                    rechazos_totales.append(rechazo)
                    
    if output_json_path and salida_conjunta:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(salida_conjunta, f, indent=4, ensure_ascii=False)
        print(f"\nGuardado JSON de propuestas atómicas validadas en {output_json_path}.")
    
    if output_rechazos_path and rechazos_totales:
        with open(output_rechazos_path, "w", encoding="utf-8") as f:
            json.dump(rechazos_totales, f, indent=4, ensure_ascii=False)
        print(f"Guardado backlog estricto de rechazos y ruido político en {output_rechazos_path}.")
            
    return salida_conjunta

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    procesar_json_fragmentos_v2(
        input_json_path="fragmentos_intermedios_v2.json",
        candidato_foco_abbr="RLA",
        output_json_path="entrevistas_procesadas_v2_final.json",
        output_rechazos_path="entrevistas_rechazadas_v2.json",
        modelo="gpt-oss-20b",
        debug_mode=True
    )
