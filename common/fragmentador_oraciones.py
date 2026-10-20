import os
import json
import re
import nltk
import argparse
import requests
import pandas as pd
import subprocess
import unicodedata
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    """Elimina acentos y convierte a minúsculas para comparaciones seguras."""
    if not isinstance(text, str): return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()

# Asegurar que el tokenizador de puntuación esté descargado
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

def es_intervencion_valida(text: str, speaker_name: str) -> bool:
    """Filtra a los entrevistadores, moderadores y retórica de menos de 10 palabras."""
    if not isinstance(speaker_name, str) or not isinstance(text, str): return False
    nombre_lower = speaker_name.lower()
    if "moderador" in nombre_lower or "panelista" in nombre_lower or "entrevistador" in nombre_lower: return False
    if len(text.split()) < 10: return False
    return True

def formato_tiempo(segundos: float) -> str:
    """Convierte segundos a formato HH:MM:SS"""
    try:
        m, s = divmod(int(float(segundos)), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except (ValueError, TypeError):
        return "00:00:00"

class FragmentadorOraciones:
    def __init__(self, api_url: str, modelo: str, debug_mode: bool = False):
        self.api_url = f"{api_url}/chat/completions" if not api_url.endswith("/chat/completions") else api_url
        self.modelo = modelo
        self.debug_mode = debug_mode
        self.tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')

        self.SYSTEM_PROMPT = """Eres un motor analítico experto en NLP y discurso político. Tu única tarea es identificar el límite semántico entre ideas en una lista numerada de oraciones y resolver sujetos tácitos.

**REGLAS ESTRICTAS DE OPERACIÓN:**
1. **Cero Manipulación de Texto:** Tienes estrictamente prohibido reescribir, resumir o extraer el texto de las oraciones. Tu salida principal será un NÚMERO ENTERO (el índice de la oración).
2. **Detección del Límite (Granularidad):** Analiza la lista de oraciones secuencialmente. Identifica en qué número de oración el candidato termina una propuesta o idea específica ANTES de saltar a otra distinta (ej. si la oración 1 y 2 hablan de hospitales, y la 3 habla de carreteras, el límite es la oración 2).
3. **Idea Continua:** Si todas las oraciones del bloque proporcionado son parte de la MISMA idea o propuesta y no hay un cambio de tema, establece "corte_encontrado": false.
4. **Resolución de Contexto OBLIGATORIA (Correferencia Autónoma):** El campo "nota_contexto" es ESTRICTAMENTE OBLIGATORIO para todos los fragmentos. Debes usarlo para anclar de qué o quién se está hablando. NUNCA DEBE ESTAR VACÍO.
   - REGLA CRÍTICA EXTREMA: El contexto será insertado literalmente para que un humano lo lea SIN tener las oraciones anteriores. JAMÁS uses metatexto o frases como "mencionado en la oración 2". Di exactamente el nombre o la entidad: e.g. "[El candidato se refiere a la concesión de Rutas de Lima]".
   - REGLA UNIVERSAL: Aún si el texto parece completamente claro, DEBES generar una nota de contexto que sitúe la oración formalmente (ej. "[El candidato argumenta sobre su plan de seguridad militar]"). Nunca devuelvas un string vacío.

**Formato de Salida Requerido:**
Retorna ÚNICAMENTE un objeto JSON válido con la siguiente estructura:
{
    "corte_encontrado": true,
    "indice_oracion_final": 2,
    "razonamiento": "Las oraciones 1 y 2 proponen reformas judiciales. La oración 3 introduce un tema de seguridad militar.",
    "nota_contexto": "[El candidato se refiere al Ministerio Público]"
}"""

    def _preparar_bloque_prompt(self, oraciones: List[str], inicio: int, tamaño_ventana: int) -> str:
        """Formatea un subconjunto de oraciones en una lista numerada para el LLM."""
        bloque = oraciones[inicio : inicio + tamaño_ventana]
        texto_numerado = ""
        for i, oracion in enumerate(bloque):
            # Usamos índice base 1 para mayor claridad del LLM
            texto_numerado += f"[{i + 1}] {oracion}\n"
        return texto_numerado

    def _llamar_llm(self, bloque_oraciones_numeradas: str) -> dict:
        """Llama a la API local de modelo LLM para preguntar el corte en base a índices."""
        user_prompt = f"Analiza el siguiente bloque de oraciones y determina dónde termina la primera idea:\n\n{bloque_oraciones_numeradas}"
        
        # Resolución dinámica del ID real cargado en LM Studio
        if not hasattr(self, 'modelo_exacto'):
            self.modelo_exacto = self.modelo
            try:
                base_url = self.api_url.replace("/chat/completions", "")
                res = requests.get(f"{base_url}/models", timeout=5)
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
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                     content = json_match.group(1).strip()
                else:
                     start = content.find('{')
                     end = content.rfind('}')
                     if start != -1 and end != -1:
                         content = content[start:end+1]
                         
                try:
                    data = json.loads(content)
                    return data
                except json.JSONDecodeError:
                    if self.debug_mode:
                        print(f"[!] FragmentadorOraciones: Error decodificando JSON del LLM:\n{content}")
                    return {"corte_encontrado": False, "indice_oracion_final": 0, "nota_contexto": ""}
            else:
                if self.debug_mode:
                    print(f"[!] FragmentadorOraciones: Error HTTP {response.status_code} - {response.text[:200]}")
        except Exception as e:
            if self.debug_mode:
                print(f"[!] FragmentadorOraciones: Error llamando al LLM: {e}")
            
        return {"corte_encontrado": False, "indice_oracion_final": 0, "nota_contexto": ""}

    def procesar(self, texto_completo: str, tamaño_ventana_base: int = 5, max_ventana: int = 12) -> List[Dict[str, str]]:
        """
        Divide el texto mediante segmentación lógica de tensores de oraciones.
        """
        # 1. Hard split: Dividir todo el texto en oraciones exactas
        oraciones_crudas = self.tokenizer.tokenize(texto_completo)
        # Limpieza de espacios residuales
        oraciones = [o.strip() for o in oraciones_crudas if o.strip()] 
        
        fragmentos_finales = []
        indice_actual = 0
        total_oraciones = len(oraciones)

        while indice_actual < total_oraciones:
            tamaño_ventana = tamaño_ventana_base
            limite_resuelto = False
            
            while not limite_resuelto:
                # Evitar desbordamiento del arreglo
                if indice_actual + tamaño_ventana > total_oraciones:
                    tamaño_ventana = total_oraciones - indice_actual

                bloque_numerado = self._preparar_bloque_prompt(oraciones, indice_actual, tamaño_ventana)
                
                # Salvaguarda: Si llegamos al final del texto o al límite máximo de la ventana
                if tamaño_ventana >= max_ventana or (indice_actual + tamaño_ventana == total_oraciones):
                    indice_corte = tamaño_ventana # Forzamos el corte tomando todo el bloque
                    limite_resuelto = True
                    nota_contexto = ""
                else:
                    if self.debug_mode:
                        print(f"Enviando oraciones {indice_actual+1} a {indice_actual+tamaño_ventana} al LLM...")
                    
                    # Llamada a su método _llamar_llm
                    respuesta_llm = self._llamar_llm(bloque_numerado) 
                    
                    if respuesta_llm.get("corte_encontrado", False):
                        # Convertimos el índice base 1 del LLM a base 0 para Python
                        indice_llm = respuesta_llm.get("indice_oracion_final", tamaño_ventana)
                        # Validar que el LLM no alucine un índice fuera de rango
                        indice_corte = min(max(1, int(indice_llm)), tamaño_ventana)
                        raw_nota = respuesta_llm.get("nota_contexto", "")
                        nota_contexto = str(raw_nota).strip() if raw_nota else ""
                        limite_resuelto = True
                    else:
                        # La idea continúa. Aumentamos la ventana (lookahead buffer)
                        tamaño_ventana += 3 
                        if self.debug_mode:
                            print(f"Idea no concluye. Expandiendo ventana a {tamaño_ventana} oraciones.")

            # Ensamblaje Hardcodeado
            oraciones_fragmento = oraciones[indice_actual : indice_actual + indice_corte]
            texto_limpio = " ".join(oraciones_fragmento)
            
            fragmentos_finales.append({
                "id": f"frag_{len(fragmentos_finales) + 1}",
                "texto": texto_limpio,
                "contexto": nota_contexto
            })

            # Avanzamos el puntero estrictamente a la siguiente oración no procesada
            indice_actual += indice_corte

        return fragmentos_finales

def fragmentar_csv_entrevistas_oraciones(
    csv_path: str,
    candidato_foco_nombre: str,
    output_json_path: str = "fragmentos_intermedios_v2.json",
    modelo: str = "gpt-oss-20b",
    api_url: str = "http://127.0.0.1:1234/v1",
    debug_mode: bool = False,
    context_length: int = 16000,
    load_model: bool = True
) -> List[Dict[str, Any]]:
    """Procesa un archivo CSV y genera los fragmentos por intervención utilizando FragmentadorOraciones."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        # Fallback a utf-8 estandar si falla utf-8-sig
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e2:
            print(f"Error leyendo {csv_path}: {e2}")
            return []
        print(f"Error leyendo {csv_path}: {e}")
        return []
        
    if load_model:
        manage_model_loading(target_model=modelo, context_length=context_length)
    
    fragmentador = FragmentadorOraciones(api_url=api_url, modelo=modelo, debug_mode=debug_mode)
    
    intervenciones_procesadas = []
    
    for index, row in df.iterrows():
        text = str(row.get("Text", ""))
        speaker_name = str(row.get("Speaker_Name", ""))
        
        # Usar la nueva columna de Es_Candidato para decidir procesamiento
        es_candidato = row.get("Es_Candidato", 0)
        
        if es_candidato != 1:
            continue
            
        if not es_intervencion_valida(text, speaker_name):
            continue
            
        t_start = formato_tiempo(row.get("Start", 0))
        t_end = formato_tiempo(row.get("End", 0))
        id_intervencion = row.get("ID_Intervencion", index)
        n_palabras = len(text.split())
        
        print(f"\\n==========================================================================================")
        print(f"FRAGMENTANDO INTERVENCIÓN {id_intervencion} de {candidato_foco_nombre} [{t_start} - {t_end}] (~{n_palabras} palabras)")
        print(f"==========================================================================================")
        
        fragmentos = fragmentador.procesar(text, tamaño_ventana_base=5, max_ventana=12)
        
        intervencion_data = {
            "id_intervencion": id_intervencion,
            "speaker": speaker_name,
            "timestamp_start": t_start,
            "timestamp_end": t_end,
            "texto_completo_intervencion": text,
            "fragmentos": fragmentos
        }
        intervenciones_procesadas.append(intervencion_data)
        
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(intervenciones_procesadas, f, indent=4, ensure_ascii=False)
        
    print(f"\\nProceso finalizado. Guardado en {output_json_path}.")
    return intervenciones_procesadas

def procesar_carpeta_fragmentos(
    input_folder: str,
    output_folder: str = "output_fragmentado",
    links_excel: str = "Discursos/Links.xlsx",
    modelo: str = "gpt-oss-20b",
    api_url: str = "http://127.0.0.1:1234/v1",
    debug_mode: bool = False,
    context_length: int = 16000
):
    """
    Procesa todos los archivos CSV de entrevistas estructuradas en `input_folder`.
    Determina automáticamente el nombre del candidato mediante `links_excel` 
    (usando el ID numérico del nombre del archivo).
    Luego guarda un JSON con los fragmentos por cada CSV en `output_folder`.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    try:
        df_links = pd.read_excel(links_excel)
    except Exception as e:
        print(f"Error al cargar el archivo de enlaces ({links_excel}): {e}")
        return
        
    # Cargar el modelo UNA VEZ al principio del procesamiento por lotes
    manage_model_loading(target_model=modelo, context_length=context_length)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            parts = filename.replace('.csv', '').split('_')
            
            try:
                vid_id = int(parts[0])
            except ValueError:
                print(f"\nSaltando '{filename}': no se pudo extraer un ID numérico válido al inicio.")
                continue
                
            fila_candidato = df_links[df_links['ID'] == vid_id]
            if fila_candidato.empty:
                print(f"\nSaltando '{filename}': no se encontró el ID {vid_id} en {links_excel}.")
                continue
                
            candidato = str(fila_candidato.iloc[0]['Candidato']).strip()
            
            archivo_path = os.path.join(input_folder, filename)
            output_json_path = os.path.join(output_folder, filename.replace('.csv', '_fragmentos.json'))
            
            print(f"\n>>> INICIANDO ARCHIVO: {filename}")
            print(f">>> Sujeto de Interés Encontrado: {candidato}")
            
            # Ejecutamos con load_model=False para no reiniciar el modelo en LM Studio por cada archivo
            fragmentar_csv_entrevistas_oraciones(
                csv_path=archivo_path,
                candidato_foco_nombre=candidato,
                output_json_path=output_json_path,
                modelo=modelo,
                api_url=api_url,
                debug_mode=debug_mode,
                context_length=context_length,
                load_model=False
            )

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Fragmentación por lotes")
    parser.add_argument("--input_folder", required=True, help="Carpeta con archivos .csv de entrevistas estructuradas")
    parser.add_argument("--output_folder", default="output_fragmentado", help="Carpeta salida para los JSONs de fragmentos")
    parser.add_argument("--links_excel", default="Discursos/Links.xlsx", help="Ruta al archivo Excel de metadatos")
    parser.add_argument("--model", default="gpt-oss-20b", help="Nombre del modelo para LM Studio")
    parser.add_argument("--ctx", type=int, default=16000, help="Ventana de contexto")
    parser.add_argument("--api_url", default="http://127.0.0.1:1234/v1", help="URL de API de LM Studio")
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: La carpeta de entrada '{args.input_folder}' no existe.")
        sys.exit(1)

    procesar_carpeta_fragmentos(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        links_excel=args.links_excel,
        modelo=args.model,
        api_url=args.api_url,
        debug_mode=True,
        context_length=args.ctx
    )
    print("\nProcesamiento por lotes de fragmentación completado.")

if __name__ == "__main__":
    main()
