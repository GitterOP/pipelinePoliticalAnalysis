# -*- coding: utf-8 -*-
"""
Etapa 8A: Evaluador de Propuestas LLM (Uno a Uno)
=================================================
1. Preparación: Reúne y acorta los contextos amplios (Plan max 2500, Entrevista max 2000)
   centrados en la propuesta para evitar colapsos. Guarda en matriz_preparada_{código}.json.
2. Inferencia: Compara cada propuesta oral de forma individual (1 a 1) contra cada match
   del plan usando un prompt corto y JSON simplificado.

Salida: Resultados/Intermedios/evaluaciones_lotes.json
"""

import os
import re
import json
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Any, Optional


def _limpiar_texto(text: str) -> str:
    """Remueve tablas markdown y normaliza espacios para evitar loops en LLMs locales."""
    if not text or pd.isna(text) or text == "N/A":
        return ""
    lines = str(text).split('\n')
    lines_clean = []
    for line in lines:
        # Ignorar líneas divisoras de tablas markdown (ej. |---|)
        if re.match(r'^[\s|:\-]+$', line):
            continue
        # Reemplazar el pipe '|' por espacios para preservar el texto y datos
        lines_clean.append(line.replace('|', ' '))
    cleaned = '\n'.join(lines_clean)
    # Normalizar espacios y saltos de línea
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _robust_find(contexto: str, propuesta: str) -> int:
    """Busca de forma robusta la posición de una propuesta dentro de su contexto."""
    if not contexto or not propuesta:
        return -1
    
    ctx_lower = contexto.lower()
    prop_lower = propuesta.lower()
    
    # 1. Búsqueda exacta cased o uncased del texto completo
    idx = contexto.find(propuesta)
    if idx != -1:
        return idx
    idx = ctx_lower.find(prop_lower)
    if idx != -1:
        return idx
        
    # 2. Normalización de caracteres especiales (guiones, comillas, espacios)
    def normalize_str(s: str) -> str:
        s = s.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
        s = re.sub(r'[«»“”"\'„‟‘’]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()
        
    ctx_norm = normalize_str(ctx_lower)
    prop_norm = normalize_str(prop_lower)
    
    idx_norm = ctx_norm.find(prop_norm)
    if idx_norm != -1:
        # Mapear de vuelta la posición en el texto original aproximado
        sub_anchor = prop_norm[:20]
        sub_anchor_clean = re.sub(r'[^\w\s]', '', sub_anchor).strip()
        words = sub_anchor_clean.split()
        if words:
            for word in words:
                if len(word) >= 4:
                    word_idx = ctx_lower.find(word)
                    if word_idx != -1:
                        return word_idx
        return 0
        
    # 3. Buscar sub-frases en el principio
    prop_clean = re.sub(r'[^\w\s\-]', ' ', prop_lower)
    prop_clean = re.sub(r'\s+', ' ', prop_clean).strip()
    
    for length in [50, 40, 30, 20]:
        if len(prop_clean) >= length:
            sub = prop_clean[:length]
            idx_sub = ctx_lower.find(sub)
            if idx_sub != -1:
                return idx_sub
            sub_nohyphen = sub.replace('-', ' ')
            idx_sub2 = ctx_lower.replace('-', ' ').find(sub_nohyphen)
            if idx_sub2 != -1:
                return idx_sub2
                
    # 4. Buscar sub-frases en el final
    for length in [50, 40, 30, 20]:
        if len(prop_clean) >= length:
            sub = prop_clean[-length:]
            idx_sub = ctx_lower.find(sub)
            if idx_sub != -1:
                return idx_sub
                
    # 5. Densidad de palabras (Fuzzy Matcher)
    words = [w for w in prop_clean.split() if len(w) >= 4]
    if words:
        best_idx = -1
        max_matches = 0
        step = 100
        for i in range(0, len(contexto), step):
            window = ctx_lower[i : i + 500]
            matches = sum(1 for w in words if w in window)
            if matches > max_matches:
                max_matches = matches
                best_idx = i
        if max_matches >= 2:
            return best_idx
            
    return -1


def _acortar_contexto(contexto: str, propuesta: str, max_chars: int = 2500) -> str:
    """Acorta el contexto alrededor de la propuesta si excede el límite de caracteres."""
    if not contexto or pd.isna(contexto) or contexto == "N/A":
        return ""
    if len(contexto) <= max_chars:
        return contexto

    # Buscar la propuesta usando robust_find
    idx = _robust_find(contexto, propuesta)
    
    if idx != -1:
        mitad = max_chars // 2
        inicio = max(0, idx - mitad)
        fin = min(len(contexto), idx + len(propuesta) + mitad)
        
        # Ajustar a la palabra completa más cercana
        if inicio > 0:
            primer_espacio = contexto.find(' ', inicio, inicio + 50)
            if primer_espacio != -1:
                inicio = primer_espacio + 1
        if fin < len(contexto):
            ultimo_espacio = contexto.rfind(' ', fin - 50, fin)
            if ultimo_espacio != -1:
                fin = ultimo_espacio
                
        recortado = contexto[inicio:fin].strip()
        prefix = "... " if inicio > 0 else ""
        suffix = " ..." if fin < len(contexto) else ""
        return prefix + recortado + suffix
    else:
        # Fallback: tomar el inicio del texto hasta max_chars
        recortado = contexto[:max_chars].strip()
        ultimo_espacio = recortado.rfind(' ')
        if ultimo_espacio != -1:
            recortado = recortado[:ultimo_espacio]
        return recortado + " ..."


def _buscar_propuesta_oral_dual(chunk_id_oral: str, archivo_entrevista: str, carpeta_atomicos: str, cache_atomicos: Dict[str, Dict[str, Tuple[str, str]]]) -> Tuple[str, str]:
    """Busca el texto raw y normalizado en el JSON atómico original para asegurar dualidad de texto."""
    if not archivo_entrevista or pd.isna(archivo_entrevista):
        return "", ""
    
    if archivo_entrevista not in cache_atomicos:
        ruta_completa = os.path.join(carpeta_atomicos, archivo_entrevista)
        if os.path.exists(ruta_completa):
            try:
                with open(ruta_completa, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                # Mapear chunk_id -> (raw_text, normalized_text)
                mapeo = {item['chunk_id']: (item.get('content', {}).get('raw_text', ''), item.get('content', {}).get('normalized_text', '')) for item in datos if 'chunk_id' in item}
                cache_atomicos[archivo_entrevista] = mapeo
            except Exception as e:
                cache_atomicos[archivo_entrevista] = {}
        else:
            cache_atomicos[archivo_entrevista] = {}
            
    return cache_atomicos[archivo_entrevista].get(chunk_id_oral, ("", ""))


def _buscar_propuesta_raw_plan(chunk_id_plan: str, candidato: str, carpeta_planes: str, cache_planes: Dict[str, Dict[str, Tuple[str, str]]]) -> Tuple[str, str]:
    """Busca el texto raw y normalizado en el JSON de plan original."""
    if not chunk_id_plan or pd.isna(chunk_id_plan) or chunk_id_plan == "SIN_MATCH":
        return "", ""
        
    # El archivo del plan es {CID}_{candidato}.json. Buscamos el que coincida con el nombre del candidato
    if candidato not in cache_planes:
        plan_archivo = None
        for filename in os.listdir(carpeta_planes):
            if filename.endswith(".json") and candidato in filename:
                plan_archivo = filename
                break
                
        if plan_archivo:
            ruta_completa = os.path.join(carpeta_planes, plan_archivo)
            try:
                with open(ruta_completa, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                mapeo = {item['chunk_id']: (item.get('content', {}).get('raw_text', ''), item.get('content', {}).get('normalized_text', '')) for item in datos if 'chunk_id' in item}
                cache_planes[candidato] = mapeo
            except Exception as e:
                cache_planes[candidato] = {}
        else:
            cache_planes[candidato] = {}
            
    return cache_planes[candidato].get(chunk_id_plan, ("", ""))


def _registrar_error_lote(log_path: str, motivo: str, chunk_id_oral: str, error_msg: str, response_raw: str):
    """Registra errores de inferencia en el archivo log."""
    import datetime
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n--- ERROR LOTE [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
        f.write(f"ID_ENTREVISTA: {chunk_id_oral}\n")
        f.write(f"MOTIVO: {motivo}\n")
        f.write(f"DETALLE ERROR: {error_msg}\n")
        f.write(f"RESPUESTA LLM CRUDA:\n{response_raw}\n")
        f.write("-" * 50 + "\n")


def _llamar_llm_lote(api_url: str, model_id: str, prompt_system: str, prompt_user: str, max_intentos: int = 5, log_path: str = "Resultados/llm_errores_audit.log", chunk_id_oral: str = "") -> Dict[str, Any]:
    """Realiza llamadas con reintentos a LM Studio y extrae el JSON limpio."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": 0.05,
        "max_tokens": 1000
    }
    
    ultimo_error = "Ninguno"
    for intento in range(max_intentos):
        try:
            response = requests.post(f"{api_url}/chat/completions", json=payload, timeout=120)
            if response.status_code == 200:
                contenido = response.json()['choices'][0]['message']['content']
                
                # Filtrado de respuesta corrupta o vacía
                if not contenido or len(set(contenido[:50])) < 3:
                    continue
                    
                # Extraer JSON entre brackets
                idx_inicio = contenido.find('{')
                idx_fin = contenido.rfind('}')
                
                if idx_inicio != -1 and idx_fin != -1:
                    json_str = contenido[idx_inicio:idx_fin+1]
                    try:
                        # Limpieza defensiva de caracteres invisibles o saltos
                        json_str_limpio = re.sub(r'[\n\r\t]', ' ', json_str)
                        datos = json.loads(json_str_limpio)
                        return datos
                    except json.JSONDecodeError as jde:
                        ultimo_error = f"JSONDecodeError: {str(jde)}"
                        # Intentar un reemplazo básico
                        try:
                            json_str_rep = json_str.replace("'", '"')
                            datos = json.loads(json_str_rep)
                            return datos
                        except:
                            _registrar_error_lote(log_path, f"Fallo parseo en intento {intento+1}", chunk_id_oral, ultimo_error, contenido)
                else:
                    ultimo_error = "No se encontraron brackets en la respuesta del LLM"
                    _registrar_error_lote(log_path, f"Fallo formato en intento {intento+1}", chunk_id_oral, ultimo_error, contenido)
            else:
                ultimo_error = f"HTTP Error {response.status_code}"
                _registrar_error_lote(log_path, f"HTTP Error en intento {intento+1}", chunk_id_oral, ultimo_error, response.text)
        except Exception as e:
            ultimo_error = f"Excepcion: {str(e)}"
            _registrar_error_lote(log_path, f"Excepcion en intento {intento+1}", chunk_id_oral, ultimo_error, "")
            
        import time
        time.sleep(2)
        
    return {}


def preparar_datos_validacion(
    archivo_matriz: str = "Resultados/Matriz.csv",
    carpeta_planes: str = "Resultados/Chunks",
    carpeta_atomicos: str = "Resultados/Atomico",
    archivo_salida_prep: str = None
):
    """
    Consolida la información y recorta los contextos (Plan max 2500 y Entrevista max 2000 chars)
    alrededor de la propuesta central para evitar sobrecarga y desbordamiento en el LLM.
    """
    if archivo_salida_prep is None:
        base_name = os.path.splitext(os.path.basename(archivo_matriz))[0]
        archivo_salida_prep = os.path.join("Resultados/Intermedios", f"matriz_preparada_{base_name}.json")
        
    print(f"\n   [Paso 8A-Prep] Iniciando consolidación y recorte de contextos en {archivo_salida_prep}...")
    if not os.path.exists(archivo_matriz):
        print(f"      [ERROR] No existe el archivo de la matriz de similitud: {archivo_matriz}")
        return
        
    df_matriz = pd.read_csv(archivo_matriz)
    grupos = df_matriz.groupby("ID_Entrevista")
    print(f"      Matriz cargada: {len(df_matriz)} filas. Total consultas únicas: {len(grupos)}")
    
    cache_atomicos = {}
    cache_planes = {}
    
    matriz_preparada = {}
    
    for chunk_id_oral, df_grupo in tqdm(grupos, desc="Consolidando y acortando contextos"):
        primer_row = df_grupo.iloc[0]
        candidato = primer_row["Candidato"]
        archivo_ent = primer_row["Archivo_Entrevista_Origen"]
        topic_macro = primer_row["Topico_Macro"]
        
        # Recuperar propuestas duales de la entrevista
        prop_oral_raw, prop_oral_norm = _buscar_propuesta_oral_dual(
            chunk_id_oral, archivo_ent, carpeta_atomicos, cache_atomicos
        )
        if not prop_oral_raw:
            prop_oral_raw = primer_row["Propuesta_Entrevista"]
        if not prop_oral_norm:
            prop_oral_norm = primer_row["Propuesta_Entrevista"]
            
        prop_oral_raw = _limpiar_texto(prop_oral_raw)
        prop_oral_norm = _limpiar_texto(prop_oral_norm)
        
        # Acortar contexto oral a 2000 caracteres
        contexto_oral = _acortar_contexto(primer_row["Contexto_Entrevista"], prop_oral_raw, max_chars=2000)
        contexto_oral = _limpiar_texto(contexto_oral)
        
        # Si la consulta no tiene matches (SIN_MATCH), declaramos lista vacía de candidatos
        if primer_row["ID_Plan_Gobierno"] == "SIN_MATCH":
            matriz_preparada[chunk_id_oral] = {
                "candidato": candidato,
                "archivo_entrevista": archivo_ent if not pd.isna(archivo_ent) else "",
                "topico_macro": topic_macro,
                "propuesta_oral_raw": prop_oral_raw,
                "propuesta_oral_norm": prop_oral_norm,
                "contexto_oral": contexto_oral,
                "candidatos_plan": []
            }
            continue
            
        candidatos_plan = []
        for idx, row in df_grupo.iterrows():
            chunk_id_plan = row["ID_Plan_Gobierno"]
            if not chunk_id_plan or pd.isna(chunk_id_plan) or chunk_id_plan == "SIN_MATCH":
                continue
                
            plan_raw, plan_norm = _buscar_propuesta_raw_plan(
                chunk_id_plan, candidato, carpeta_planes, cache_planes
            )
            
            # Acortar contexto del plan a 2500 caracteres centrado en el fragmento
            contexto_plan = _acortar_contexto(row["Contexto_Plan"], plan_raw if plan_raw else row.get("Propuesta_Plan", ""), max_chars=2500)
            contexto_plan = _limpiar_texto(contexto_plan)
            
            candidatos_plan.append({
                "chunk_id_plan": chunk_id_plan,
                "prop_raw": _limpiar_texto(plan_raw if plan_raw else row.get("Propuesta_Plan", "")),
                "prop_norm": _limpiar_texto(plan_norm if plan_norm else row.get("Propuesta_Plan", "")),
                "context_raw": contexto_plan,
                "rank": int(row["Rank_Match"]),
                "score_semantico": float(row["Score_Semantico"]),
                "score_lexico": float(row["Score_Lexico"])
            })
            
        matriz_preparada[chunk_id_oral] = {
            "candidato": candidato,
            "archivo_entrevista": archivo_ent if not pd.isna(archivo_ent) else "",
            "topico_macro": topic_macro,
            "propuesta_oral_raw": prop_oral_raw,
            "propuesta_oral_norm": prop_oral_norm,
            "contexto_oral": contexto_oral,
            "candidatos_plan": candidatos_plan
        }
        
    os.makedirs(os.path.dirname(archivo_salida_prep), exist_ok=True)
    with open(archivo_salida_prep, 'w', encoding='utf-8') as f:
        json.dump(matriz_preparada, f, indent=2, ensure_ascii=False)
    print(f"      [OK] Archivo consolidado acortado guardado en: {archivo_salida_prep}")


def preparar_csv_validacion(
    archivo_matriz: str = "Resultados/Matriz.csv",
    carpeta_planes: str = "Resultados/Chunks",
    carpeta_atomicos: str = "Resultados/Atomico",
    archivo_salida_csv: str = "Resultados/Intermedios/matriz_preparada.csv"
) -> pd.DataFrame:
    """
    Consolida la información, recorta los contextos (Plan max 2500 y Entrevista max 2000 chars)
    alrededor de la propuesta central, añade columnas indicando si se recortó, y guarda en CSV.
    """
    print(f"\n   [Prep CSV] Creando matriz preparada en: {archivo_salida_csv}...")
    if not os.path.exists(archivo_matriz):
        print(f"      [ERROR] No existe el archivo de la matriz de similitud: {archivo_matriz}")
        return pd.DataFrame()
        
    df_matriz = pd.read_csv(archivo_matriz)
    grupos = df_matriz.groupby("ID_Entrevista")
    print(f"      Matriz cargada: {len(df_matriz)} filas. Total consultas únicas: {len(grupos)}")
    
    cache_atomicos = {}
    cache_planes = {}
    
    filas_preparadas = []
    
    for chunk_id_oral, df_grupo in tqdm(grupos, desc="Preparando CSV de Validación"):
        primer_row = df_grupo.iloc[0]
        candidato = primer_row["Candidato"]
        archivo_ent = primer_row["Archivo_Entrevista_Origen"]
        topic_macro = primer_row["Topico_Macro"]
        
        # Recuperar propuestas duales de la entrevista
        prop_oral_raw, prop_oral_norm = _buscar_propuesta_oral_dual(
            chunk_id_oral, archivo_ent, carpeta_atomicos, cache_atomicos
        )
        if not prop_oral_raw:
            prop_oral_raw = primer_row["Propuesta_Entrevista"]
        if not prop_oral_norm:
            prop_oral_norm = primer_row["Propuesta_Entrevista"]
            
        prop_oral_raw = _limpiar_texto(prop_oral_raw)
        prop_oral_norm = _limpiar_texto(prop_oral_norm)
        
        # Determinar si el contexto oral excede el límite
        contexto_oral_original = primer_row["Contexto_Entrevista"]
        recortado_oral = False
        if pd.notna(contexto_oral_original) and contexto_oral_original != "N/A":
            recortado_oral = len(str(contexto_oral_original)) > 2000
            
        # Acortar contexto oral a 2000 caracteres
        contexto_oral = _acortar_contexto(contexto_oral_original, prop_oral_raw, max_chars=2000)
        contexto_oral = _limpiar_texto(contexto_oral)
        
        for idx, row in df_grupo.iterrows():
            chunk_id_plan = row["ID_Plan_Gobierno"]
            
            if chunk_id_plan == "SIN_MATCH" or pd.isna(chunk_id_plan):
                prop_plan_raw = ""
                prop_plan_norm = ""
                contexto_plan = ""
                recortado_plan = False
            else:
                plan_raw, plan_norm = _buscar_propuesta_raw_plan(
                    chunk_id_plan, candidato, carpeta_planes, cache_planes
                )
                
                # Determinar si el contexto del plan excede el límite
                contexto_plan_original = row["Contexto_Plan"]
                recortado_plan = False
                if pd.notna(contexto_plan_original) and contexto_plan_original != "N/A":
                    recortado_plan = len(str(contexto_plan_original)) > 2500
                
                # Acortar contexto del plan a 2500 caracteres centrado en el fragmento
                contexto_plan = _acortar_contexto(contexto_plan_original, plan_raw if plan_raw else row.get("Propuesta_Plan", ""), max_chars=2500)
                contexto_plan = _limpiar_texto(contexto_plan)
                
                prop_plan_raw = _limpiar_texto(plan_raw if plan_raw else row.get("Propuesta_Plan", ""))
                prop_plan_norm = _limpiar_texto(plan_norm if plan_norm else row.get("Propuesta_Plan", ""))
                
            filas_preparadas.append({
                "ID_Entrevista": chunk_id_oral,
                "Candidato": candidato,
                "Archivo_Entrevista_Origen": archivo_ent if not pd.isna(archivo_ent) else "",
                "Topico_Macro": topic_macro,
                "Propuesta_Entrevista_Raw": prop_oral_raw,
                "Propuesta_Entrevista_Norm": prop_oral_norm,
                "Contexto_Entrevista_Raw": contexto_oral,
                "Rank_Match": row["Rank_Match"],
                "ID_Plan_Gobierno": chunk_id_plan,
                "Propuesta_Plan_Raw": prop_plan_raw,
                "Propuesta_Plan_Norm": prop_plan_norm,
                "Contexto_Plan_Raw": contexto_plan,
                "Score_Semantico": row["Score_Semantico"],
                "Score_Lexico": row["Score_Lexico"],
                "Recortado_Entrevista": recortado_oral,
                "Recortado_Plan": recortado_plan
            })
            
    df_preparado = pd.DataFrame(filas_preparadas)
    
    os.makedirs(os.path.dirname(archivo_salida_csv), exist_ok=True)
    df_preparado.to_csv(archivo_salida_csv, index=False, encoding="utf-8-sig")
    print(f"      [OK] Guardado CSV con {len(df_preparado)} filas.")
    return df_preparado


def preparar_datos_y_csv_validacion(
    archivo_matriz: str = "Resultados/Matriz.csv",
    carpeta_planes: str = "Resultados/Chunks",
    carpeta_atomicos: str = "Resultados/Atomico",
    archivo_salida_csv: str = "Resultados/Intermedios/matriz_preparada.csv",
    archivo_salida_json: str = "Resultados/Intermedios/matriz_preparada_Matriz.json"
) -> pd.DataFrame:
    """
    Consolida la información en una sola pasada y genera simultáneamente:
    1. Un archivo JSON estructurado para la inferencia del LLM.
    2. Un archivo CSV preparado con columnas auxiliares indicando recortes.
    Ambas salidas garantizan total consistencia semántica y de textos.
    """
    print(f"\n   [Prep Unificado] Iniciando preparación paralela...")
    print(f"      CSV de salida: {archivo_salida_csv}")
    print(f"      JSON de salida: {archivo_salida_json}")
    
    if not os.path.exists(archivo_matriz):
        print(f"      [ERROR] No existe el archivo de la matriz de similitud: {archivo_matriz}")
        return pd.DataFrame()
        
    df_matriz = pd.read_csv(archivo_matriz)
    grupos = df_matriz.groupby("ID_Entrevista")
    print(f"      Matriz cargada: {len(df_matriz)} filas. Total consultas únicas: {len(grupos)}")
    
    cache_atomicos = {}
    cache_planes = {}
    
    matriz_preparada_json = {}
    filas_preparadas_csv = []
    
    for chunk_id_oral, df_grupo in tqdm(grupos, desc="Preparando CSV y JSON unificados"):
        primer_row = df_grupo.iloc[0]
        candidato = primer_row["Candidato"]
        archivo_ent = primer_row["Archivo_Entrevista_Origen"]
        topic_macro = primer_row["Topico_Macro"]
        
        # Recuperar propuestas duales de la entrevista
        prop_oral_raw, prop_oral_norm = _buscar_propuesta_oral_dual(
            chunk_id_oral, archivo_ent, carpeta_atomicos, cache_atomicos
        )
        if not prop_oral_raw:
            prop_oral_raw = primer_row["Propuesta_Entrevista"]
        if not prop_oral_norm:
            prop_oral_norm = primer_row["Propuesta_Entrevista"]
            
        prop_oral_raw = _limpiar_texto(prop_oral_raw)
        prop_oral_norm = _limpiar_texto(prop_oral_norm)
        
        # Determinar si el contexto oral excede el límite
        contexto_oral_original = primer_row["Contexto_Entrevista"]
        recortado_oral = False
        if pd.notna(contexto_oral_original) and contexto_oral_original != "N/A":
            recortado_oral = len(str(contexto_oral_original)) > 2000
            
        # Acortar contexto oral a 2000 caracteres
        contexto_oral = _acortar_contexto(contexto_oral_original, prop_oral_raw, max_chars=2000)
        contexto_oral = _limpiar_texto(contexto_oral)
        
        # Inicializar estructura JSON para esta consulta
        matriz_preparada_json[chunk_id_oral] = {
            "candidato": candidato,
            "archivo_entrevista": archivo_ent if not pd.isna(archivo_ent) else "",
            "topico_macro": topic_macro,
            "propuesta_oral_raw": prop_oral_raw,
            "propuesta_oral_norm": prop_oral_norm,
            "contexto_oral": contexto_oral,
            "candidatos_plan": []
        }
        
        # Si la consulta no tiene matches (SIN_MATCH), procesamos directo
        if primer_row["ID_Plan_Gobierno"] == "SIN_MATCH":
            # Para el CSV
            filas_preparadas_csv.append({
                "ID_Entrevista": chunk_id_oral,
                "Candidato": candidato,
                "Archivo_Entrevista_Origen": archivo_ent if not pd.isna(archivo_ent) else "",
                "Topico_Macro": topic_macro,
                "Propuesta_Entrevista_Raw": prop_oral_raw,
                "Propuesta_Entrevista_Norm": prop_oral_norm,
                "Contexto_Entrevista_Raw": contexto_oral,
                "Rank_Match": primer_row["Rank_Match"],
                "ID_Plan_Gobierno": "SIN_MATCH",
                "Propuesta_Plan_Raw": "",
                "Propuesta_Plan_Norm": "",
                "Contexto_Plan_Raw": "",
                "Score_Semantico": primer_row["Score_Semantico"],
                "Score_Lexico": primer_row["Score_Lexico"],
                "Recortado_Entrevista": recortado_oral,
                "Recortado_Plan": False
            })
            continue
            
        candidatos_plan = []
        for idx, row in df_grupo.iterrows():
            chunk_id_plan = row["ID_Plan_Gobierno"]
            if not chunk_id_plan or pd.isna(chunk_id_plan) or chunk_id_plan == "SIN_MATCH":
                # Fila en CSV para el caso sin match pero que estaba en el df_grupo
                filas_preparadas_csv.append({
                    "ID_Entrevista": chunk_id_oral,
                    "Candidato": candidato,
                    "Archivo_Entrevista_Origen": archivo_ent if not pd.isna(archivo_ent) else "",
                    "Topico_Macro": topic_macro,
                    "Propuesta_Entrevista_Raw": prop_oral_raw,
                    "Propuesta_Entrevista_Norm": prop_oral_norm,
                    "Contexto_Entrevista_Raw": contexto_oral,
                    "Rank_Match": row["Rank_Match"],
                    "ID_Plan_Gobierno": "SIN_MATCH",
                    "Propuesta_Plan_Raw": "",
                    "Propuesta_Plan_Norm": "",
                    "Contexto_Plan_Raw": "",
                    "Score_Semantico": row["Score_Semantico"],
                    "Score_Lexico": row["Score_Lexico"],
                    "Recortado_Entrevista": recortado_oral,
                    "Recortado_Plan": False
                })
                continue
                
            plan_raw, plan_norm = _buscar_propuesta_raw_plan(
                chunk_id_plan, candidato, carpeta_planes, cache_planes
            )
            
            # Determinar si el contexto del plan excede el límite
            contexto_plan_original = row["Contexto_Plan"]
            recortado_plan = False
            if pd.notna(contexto_plan_original) and contexto_plan_original != "N/A":
                recortado_plan = len(str(contexto_plan_original)) > 2500
            
            # Acortar contexto del plan a 2500 caracteres centrado en el fragmento
            contexto_plan = _acortar_contexto(contexto_plan_original, plan_raw if plan_raw else row.get("Propuesta_Plan", ""), max_chars=2500)
            contexto_plan = _limpiar_texto(contexto_plan)
            
            prop_plan_raw = _limpiar_texto(plan_raw if plan_raw else row.get("Propuesta_Plan", ""))
            prop_plan_norm = _limpiar_texto(plan_norm if plan_norm else row.get("Propuesta_Plan", ""))
            
            # Estructurar para el JSON
            candidatos_plan.append({
                "chunk_id_plan": chunk_id_plan,
                "prop_raw": prop_plan_raw,
                "prop_norm": prop_plan_norm,
                "context_raw": contexto_plan,
                "rank": int(row["Rank_Match"]),
                "score_semantico": float(row["Score_Semantico"]),
                "score_lexico": float(row["Score_Lexico"])
            })
            
            # Fila para el CSV
            filas_preparadas_csv.append({
                "ID_Entrevista": chunk_id_oral,
                "Candidato": candidato,
                "Archivo_Entrevista_Origen": archivo_ent if not pd.isna(archivo_ent) else "",
                "Topico_Macro": topic_macro,
                "Propuesta_Entrevista_Raw": prop_oral_raw,
                "Propuesta_Entrevista_Norm": prop_oral_norm,
                "Contexto_Entrevista_Raw": contexto_oral,
                "Rank_Match": row["Rank_Match"],
                "ID_Plan_Gobierno": chunk_id_plan,
                "Propuesta_Plan_Raw": prop_plan_raw,
                "Propuesta_Plan_Norm": prop_plan_norm,
                "Contexto_Plan_Raw": contexto_plan,
                "Score_Semantico": row["Score_Semantico"],
                "Score_Lexico": row["Score_Lexico"],
                "Recortado_Entrevista": recortado_oral,
                "Recortado_Plan": recortado_plan
            })
            
        matriz_preparada_json[chunk_id_oral]["candidatos_plan"] = candidatos_plan

    # Guardar JSON
    os.makedirs(os.path.dirname(archivo_salida_json), exist_ok=True)
    with open(archivo_salida_json, 'w', encoding='utf-8') as f:
        json.dump(matriz_preparada_json, f, indent=2, ensure_ascii=False)
    print(f"      [OK] JSON estructurado guardado en: {archivo_salida_json}")
        
    # Guardar CSV
    df_preparado = pd.DataFrame(filas_preparadas_csv)
    os.makedirs(os.path.dirname(archivo_salida_csv), exist_ok=True)
    df_preparado.to_csv(archivo_salida_csv, index=False, encoding="utf-8-sig")
    print(f"      [OK] CSV de diagnóstico guardado en: {archivo_salida_csv}")
    
    return df_preparado


def evaluar_lotes(
    archivo_matriz: str = "Resultados/Matriz.csv",
    carpeta_planes: str = "Resultados/Chunks",
    carpeta_atomicos: str = "Resultados/Atomico",
    archivo_salida_json: str = "Resultados/Intermedios/evaluaciones_lotes.json",
    model_id: str = "openai/gpt-oss-20b",
    api_url: str = "http://127.0.0.1:1234/v1",
    archivo_prep: str = None
):
    """
    Punto de entrada de la Etapa 8A (Versión Uno a Uno).
    
    Compara 1 a 1 cada propuesta del plan contra la oral.
    Optimiza a nivel de candidatos para máxima robustez y soporte de interrupciones.
    """
    print("=" * 70)
    print("ETAPA 8A: Evaluador LLM (Uno a Uno)")
    print("=" * 70)
    
    if archivo_prep is None:
        base_name = os.path.splitext(os.path.basename(archivo_matriz))[0]
        archivo_prep = os.path.join("Resultados/Intermedios", f"matriz_preparada_{base_name}.json")
    
    # 1. Asegurar preparación de datos
    if not os.path.exists(archivo_prep):
        preparar_datos_validacion(archivo_matriz, carpeta_planes, carpeta_atomicos, archivo_prep)
    else:
        print(f"   Usando archivo de preparación pre-existente: {archivo_prep}")
        
    with open(archivo_prep, 'r', encoding='utf-8') as f:
        matriz_preparada = json.load(f)
        
    # Cargar checkpoints si existen
    evaluaciones = {}
    if os.path.exists(archivo_salida_json):
        try:
            with open(archivo_salida_json, 'r', encoding='utf-8') as f:
                evaluaciones = json.load(f)
            print(f"   Checkpoint recuperado: {len(evaluaciones)} queries registradas.")
        except Exception as e:
            print(f"   Aviso: No se pudo cargar el checkpoint ({e}). Iniciando desde cero.")
            
    # loop over queries
    queries_total = len(matriz_preparada)
    log_path = "Resultados/llm_errores_audit.log"
    
    # Prompt de sistema para evaluación 1 a 1
    prompt_system = """Eres un analista político y auditor de discurso implacable. Tu tarea es comparar una promesa hecha en televisión (Entrevista) contra una propuesta específica del documento oficial (Plan de Gobierno) para evaluar su coherencia.
Analiza la relación lógica de la propuesta del plan frente a la entrevista.

Retorna ÚNICAMENTE un objeto JSON válido con esta estructura exacta:
{
  "relacion_logica": "EQUIVALENCIA" | "CONTRADICCION" | "FALSA_EQUIVALENCIA" | "NOVEDAD_ABSOLUTA",
  "delta_de_detalle": "ENTREVISTA_MAS_DETALLADA" | "PLAN_MAS_DETALLADO" | "DETALLE_SIMILAR",
  "explicacion": "Justifica tu decisión analíticamente en máximo 40 palabras."
}

Definiciones de Relación Lógica:
- EQUIVALENCIA: Ambas fuentes proponen el mismo mecanismo de acción o reforma específica.
- CONTRADICCION: La propuesta en la entrevista choca o se opone al espíritu/letra de lo escrito en el plan.
- FALSA_EQUIVALENCIA: El plan menciona un lineamiento general o valor abstracto, pero el candidato lo usa como excusa para prometer en TV una política específica que NO está escrita ni sugerida en el plan (rellena vacíos con demagogia).
- NOVEDAD_ABSOLUTA: El tema no existe en el plan de gobierno o no tiene ninguna relación con las propuestas presentadas."""

    # Contador de guardados para checkpointing
    guardados = 0
    
    for chunk_id_oral, data_prep in tqdm(matriz_preparada.items(), desc="Evaluando 1 a 1"):
        candidato = data_prep["candidato"]
        prop_oral_raw = data_prep["propuesta_oral_raw"]
        prop_oral_norm = data_prep["propuesta_oral_norm"]
        contexto_oral = data_prep["contexto_oral"]
        candidatos_plan = data_prep["candidatos_plan"]
        
        # Si ya existe en checkpoint, lo tomamos como base
        if chunk_id_oral in evaluaciones:
            resultado_query = evaluaciones[chunk_id_oral]
        else:
            resultado_query = {
                "candidato": candidato,
                "propuesta_oral_raw": prop_oral_raw,
                "propuesta_oral_norm": prop_oral_norm,
                "contexto_oral": contexto_oral,
                "lotes_evaluaciones": {}
            }
            
        if not candidatos_plan:
            # Caso directo sin match
            evaluaciones[chunk_id_oral] = resultado_query
            continue
            
        # Evaluar uno a uno cada candidato del plan que no esté evaluado
        cambios = False
        for cand in candidatos_plan:
            plan_id = cand["chunk_id_plan"]
            
            # Si ya se evaluó este candidato específico, saltarlo
            if plan_id in resultado_query["lotes_evaluaciones"]:
                continue
                
            prompt_user = f"""Contexto Amplio (Entrevista): "{contexto_oral}"
Propuesta (Entrevista - Normalizado): "{prop_oral_norm}"

Propuesta a evaluar (Plan de Gobierno):
Contexto Amplio (Plan): "{cand['context_raw']}"
Propuesta Escrita (Plan - Normalizado): "{cand['prop_norm']}"

INSTRUCCIÓN: Compara la propuesta del plan anterior frente a la propuesta de la entrevista y emite la relación lógica, delta de detalle y explicación en el JSON solicitado:"""

            # Invocación al LLM
            evaluacion_ind = _llamar_llm_lote(
                api_url=api_url,
                model_id=model_id,
                prompt_system=prompt_system,
                prompt_user=prompt_user,
                chunk_id_oral=chunk_id_oral,
                log_path=log_path
            )
            
            # Validaciones preventivas básicas
            relacion = evaluacion_ind.get("relacion_logica", "ERROR")
            if relacion not in ["EQUIVALENCIA", "CONTRADICCION", "FALSA_EQUIVALENCIA", "NOVEDAD_ABSOLUTA"]:
                relacion = "NOVEDAD_ABSOLUTA"
                
            detalle = evaluacion_ind.get("delta_de_detalle", "NO_APLICA")
            if detalle not in ["ENTREVISTA_MAS_DETALLADA", "PLAN_MAS_DETALLADO", "DETALLE_SIMILAR", "NO_APLICA"]:
                detalle = "NO_APLICA"
                
            explicacion = evaluacion_ind.get("explicacion", "Sin explicacion.")
            
            # Guardar en la estructura
            resultado_query["lotes_evaluaciones"][plan_id] = {
                "relacion_logica": relacion,
                "delta_de_detalle": detalle,
                "explicacion": explicacion,
                "rank": cand["rank"],
                "score_semantico": cand["score_semantico"],
                "score_lexico": cand["score_lexico"],
                "prop_raw_plan": cand["prop_raw"],
                "prop_norm_plan": cand["prop_norm"],
                "contexto_plan": cand["context_raw"]
            }
            cambios = True
            
        # Si se hicieron evaluaciones, registrar y guardar
        evaluaciones[chunk_id_oral] = resultado_query
        if cambios:
            guardados += 1
            # Guardar periódicamente cada 10 consultas de entrevista
            if guardados % 10 == 0:
                os.makedirs(os.path.dirname(archivo_salida_json), exist_ok=True)
                with open(archivo_salida_json, 'w', encoding='utf-8') as f:
                    json.dump(evaluaciones, f, indent=2, ensure_ascii=False)
                    
    # Guardado final de la etapa
    os.makedirs(os.path.dirname(archivo_salida_json), exist_ok=True)
    with open(archivo_salida_json, 'w', encoding='utf-8') as f:
        json.dump(evaluaciones, f, indent=2, ensure_ascii=False)
        
    print(f"\n   [OK] Etapa 8A finalizada con exito.")
    print(f"        Total queries evaluadas guardadas en checkpoint: {len(evaluaciones)}")
    print(f"        Resultados escritos en: {archivo_salida_json}")
    print("=" * 70)


if __name__ == "__main__":
    evaluar_lotes()
