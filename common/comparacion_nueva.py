# -*- coding: utf-8 -*-
"""
Etapa 7D: Ensamblaje de Matriz.csv
===================================
Toma el pool rerankeado (o el pool de candidatos si no se ejecuto 7C)
y ensambla la Matriz.csv final con los Top-K matches por query.

La clasificacion es por niveles de score (alto, medio, bajo) de forma
referencial, ya que el modelo de clasificacion del Paso 8 recibe el
Top-10 completo independientemente del nivel.

Entrada: pool_rerankeado.json (o pool_candidatos.json)
Salida:  Resultados/Matriz.csv
"""

import json
import os
import re
import pandas as pd
from typing import List, Dict, Optional
from tqdm.auto import tqdm


# ============================================================================
# CLASIFICACION POR NIVELES DE SCORE
# ============================================================================

def _clasificar_nivel_score(score: float, usa_reranker: bool) -> str:
    """
    Clasifica el match en niveles de score (referencial).
    
    Si usa reranker (Etapa 7C ejecutada):
        - ALTO:  score_reranker >= 0.60
        - MEDIO: score_reranker >= 0.30
        - BAJO:  score_reranker < 0.30
    
    Si solo usa retrieval hibrido (sin reranker):
        - ALTO:  score_hibrido >= 0.65
        - MEDIO: score_hibrido >= 0.40
        - BAJO:  score_hibrido < 0.40
    """
    if usa_reranker:
        if score >= 0.60:
            return "ALTO"
        elif score >= 0.30:
            return "MEDIO"
        else:
            return "BAJO"
    else:
        if score >= 0.65:
            return "ALTO"
        elif score >= 0.40:
            return "MEDIO"
        else:
            return "BAJO"


# ============================================================================
# ENRIQUECIMIENTO DE CONTEXTO
# ============================================================================

def _cargar_consolidados(carpeta_consolidados: str) -> Dict[str, pd.DataFrame]:
    """Carga todos los CSVs consolidados en un cache por nombre base."""
    cache = {}
    if not os.path.exists(carpeta_consolidados):
        return cache
    
    for csv_file in os.listdir(carpeta_consolidados):
        if csv_file.endswith(".csv"):
            try:
                csv_path = os.path.join(carpeta_consolidados, csv_file)
                cache[csv_file.replace('.csv', '')] = pd.read_csv(csv_path)
            except Exception as e:
                print(f"   Aviso: Error cargando '{csv_file}': {e}")
    
    return cache


def _obtener_contexto_oral(chunk_id_oral: str, archivo_entrevista: str,
                           cache_consolidados: Dict[str, pd.DataFrame]) -> str:
    """
    Obtiene el contexto oral N0 (texto de la intervencion original)
    desde los CSVs consolidados.
    """
    base = archivo_entrevista.replace('_atomico.json', '').replace('.json', '')
    df_consolidado = cache_consolidados.get(base, None)
    
    if df_consolidado is None:
        return "N/A"
    
    # Extraer id_intervencion del chunk_id (ej: ENT_KF_33_frag_3_prop_1 -> 33)
    match_id = re.search(r'ENT_\w+_(\d+)_frag', chunk_id_oral)
    if not match_id:
        return "N/A"
    
    id_intervencion = int(match_id.group(1))
    
    # Buscar columna de ID
    col_id = None
    for c in ['ID_Intervencion', 'id_intervencion', 'ID', 'Id', 'id']:
        if c in df_consolidado.columns:
            col_id = c
            break
    
    col_txt = None
    for c in ['Text', 'Texto', 'texto', 'Intervencion', 'intervencion', 'Contenido']:
        if c in df_consolidado.columns:
            col_txt = c
            break
    
    if col_id and col_txt:
        fila = df_consolidado[df_consolidado[col_id] == id_intervencion]
        if not fila.empty:
            return str(fila.iloc[0][col_txt])
    
    return "N/A"


def _obtener_contexto_plan(chunk_id_plan: str, plan_chunks: List[Dict]) -> str:
    """
    Obtiene el contexto del plan N1 (+-1 bloques adyacentes).
    """
    # Extraer numero de bloque del chunk_id
    match_b = re.search(r'_B(\d+)_', chunk_id_plan)
    if not match_b:
        # Fallback: devolver el raw_text del chunk
        for c in plan_chunks:
            if c.get("chunk_id") == chunk_id_plan:
                return c.get("content", {}).get("raw_text", "N/A")
        return "N/A"
    
    bloque_base = int(match_b.group(1))
    bloques_permitidos = set(range(max(0, bloque_base - 1), bloque_base + 2))
    
    textos_plan_unidos = []
    textos_vistos = set()
    
    for pb in plan_chunks:
        b_id = pb.get("chunk_id", "")
        mb = re.search(r'_B(\d+)_', b_id)
        if mb:
            pb_num = int(mb.group(1))
            if pb_num in bloques_permitidos:
                t = pb.get("content", {}).get("raw_text", "")
                if t and t not in textos_vistos:
                    textos_vistos.add(t)
                    # Limpiar tablas markdown
                    t_clean = re.sub(r'\|.*\|', '', t).strip()
                    if len(t_clean) < 10:
                        t_clean = t
                    textos_plan_unidos.append(t_clean)
    
    if textos_plan_unidos:
        return " [...] ".join(textos_plan_unidos)
    
    return "N/A"


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def ensamblar_matriz(
    archivo_pool: str = "Resultados/Intermedios/pool_rerankeado.json",
    carpeta_planes: str = "Resultados/Chunks",
    carpeta_consolidados: str = "Resultados/Consolidado",
    archivo_salida: str = "Resultados/Matriz.csv",
    top_n: int = 10
):
    """
    Punto de entrada principal de la Etapa 7D.
    
    Ensambla la Matriz.csv final a partir del pool de candidatos (rerankeado o no).
    Incluye Top-N matches por propuesta oral para que el Paso 8 (clasificacion)
    pueda evaluar multiples candidatos.
    
    Parametros:
    -----------
    archivo_pool : str
        Ruta al JSON de pool rerankeado (7C) o pool candidatos (7B).
        Si archivo_pool no existe, intenta cargar pool_candidatos.json como fallback.
    carpeta_planes : str
        Ruta a la carpeta con los JSONs de planes de gobierno
    carpeta_consolidados : str
        Ruta a la carpeta con los CSVs consolidados (contexto oral)
    archivo_salida : str
        Ruta al CSV de salida (Matriz.csv)
    top_n : int
        Cuantos matches incluir por propuesta oral (default: 10)
    """
    print("=" * 70)
    print("ETAPA 7D: Ensamblaje de Matriz.csv")
    print("=" * 70)

    # 1. Determinar si se uso reranker o no
    usa_reranker = False
    if os.path.exists(archivo_pool):
        print(f"   Cargando pool desde: {archivo_pool}")
    else:
        # Fallback al pool sin reranker
        fallback = archivo_pool.replace("pool_rerankeado", "pool_candidatos")
        if os.path.exists(fallback):
            archivo_pool = fallback
            print(f"   Pool rerankeado no encontrado. Usando pool de candidatos: {fallback}")
        else:
            print(f"[ERROR] No se encontro ningun pool de candidatos.")
            print("   Ejecute primero la Etapa 7B y/o 7C.")
            return pd.DataFrame()

    with open(archivo_pool, 'r', encoding='utf-8') as f:
        pool_data = json.load(f)
    print(f"   Total queries en pool: {len(pool_data)}")

    # Detectar si es pool rerankeado
    if pool_data and pool_data[0].get("matches"):
        first_match = pool_data[0]["matches"][0]
        usa_reranker = "score_reranker" in first_match
    
    print(f"   Modo: {'Con Reranker (Etapa 7C)' if usa_reranker else 'Solo Retrieval Hibrido (Etapa 7B)'}")

    # 2. Cargar planes de gobierno para contexto
    print(f"\n[Fase 1] Cargando planes de gobierno para contexto...")
    planes_cache = {}
    if os.path.exists(carpeta_planes):
        for filename in os.listdir(carpeta_planes):
            if filename.endswith(".json"):
                filepath = os.path.join(carpeta_planes, filename)
                candidato = filename.replace(".json", "")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        planes_cache[candidato] = json.load(f)
                except Exception as e:
                    print(f"   Error cargando plan {filename}: {e}")

    # 3. Cargar CSVs consolidados para contexto oral
    print(f"[Fase 2] Cargando CSVs consolidados...")
    cache_consolidados = _cargar_consolidados(carpeta_consolidados)
    print(f"   CSVs consolidados cargados: {len(cache_consolidados)}")

    # 4. Ensamblar filas de la Matriz
    print(f"\n[Fase 3] Ensamblando Matriz (Top-{top_n} por query)...")
    filas = []

    # Envolver con tqdm para visualizar el progreso del ensamblaje
    for pool_entry in tqdm(pool_data, desc="Etapa 7D - Ensamblando Matriz"):
        chunk_id_oral = pool_entry["chunk_id_oral"]
        candidato = pool_entry["candidato"]
        texto_oral = pool_entry.get("texto_original", "")
        topicos_oral = pool_entry.get("topicos_oral", [])
        keywords_oral = pool_entry.get("keywords_oral", [])
        archivo_ent = pool_entry.get("archivo_entrevista", "")
        matches = pool_entry.get("matches", [])

        # Contexto oral
        contexto_oral = _obtener_contexto_oral(chunk_id_oral, archivo_ent, cache_consolidados)

        # Plan de gobierno del candidato
        plan_chunks = planes_cache.get(candidato, [])

        # Tomar solo top_n matches
        matches_seleccionados = matches[:top_n]

        if not matches_seleccionados:
            # Sin matches encontrados
            filas.append({
                "Archivo_Entrevista_Origen": archivo_ent,
                "ID_Entrevista": chunk_id_oral,
                "Topico_Macro": ", ".join(topicos_oral),
                "Contexto_Entrevista": contexto_oral,
                "Propuesta_Entrevista": texto_oral,
                "Rank_Match": 0,
                "ID_Plan_Gobierno": "SIN_MATCH",
                "Contexto_Plan": "N/A",
                "Propuesta_Plan": "N/A",
                "Score_Principal": 0.0,
                "Score_Semantico": 0.0,
                "Score_Lexico": 0.0,
                "Score_Reranker": 0.0 if usa_reranker else None,
                "Nivel_Score": "SIN_MATCH",
                "Candidato": candidato
            })
            continue

        for match in matches_seleccionados:
            chunk_id_plan = match.get("chunk_id_plan", "N/A")
            texto_plan = match.get("texto_plan", "N/A")

            # Score principal
            if usa_reranker:
                score_principal = match.get("score_reranker", 0.0)
            else:
                score_principal = match.get("score_hibrido", 0.0)

            # Nivel de score (referencial)
            nivel = _clasificar_nivel_score(score_principal, usa_reranker)

            # Contexto del plan (N1)
            contexto_plan = _obtener_contexto_plan(chunk_id_plan, plan_chunks)

            filas.append({
                "Archivo_Entrevista_Origen": archivo_ent,
                "ID_Entrevista": chunk_id_oral,
                "Topico_Macro": ", ".join(topicos_oral),
                "Contexto_Entrevista": contexto_oral,
                "Propuesta_Entrevista": texto_oral,
                "Rank_Match": match.get("rank", 0),
                "ID_Plan_Gobierno": chunk_id_plan,
                "Contexto_Plan": contexto_plan,
                "Propuesta_Plan": texto_plan,
                "Score_Principal": round(score_principal, 6),
                "Score_Semantico": round(match.get("score_semantico", 0.0), 6),
                "Score_Lexico": round(match.get("score_lexico", 0.0), 4),
                "Score_Reranker": round(match.get("score_reranker", 0.0), 6) if usa_reranker else None,
                "Nivel_Score": nivel,
                "Candidato": candidato
            })

    # 5. Construir DataFrame y guardar
    df_final = pd.DataFrame(filas)

    # Remover columna Score_Reranker si no se uso reranker
    if not usa_reranker and "Score_Reranker" in df_final.columns:
        df_final = df_final.drop(columns=["Score_Reranker"])

    # Ordenar por candidato, entrevista, y rank
    df_final = df_final.sort_values(
        by=["Candidato", "Archivo_Entrevista_Origen", "ID_Entrevista", "Rank_Match"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    df_final.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

    # 6. Estadisticas
    total_filas = len(df_final)
    queries_unicas = df_final["ID_Entrevista"].nunique()
    
    if "Nivel_Score" in df_final.columns:
        distribucion = df_final["Nivel_Score"].value_counts()
    else:
        distribucion = pd.Series()

    print(f"\n{'=' * 70}")
    print(f"ETAPA 7D COMPLETADA")
    print(f"   Total filas en Matriz: {total_filas}")
    print(f"   Queries unicas: {queries_unicas}")
    print(f"   Matches por query: Top-{top_n}")
    print(f"   Modo: {'Con Reranker' if usa_reranker else 'Solo Retrieval Hibrido'}")
    print(f"\n   Distribucion de Niveles de Score:")
    if not distribucion.empty:
        for nivel, count in distribucion.items():
            pct = count / total_filas * 100
            print(f"      {nivel}: {count} ({pct:.1f}%)")
    print(f"\n   Archivo: {archivo_salida}")
    print(f"{'=' * 70}")

    return df_final
