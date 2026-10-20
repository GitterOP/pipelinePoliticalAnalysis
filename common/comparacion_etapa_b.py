# -*- coding: utf-8 -*-
"""
Etapa 7B: Retrieval Hibrido (Coseno + BM25) -> Pool Top-K
==========================================================
Para cada query precomputada, busca los K mejores candidatos
en el plan de gobierno correspondiente usando fusion hibrida:
  - 70% similitud coseno (vector_hyde vs vectores_plan)
  - 30% score BM25 Okapi (texto + hyde_doc + variaciones)

Salida: Resultados/Intermedios/pool_candidatos.json
Nota: Esta etapa NO requiere GPU. Solo CPU (NumPy + BM25).
"""

import json
import os
import re
import glob
import unicodedata
import numpy as np
from typing import List, Dict, Optional
from tqdm.auto import tqdm


# ============================================================================
# UTILIDADES DE TOKENIZACION (Portado de NewMatchMet/4_metodo_match)
# ============================================================================

STOPWORDS_ES = set([
    "a", "al", "ante", "bajo", "con", "de", "del", "desde", "durante",
    "e", "el", "en", "entre", "ha", "hay", "he", "la", "las", "le",
    "les", "lo", "los", "me", "mi", "mis", "no", "nos", "o", "para",
    "por", "que", "se", "si", "su", "sus", "te", "tu", "un", "una",
    "unas", "uno", "unos", "y", "ya", "es", "son", "fue", "como",
    "mas", "pero", "esto", "esta", "ese", "esa", "esos", "esas",
    "todo", "toda", "todos", "todas", "muy", "ser", "estar", "tiene",
    "tiene", "hacer", "puede", "debe", "cada"
])


def _remove_accents(input_str: str) -> str:
    """Remueve acentos de un string para normalizacion lexica."""
    if not input_str:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def _simple_tokenizer(text: str) -> List[str]:
    """Tokeniza texto removiendo acentos, puntuacion y stopwords."""
    if not text:
        return []
    text = _remove_accents(text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS_ES and len(t) > 1]


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Similitud coseno entre dos vectores numpy."""
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)


# ============================================================================
# BASE DE DATOS DE PLANES (Index BM25 + Vectores)
# ============================================================================

class PlanDatabase:
    """
    Carga y mantiene en memoria los planes de gobierno de todos los candidatos,
    incluyendo vectores precalculados e indices BM25 por candidato.
    """

    def __init__(self, carpeta_planes: str):
        from rank_bm25 import BM25Okapi

        self.candidatos = {}
        print(f"   Cargando base de datos vectorial de planes desde: {carpeta_planes}")

        for filepath in sorted(glob.glob(os.path.join(carpeta_planes, "*.json"))):
            filename = os.path.basename(filepath)
            candidato = filename.replace(".json", "")

            with open(filepath, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            vectores = []
            corpus_tokens = []
            dict_chunks = {}
            chunk_ids_ordenados = []

            for c in chunks:
                cid = c.get('chunk_id', '')
                vec = np.array(c.get('vector', []), dtype=np.float32)
                text = c.get('content', {}).get('normalized_text', '')

                if len(vec) == 0:
                    continue

                vectores.append(vec)
                corpus_tokens.append(_simple_tokenizer(text))
                dict_chunks[cid] = c
                chunk_ids_ordenados.append(cid)

            bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

            self.candidatos[candidato] = {
                'chunks_list': chunks,
                'dict_chunks': dict_chunks,
                'chunk_ids': chunk_ids_ordenados,
                'vectores': np.array(vectores, dtype=np.float32),
                'bm25': bm25,
                'corpus_tokens': corpus_tokens
            }

            print(f"      {candidato}: {len(chunk_ids_ordenados)} chunks cargados")

    def get_candidate_data(self, candidato: str) -> Optional[Dict]:
        """Retorna los datos del candidato o None si no existe."""
        return self.candidatos.get(candidato)


# ============================================================================
# RETRIEVAL HIBRIDO
# ============================================================================

def _buscar_hibrido(
    query: Dict,
    plan_data: Dict,
    top_k: int = 10,
    peso_semantico: float = 0.70,
    peso_lexico: float = 0.30,
    normalizador_bm25: float = 15.0,
    top_retrieval: int = 100
) -> List[Dict]:
    """
    Ejecuta busqueda hibrida (coseno + BM25) para una query contra un plan.
    
    Retorna los top_k candidatos ordenados por score hibrido.
    """
    plan_vectores = plan_data['vectores']       # np.ndarray (N, 4096)
    plan_chunk_ids = plan_data['chunk_ids']      # list[str]
    plan_dict = plan_data['dict_chunks']         # dict[str, dict]
    bm25 = plan_data['bm25']                     # BM25Okapi

    # Vector de busqueda: usar hyde si disponible, si no el original
    query_vec = np.array(query.get("vector_hyde", query.get("vector_original", [])), dtype=np.float32)
    if len(query_vec) == 0:
        return []

    n_chunks = len(plan_chunk_ids)
    if n_chunks == 0:
        return []

    # --- Busqueda Semantica (Coseno) ---
    # Vectorizada: similitud coseno de 1 query contra N chunks
    norms_plan = np.linalg.norm(plan_vectores, axis=1)
    norm_query = np.linalg.norm(query_vec)
    
    if norm_query == 0:
        sem_scores = np.zeros(n_chunks)
    else:
        # dot product / (norm_query * norms_plan)
        dots = plan_vectores @ query_vec
        denominadores = norms_plan * norm_query
        denominadores = np.where(denominadores == 0, 1.0, denominadores)
        sem_scores = dots / denominadores

    # Top-N semanticos
    n_top_sem = min(top_retrieval, n_chunks)
    sem_top_indices = np.argpartition(sem_scores, -n_top_sem)[-n_top_sem:]
    top_sem = {plan_chunk_ids[idx]: float(sem_scores[idx]) for idx in sem_top_indices}

    # --- Busqueda Lexica (BM25) ---
    texto_original = query.get("texto_original", "")
    hyde_doc = query.get("hyde_doc", "")
    variaciones = " ".join(query.get("variaciones", []))
    lex_query_text = f"{texto_original} {hyde_doc} {variaciones}"
    
    if bm25 is not None:
        lex_tokens = _simple_tokenizer(lex_query_text)
        lex_scores = bm25.get_scores(lex_tokens)
    else:
        lex_scores = np.zeros(n_chunks)

    # Top-N lexicos
    n_top_lex = min(top_retrieval, n_chunks)
    lex_top_indices = np.argpartition(lex_scores, -n_top_lex)[-n_top_lex:]
    top_lex = {plan_chunk_ids[idx]: float(lex_scores[idx]) for idx in lex_top_indices}

    # --- Fusion Hibrida ---
    candidatos_ids = set(top_sem.keys()).union(set(top_lex.keys()))

    pool = []
    for rid in candidatos_ids:
        s_sem = top_sem.get(rid, 0.0)
        s_lex = top_lex.get(rid, 0.0)
        # Normalizacion del score BM25 al rango ~[0, 1]
        s_lex_norm = s_lex / normalizador_bm25 if normalizador_bm25 > 0 else 0.0
        score_hibrido = (s_sem * peso_semantico) + (s_lex_norm * peso_lexico)
        pool.append({
            "chunk_id_plan": rid,
            "score_hibrido": round(score_hibrido, 6),
            "score_semantico": round(s_sem, 6),
            "score_lexico": round(s_lex, 4),
            "texto_plan": plan_dict.get(rid, {}).get('content', {}).get('normalized_text', '')
        })

    # Ordenar por score hibrido descendente y tomar top_k
    pool.sort(key=lambda x: x["score_hibrido"], reverse=True)
    top_matches = pool[:top_k]

    # Asignar rank
    for rank, m in enumerate(top_matches, 1):
        m["rank"] = rank

    return top_matches


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def retrieval_hibrido(
    archivo_queries: str = "Resultados/Intermedios/queries_precomputadas.json",
    carpeta_planes: str = "Resultados/Chunks",
    archivo_salida: str = "Resultados/Intermedios/pool_candidatos.json",
    top_k: int = 10,
    peso_semantico: float = 0.70,
    peso_lexico: float = 0.30,
    normalizador_bm25: float = 15.0,
    top_retrieval: int = 100
):
    """
    Punto de entrada principal de la Etapa 7B.
    
    Para cada query precomputada en archivo_queries, busca los top_k mejores
    candidatos en el plan de gobierno correspondiente usando busqueda hibrida.
    
    Parametros:
    -----------
    archivo_queries : str
        Ruta al JSON generado por Etapa 7A
    carpeta_planes : str
        Ruta a la carpeta con los JSONs de planes de gobierno vectorizados
    archivo_salida : str
        Ruta al JSON intermedio de salida con los pools de candidatos
    top_k : int
        Numero de candidatos a retornar por query (default: 10)
    peso_semantico : float
        Peso de la busqueda coseno en la fusion (default: 0.70)
    peso_lexico : float
        Peso de la busqueda BM25 en la fusion (default: 0.30)
    normalizador_bm25 : float
        Divisor para normalizar scores BM25 al rango [0,1] (default: 15.0)
    top_retrieval : int
        Candidatos pre-filtrados por cada metodo antes de la fusion (default: 100)
    """
    print("=" * 70)
    print("ETAPA 7B: Retrieval Hibrido (Coseno + BM25) -> Pool Top-K")
    print("=" * 70)
    print(f"   Configuracion: Top-K={top_k}, Peso Sem={peso_semantico}, Peso Lex={peso_lexico}")

    # 1. Cargar queries precomputadas
    print(f"\n[Fase 1] Cargando queries precomputadas: {archivo_queries}")
    if not os.path.exists(archivo_queries):
        print(f"[ERROR] No se encontro el archivo de queries: {archivo_queries}")
        print("   Ejecute primero la Etapa 7A (precomputar_queries).")
        return

    with open(archivo_queries, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    print(f"   Total queries cargadas: {len(queries)}")

    # 2. Cargar base de datos de planes
    print(f"\n[Fase 2] Cargando planes de gobierno...")
    plan_db = PlanDatabase(carpeta_planes)

    # 3. Ejecutar busqueda hibrida para cada query
    print(f"\n[Fase 3] Ejecutando busqueda hibrida...")
    resultados = []
    candidatos_no_encontrados = set()

    # Envolver con tqdm para visualizar progreso
    for q in tqdm(queries, desc="Etapa 7B - Busqueda Hibrida"):
        candidato = q["candidato"]
        plan_data = plan_db.get_candidate_data(candidato)

        if plan_data is None:
            if candidato not in candidatos_no_encontrados:
                print(f"   [AVISO] Plan no encontrado para candidato: {candidato}")
                candidatos_no_encontrados.add(candidato)
            continue

        matches = _buscar_hibrido(
            query=q,
            plan_data=plan_data,
            top_k=top_k,
            peso_semantico=peso_semantico,
            peso_lexico=peso_lexico,
            normalizador_bm25=normalizador_bm25,
            top_retrieval=top_retrieval
        )

        resultados.append({
            "chunk_id_oral": q["chunk_id"],
            "candidato": candidato,
            "archivo_entrevista": q.get("archivo_entrevista", ""),
            "texto_original": q["texto_original"],
            "topicos_oral": q.get("topicos", []),
            "keywords_oral": q.get("keywords", []),
            "total_chunks_plan": len(plan_data['chunk_ids']),
            "matches": matches
        })

    # 4. Guardar resultados
    print(f"\n[Fase 4] Guardando pool de candidatos...")
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # 5. Estadisticas
    total_con_matches = sum(1 for r in resultados if r["matches"])
    total_sin_matches = sum(1 for r in resultados if not r["matches"])
    
    if resultados:
        scores_top1 = [r["matches"][0]["score_hibrido"] for r in resultados if r["matches"]]
        if scores_top1:
            score_medio = np.mean(scores_top1)
            score_mediana = np.median(scores_top1)
        else:
            score_medio = 0.0
            score_mediana = 0.0
    else:
        score_medio = 0.0
        score_mediana = 0.0

    print(f"\n{'=' * 70}")
    print(f"ETAPA 7B COMPLETADA")
    print(f"   Total queries procesadas: {len(resultados)}")
    print(f"   Con matches: {total_con_matches}")
    print(f"   Sin matches: {total_sin_matches}")
    print(f"   Score hibrido Top-1 (media): {score_medio:.4f}")
    print(f"   Score hibrido Top-1 (mediana): {score_mediana:.4f}")
    print(f"   Archivo: {archivo_salida}")
    print(f"{'=' * 70}")
