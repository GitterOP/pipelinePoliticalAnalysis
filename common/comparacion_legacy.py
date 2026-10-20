import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class MotorComparacionSemantica:
    """
    Motor matemático para evaluar la coherencia discursiva cruzando 
    propuestas de entrevistas (vectores) contra planes de gobierno oficiales.
    """
    def __init__(self, umbral_estricto: float = 0.80, umbral_minimo: float = 0.50, umbral_jaccard: float = 0.25):
        self.umbral_estricto = umbral_estricto
        self.umbral_minimo = umbral_minimo
        self.umbral_jaccard = umbral_jaccard
        self.entrevistas = []
        self.plan_gobierno = []

    def cargar_datos(self, ruta_entrevistas: str, ruta_plan: str):
        """Carga los corpus vectorizados en memoria."""
        with open(ruta_entrevistas, 'r', encoding='utf-8') as f:
            self.entrevistas = json.load(f)
        with open(ruta_plan, 'r', encoding='utf-8') as f:
            self.plan_gobierno = json.load(f)
        print(f"Datos cargados: {len(self.entrevistas)} propuestas orales | {len(self.plan_gobierno)} propuestas oficiales.")

    @staticmethod
    def similitud_coseno(vec1: List[float], vec2: List[float]) -> float:
        """Calcula la distancia del coseno entre dos vectores densos."""
        v1, v2 = np.array(vec1), np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def indice_jaccard(lista1: List[str], lista2: List[str]) -> float:
        """Calcula la intersección sobre la unión para las palabras clave."""
        set1 = set([k.lower().strip() for k in lista1])
        set2 = set([k.lower().strip() for k in lista2])
        if not set1 and not set2:
            return 0.0
        interseccion = set1.intersection(set2)
        union = set1.union(set2)
        return len(interseccion) / len(union)

    def hay_interseccion_topicos(self, topicos_a: List[str], topicos_b: List[str]) -> bool:
        """Filtro categórico duro: verifica si comparten al menos un eje temático."""
        return bool(set(topicos_a).intersection(set(topicos_b)))

    def ejecutar_emparejamiento(self, df_intervenciones: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Ejecuta el cruce algorítmico y retorna un DataFrame con los resultados.
        """
        import re as _re
        resultados = []

        for prop_oral in self.entrevistas:
            chunk_id_oral = prop_oral.get("chunk_id", "Desconocido")
            texto_oral = prop_oral["content"]["normalized_text"]
            
            # --- Contexto Entrevista N0 (solo la intervención central) ---
            contexto_oral = prop_oral.get("content", {}).get("raw_text", "N/A")
            
            # Extraer id_intervencion para buscar el texto completo del consolidado
            id_intervencion = prop_oral.get("meta_intervencion", {}).get("id_intervencion", None)
            
            if id_intervencion is None:
                # Fallback: parsear del chunk_id (ej: ENT_KF_33_frag_3_prop_1 → 33)
                match_id = _re.search(r'ENT_\w+_(\d+)_frag', chunk_id_oral)
                if match_id:
                    id_intervencion = int(match_id.group(1))
            
            if df_intervenciones is not None and id_intervencion is not None:
                try:
                    # Protocolo N0: solo la intervención central (sin expansión)
                    idx_obj = int(id_intervencion)
                    ids_buscados = [idx_obj]  # N0: solo la intervención exacta
                    
                    # Columnas reales del CSV Consolidado
                    col_id = None
                    for c in ['ID_Intervencion', 'id_intervencion', 'ID', 'Id', 'id']:
                        if c in df_intervenciones.columns:
                            col_id = c
                            break
                    
                    col_txt = None
                    for c in ['Text', 'Texto', 'texto', 'Intervencion', 'intervencion', 'Contenido']:
                        if c in df_intervenciones.columns:
                            col_txt = c
                            break
                            
                    if col_id and col_txt:
                        filas = df_intervenciones[df_intervenciones[col_id].isin(ids_buscados)]
                        if not filas.empty:
                            filas = filas.sort_values(by=col_id)
                            textos_unidos = []
                            for _, fil in filas.iterrows():
                                prefijo = f"[{fil[col_id]}]"
                                textos_unidos.append(f"{prefijo}: {fil[col_txt]}")
                            contexto_oral = "\n".join(textos_unidos)
                except Exception as e:
                    pass # Fallback al contexto raw_text individual

            topicos_oral = prop_oral["tags"]["topics"]
            keywords_oral = prop_oral["tags"]["keywords"]
            vector_oral = prop_oral.get("vector", [])

            if not vector_oral:
                continue

            mejor_match = None
            max_similitud = -1.0
            jaccard_del_mejor = 0.0

            # 1. Filtrado Categórico e iteración sobre el plan
            for prop_oficial in self.plan_gobierno:
                topicos_oficial = prop_oficial["tags"]["topics"]
                
                # Hard Filter: Solo comparar si pertenecen a la misma dimensión macro
                if not self.hay_interseccion_topicos(topicos_oral, topicos_oficial):
                    continue

                vector_oficial = prop_oficial.get("vector", [])
                if not vector_oficial:
                    continue

                # 2. Vectorización y Similitud Matemática
                similitud = self.similitud_coseno(vector_oral, vector_oficial)

                if similitud > max_similitud:
                    max_similitud = similitud
                    jaccard_del_mejor = self.indice_jaccard(keywords_oral, prop_oficial["tags"]["keywords"])
                    mejor_match = prop_oficial

            # 3. Clasificación Lógica (Thresholding)
            clase_coherencia = "SIN_MATCH_TEMATICO"
            texto_oficial_match = "N/A"
            contexto_oficial_match = "N/A"
            id_oficial_match = "N/A"

            if mejor_match:
                texto_oficial_match = mejor_match["content"]["normalized_text"]
                id_oficial_match = mejor_match.get("chunk_id", "N/A")

                # --- Contexto Plan N1 (±1 bloques) ---
                contexto_oficial_match = mejor_match.get("metadata", {}).get("texto_analizado_original", mejor_match.get("content", {}).get("raw_text", "N/A"))
                
                try:
                    import re
                    # Extraer bloque base (ej: FP26_2026_P02_B0050_S01 -> 50)
                    match_b = re.search(r'_B(\d+)_', id_oficial_match)
                    if match_b:
                        bloque_base = int(match_b.group(1))
                        bloques_permitidos = set(range(max(0, bloque_base - 1), bloque_base + 2)) # B-1 a B+1 (N1)
                        
                        textos_plan_unidos = []
                        textos_vistos = set()
                        # Recorrer todo el plan para encadenar los bloques
                        for pb in self.plan_gobierno:
                            b_id = pb.get("chunk_id", "")
                            mb = re.search(r'_B(\d+)_', b_id)
                            if mb:
                                pb_num = int(mb.group(1))
                                if pb_num in bloques_permitidos:
                                    t = pb.get("content", {}).get("raw_text", "")
                                    if t and t not in textos_vistos:
                                        textos_vistos.add(t)
                                        # Limpiar md tables
                                        t_clean = re.sub(r'\|.*\|', '', t).strip()
                                        if len(t_clean) < 10: t_clean = t # Si era solo tabla, devolvemos original
                                        textos_plan_unidos.append(t_clean)
                        
                        if textos_plan_unidos:
                            contexto_oficial_match = " [...] ".join(textos_plan_unidos)
                except Exception as e:
                    pass # Fallback al contexto estandarizado


                if max_similitud >= self.umbral_estricto:
                    clase_coherencia = "COHERENTE_DIRECTO"
                elif self.umbral_minimo <= max_similitud < self.umbral_estricto:
                    # 4. Red de Seguridad Léxica (Zona Gris)
                    if jaccard_del_mejor >= self.umbral_jaccard:
                        clase_coherencia = "COHERENTE_POR_LEXICO"
                    else:
                        clase_coherencia = "RELACIONADA_GENERICA"
                else:
                    clase_coherencia = "EMERGENTE_NUEVA"

            # Compilación del registro
            resultados.append({
                "ID_Entrevista": chunk_id_oral,
                "Topico_Macro": ", ".join(topicos_oral),
                "Contexto_Entrevista": contexto_oral,
                "Propuesta_Entrevista": texto_oral,
                "ID_Plan_Gobierno": id_oficial_match,
                "Contexto_Plan": contexto_oficial_match,
                "Propuesta_Plan": texto_oficial_match,
                "Similitud_Coseno": round(max_similitud, 4) if mejor_match else 0.0,
                "Similitud_Jaccard": round(jaccard_del_mejor, 4) if mejor_match else 0.0,
                "Clasificacion": clase_coherencia
            })

        return pd.DataFrame(resultados)

if __name__ == "__main__":
    # Inicialización del motor con los umbrales de investigación
    motor = MotorComparacionSemantica(
        umbral_estricto=0.80, 
        umbral_minimo=0.55, 
        umbral_jaccard=0.25
    )
    
    # Carga de los archivos generados en la fase de vectorización
    print("Iniciando motor de emparejamiento semántico...")
    motor.cargar_datos(
        ruta_entrevistas=r"Discursos\resultado_atomico_test.json", 
        ruta_plan=r"Planes\base_datos_final.json"
    )
    
    # Ejecución matricial
    df_resultados = motor.ejecutar_emparejamiento(df_intervenciones=None)
    
    # Exportación a formato tabular para análisis estadístico
    df_resultados.to_csv("matriz_coherencia_discursiva.csv", index=False, encoding="utf-8-sig")
    print("\nProceso finalizado. Matriz de coherencia exportada a 'matriz_coherencia_discursiva.csv'.")
    
    # Muestra de la distribución de clasificaciones
    print("\nDistribución de Coherencia Discursiva:")
    print(df_resultados["Clasificacion"].value_counts())

import os

def comparar_carpetas(
    carpeta_entrevistas: str, 
    carpeta_planes: str, 
    carpeta_resultados_consolidado: str = "Resultados/Consolidado",
    archivo_salida_final: str = "matriz_coherencia_discursiva_consolidada.csv",
    umbral_estricto: float = 0.80, 
    umbral_minimo: float = 0.55, 
    umbral_jaccard: float = 0.25
) -> pd.DataFrame:
    """
    Ejecuta el Motor de Comparación sobre todos los archivos JSON en una carpeta cruzándolos 
    exclusivamente con sus planes de gobierno correspondientes.
    Consolida todo en un único CSV maestro.
    """
    print("1. Iniciando Módulo de Comparación Semántica Mapeada...")
    motor = MotorComparacionSemantica(
        umbral_estricto=umbral_estricto, 
        umbral_minimo=umbral_minimo, 
        umbral_jaccard=umbral_jaccard
    )
    
    # Caché de CSVs consolidados por archivo de entrevista
    cache_consolidados = {}
    if os.path.exists(carpeta_resultados_consolidado):
        csvs = [f for f in os.listdir(carpeta_resultados_consolidado) if f.endswith(".csv")]
        if csvs:
            for csv_file in csvs:
                try:
                    csv_path = os.path.join(carpeta_resultados_consolidado, csv_file)
                    cache_consolidados[csv_file.replace('.csv', '')] = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"⚠️ Error cargando '{csv_file}': {e}")
            print(f"-> CSVs Consolidados cargados: {len(cache_consolidados)} archivos en caché.")
        else:
            print(f"⚠️ No hay archivos CSV en '{carpeta_resultados_consolidado}'. Se usará fallback atómico.")
    else:
        print(f"⚠️ Carpeta '{carpeta_resultados_consolidado}' no encontrada. Se usará fallback atómico.")
        
    print(f"\n2. Ejecutando cruce algorítmico individual sobre entrevistas en: {carpeta_entrevistas}...")
    resultados_totales = []
    
    if os.path.exists(carpeta_entrevistas):
        for filename in os.listdir(carpeta_entrevistas):
            if filename.endswith(".json"):
                
                # Extracción del mapeo: ID_CID_candidato_atomico.json -> CID_candidato.json
                base = filename.replace('_atomico.json', '').replace('.json', '')
                parts = base.split('_')
                if len(parts) >= 3:
                    cid_candidato = "_".join(parts[1:]) 
                    plan_filename = f"{cid_candidato}.json"
                else:
                    print(f"  ⚠️ El archivo {filename} no sigue el formato ID_CID_candidato. Saltando.")
                    continue
                    
                plan_path = os.path.join(carpeta_planes, plan_filename)
                if not os.path.exists(plan_path):
                    print(f"  ⚠️ No se encontró el plan correspondiente '{plan_filename}' para la entrevista '{filename}'. Saltando.")
                    continue

                # Buscar el CSV consolidado ESPECÍFICO para esta entrevista (ej: 10_1_keiko)
                df_consolidado_especifico = cache_consolidados.get(base, None)
                if df_consolidado_especifico is None:
                    print(f"  ⚠️ No se encontró consolidado '{base}.csv' para '{filename}'. Contexto N2 oral no disponible.")

                try:
                    # 1. Cargar el plan de gobierno ESPECÍFICO de este candidato
                    with open(plan_path, 'r', encoding='utf-8') as f:
                        data_plan = json.load(f)
                        for prop in data_plan:
                            if "metadata" not in prop:
                                prop["metadata"] = {}
                            prop["metadata"]["archivo_plan_origen"] = plan_filename
                    motor.plan_gobierno = data_plan
                    
                    # 2. Cargar las propuestas de la entrevista
                    archivo_path = os.path.join(carpeta_entrevistas, filename)
                    with open(archivo_path, 'r', encoding='utf-8') as f:
                        entrevistas = json.load(f)
                        
                    motor.entrevistas = entrevistas
                    
                    # 3. Emparejamiento Atómico con Contexto N2 del consolidado específico
                    df_parcial = motor.ejecutar_emparejamiento(df_intervenciones=df_consolidado_especifico)
                    
                    # Consolidación
                    if not df_parcial.empty:
                        df_parcial["Archivo_Entrevista_Origen"] = filename
                        resultados_totales.append(df_parcial)
                        print(f"  -> Match Completado: '{filename}' vs '{plan_filename}' (Filas: {len(df_parcial)})")
                except Exception as e:
                    print(f"Error comparando {filename} con su plan respectivo: {e}")
    else:
        print(f"La carpeta '{carpeta_entrevistas}' no existe.")
        return pd.DataFrame()
        
    if not resultados_totales:
        print("\nNo se pudo generar ningún resultado (matrices vacías o error en archivos).")
        return pd.DataFrame()
        
    print(f"\n3. Exportando matriz analítica consolidada a '{archivo_salida_final}'...")
    df_final = pd.concat(resultados_totales, ignore_index=True)
    
    # Reorganizar columnas para mejor legibilidad en CSV
    cols = df_final.columns.tolist()
    if 'Archivo_Entrevista_Origen' in cols:
        cols.insert(0, cols.pop(cols.index('Archivo_Entrevista_Origen')))
        df_final = df_final[cols]
        
    df_final.to_csv(archivo_salida_final, index=False, encoding="utf-8-sig")
    
    print("\n=== CLASIFICACIÓN FINAL DE COHERENCIA (CONSOLIDADA) ===")
    print(df_final["Clasificacion"].value_counts().to_string())
    
    return df_final