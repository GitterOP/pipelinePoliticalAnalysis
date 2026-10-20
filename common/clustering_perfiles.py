import os
import json
import pandas as pd
import numpy as np

def procesar_clustering(matriz_csv_path, carpeta_planes, carpeta_entrevistas, output_csv):
    """
    Transforma un conjunto de datos a nivel de propuestas individuales en una 
    matriz unificada a nivel de candidato, incorporando metadatos estadísticos 
    y tres representaciones vectoriales distintas (Global Ponderado, Oral, y Plan).
    """
    print("Iniciando procesamiento de perfiles tridimensionales para clustering...")
    
    # 1. Cargar Matriz_Final.csv
    if not os.path.exists(matriz_csv_path):
        raise FileNotFoundError(f"No se encontro el archivo: {matriz_csv_path}")
        
    df_matriz = pd.read_csv(matriz_csv_path)
    
    # Función para derivar el ID del candidato desde Archivo_Entrevista_Origen
    # Ejemplo: '10_2_chau_atomico.json' -> '2_chau'
    def extraer_candidato(filename):
        if pd.isna(filename):
            return None
        parts = str(filename).replace('_atomico.json', '').split('_', 1)
        if len(parts) > 1:
            return parts[1]
        return parts[0]
        
    df_matriz['Candidato_ID'] = df_matriz['Archivo_Entrevista_Origen'].apply(extraer_candidato)
    df_matriz = df_matriz.dropna(subset=['Candidato_ID'])
    candidatos_unicos = df_matriz['Candidato_ID'].unique()
    
    print(f"Candidatos encontrados en {matriz_csv_path}: {candidatos_unicos}")
    
    perfiles = []
    
    for candidato in candidatos_unicos:
        print(f"\n--- Procesando candidato: {candidato} ---")
        
        # --- A. Cálculo de Métricas de Comportamiento (Escalares) ---
        df_cand = df_matriz[df_matriz['Candidato_ID'] == candidato]
        
        # Calcular proporciones de la columna Clasificacion
        pct_clasif_data = {}
        if 'Clasificacion' in df_cand.columns:
            clasificaciones = df_cand['Clasificacion'].value_counts(normalize=True).to_dict()
            pct_clasif_data.update({f"Pct_{k}": v * 100 for k, v in clasificaciones.items()})
            
        # Calcular proporciones de LLM_Relacion_Logica
        if 'LLM_Relacion_Logica' in df_cand.columns:
            relaciones = df_cand['LLM_Relacion_Logica'].value_counts(normalize=True).to_dict()
            pct_clasif_data.update({f"Pct_RelLog_{k}": v * 100 for k, v in relaciones.items()})

        # Calcular proporciones de LLM_Delta_Detalle
        if 'LLM_Delta_Detalle' in df_cand.columns:
            detalles = df_cand['LLM_Delta_Detalle'].value_counts(normalize=True).to_dict()
            pct_clasif_data.update({f"Pct_DeltaDet_{k}": v * 100 for k, v in detalles.items()})
            
        # Calcular proporciones de Topico_Macro (agrupando minoritarios en 'OTROS')
        if 'Topico_Macro' in df_cand.columns:
            topicos_counts = df_cand['Topico_Macro'].value_counts(normalize=True)
            otros_pct = 0.0
            for topico, pct in topicos_counts.items():
                if pd.isna(topico): continue
                # Limpiar el string para que sea un buen nombre de columna
                t_nombre = str(topico).replace(', ', '_Y_').replace(' ', '_').replace(',', '')
                if pct < 0.05: # Agrupar los tópicos que representan menos del 5% del candidato
                    otros_pct += pct
                else:
                    pct_clasif_data[f"Pct_Topico_{t_nombre}"] = pct * 100
                    
            if otros_pct > 0:
                pct_clasif_data["Pct_Topico_OTROS"] = otros_pct * 100
            
        # Promedios de similitud (si existen las columnas)
        prom_coseno = df_cand['Score_Semantico'].mean() if 'Score_Semantico' in df_cand.columns else 0
        prom_jaccard = df_cand['Score_Lexico'].mean() if 'Score_Lexico' in df_cand.columns else 0
        
        # --- B. Recolección de Vectores ---
        vectores_plan = []
        vectores_oral = []
        
        # 1. Cargar vectores del plan de gobierno (Escritas)
        # Por ej: Resultados/Chunks/2_chau.json
        plan_path = os.path.join(carpeta_planes, f"{candidato}.json")
        if os.path.exists(plan_path):
            with open(plan_path, 'r', encoding='utf-8') as f:
                data_plan = json.load(f)
                for item in data_plan:
                    if 'vector' in item and item['vector']:
                        vectores_plan.append(item['vector'])
        else:
            print(f"  [!] Archivo de plan no encontrado: {plan_path}")
            
        # 2. Cargar vectores orales (Entrevistas)
        # Buscar en la carpeta de entrevistas todos los que correspondan al candidato
        if os.path.exists(carpeta_entrevistas):
            for filename in os.listdir(carpeta_entrevistas):
                if filename.endswith(f"_{candidato}_atomico.json"):
                    oral_path = os.path.join(carpeta_entrevistas, filename)
                    with open(oral_path, 'r', encoding='utf-8') as f:
                        data_oral = json.load(f)
                        for item in data_oral:
                            if 'vector' in item and item['vector']:
                                vectores_oral.append(item['vector'])
        else:
            print(f"  [!] Carpeta de entrevistas no encontrada: {carpeta_entrevistas}")
            
        total_escritas = len(vectores_plan)
        total_orales = len(vectores_oral)
        print(f"  > Vectores encontrados: {total_escritas} de Plan, {total_orales} de Oral.")
        
        # Determinar dimensionalidad
        dim_vector = 0
        if vectores_plan:
            dim_vector = len(vectores_plan[0])
        elif vectores_oral:
            dim_vector = len(vectores_oral[0])
            
        if dim_vector == 0:
            print(f"  [X] Omitiendo {candidato} porque no tiene vectores generados.")
            continue
            
        # Operaciones de Mean Pooling (Matrices)
        arr_plan = np.array(vectores_plan)
        arr_oral = np.array(vectores_oral)
        
        # Vector Plan (Promedio simple)
        v_plan = np.mean(arr_plan, axis=0) if arr_plan.size > 0 else np.zeros(dim_vector)
        
        # Vector Oral (Promedio simple)
        v_oral = np.mean(arr_oral, axis=0) if arr_oral.size > 0 else np.zeros(dim_vector)
        
        # Vector Global Ponderado: Plan = peso 2.0, Oral = peso 1.0
        vtodos = vectores_plan + vectores_oral
        pesos = [2.0] * total_escritas + [1.0] * total_orales
        
        if len(vtodos) > 0:
            v_global = np.average(np.array(vtodos), axis=0, weights=pesos)
        else:
            v_global = np.zeros(dim_vector)
            
        # --- C. Ensamblaje del "Súper Vector" por Candidato ---
        perfil = {
            'Candidato_ID': candidato,
            'Total_Orales': total_orales,
            'Total_Escritas': total_escritas,
            'Promedio_Score_Semantico': prom_coseno,
            'Promedio_Score_Lexico': prom_jaccard
        }
        
        # Agregar los porcentajes de clasificación
        perfil.update(pct_clasif_data)
        
        # Agregar dimensiones globales
        for i in range(dim_vector): perfil[f'Dim_Global_{i+1}'] = v_global[i]
        # Agregar dimensiones orales
        for i in range(dim_vector): perfil[f'Dim_Oral_{i+1}'] = v_oral[i]
        # Agregar dimensiones plan
        for i in range(dim_vector): perfil[f'Dim_Plan_{i+1}'] = v_plan[i]
            
        perfiles.append(perfil)
        
    if not perfiles:
        print("\nNo se pudo generar ningún perfil. Revise los datos de entrada.")
        return
        
    # Convertir a DataFrame y guardar
    df_perfiles = pd.DataFrame(perfiles)
    
    # Rellenar NaNs en columans de proporciones de clasificación con ceros
    pct_cols = [col for col in df_perfiles.columns if col.startswith('Pct_')]
    for col in pct_cols:
        df_perfiles[col] = df_perfiles[col].fillna(0)
        
    # Asegurar orden de columnas principal
    columnas_principales = ['Candidato_ID', 'Total_Orales', 'Total_Escritas', 
                            'Promedio_Score_Semantico', 'Promedio_Score_Lexico']
    columnas_principales += pct_cols
    restantes = [c for c in df_perfiles.columns if c not in columnas_principales]
    df_perfiles = df_perfiles[columnas_principales + restantes]
    
    df_perfiles.to_csv(output_csv, index=False)
    print(f"\n=======================================================")
    print(f"Proceso completado.")
    print(f"Total de candidatos procesados: {len(perfiles)}")
    print(f"Dimensiones de la matriz final CSV: {df_perfiles.shape}")
    print(f"Dimensiones de cada súper-vector interno: {dim_vector * 3}")
    print(f"El artefacto fue guardado exitosamente en: {output_csv}")
    print(f"=======================================================")

if __name__ == '__main__':
    # Configuración de Rutas (Relativas al origen del script)
    MATRIZ_FINAL_CSV = os.path.join('Resultados', 'Matriz_Final.csv')
    CARPETA_PLANES = os.path.join('Resultados', 'Chunks')
    CARPETA_ENTREVISTAS = os.path.join('Resultados', 'Atomico')
    OUTPUT_CSV = os.path.join('Resultados', 'Perfiles_Candidatos_Cluster_Ready.csv')
    
    procesar_clustering(MATRIZ_FINAL_CSV, CARPETA_PLANES, CARPETA_ENTREVISTAS, OUTPUT_CSV)
