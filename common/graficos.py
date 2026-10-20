import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
import re
import unicodedata

warnings.filterwarnings('ignore')

# Mapeo global de identificadores a nombres formales y completos en español
MAPEO_CANDIDATOS = {
    "1_keiko": "Keiko Fujimori",
    "2_rla": "Rafael López Aliaga",
    "3_carlosal": "Carlos Álvarez",
    "4_chau": "A. López-Chau",
    "5_jorgeni": "Jorge Nieto",
    "6_robertosa": "Roberto Sánchez",
    "7_cesara": "César Acuña",
    "8_cerron": "Vladimir Cerrón"
}

MAPEO_TOPICOS = {
    'SEGURIDAD_CIUDADANA': 'Citizen Security',
    'ECONOMIA_Y_EMPLEO': 'Economy and Employment',
    'JUSTICIA_Y_ANTICORRUPCION': 'Justice and Anti-Corruption',
    'SALUD_Y_BIENESTAR': 'Health and Well-being',
    'EDUCACION_Y_CIENCIA': 'Education and Science',
    'INFRAESTRUCTURA_Y_TERRITORIO': 'Infrastructure and Territory',
    'MINERIA_Y_AMBIENTE': 'Mining and Environment',
    'REFORMA_ESTADO': 'State Reform',
    'OTROS': 'Others'
}

MAPEO_LOGICA = {
    'EQUIVALENCIA': 'Equivalence',
    'FALSA_EQUIVALENCIA': 'False Equivalence',
    'NOVEDAD_ABSOLUTA': 'Absolute Novelty',
    'CONTRADICCION': 'Contradiction',
    'ERROR': 'Error'
}

MAPEO_DETALLE = {
    'PLAN_MAS_DETALLADO': 'Plan More Detailed',
    'ENTREVISTA_MAS_DETALLADA': 'Interview More Detailed',
    'DETALLE_SIMILAR': 'Similar Detail',
    'NO_APLICA': 'Not Applicable'
}

def _limpiar_nombre_archivo(texto: str) -> str:
    """Elimina acentos, eñes y caracteres especiales para guardar archivos de forma segura en Windows."""
    texto = unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('ASCII')
    return texto.lower().replace(" ", "_")


def generar_reporte_grafico(ruta_csv: str, dir_salida: str = "Resultados", guardar_como_png: bool = False, mostrar_inline: bool = True):
    """
    Lee la matriz auditada final y genera 4 gráficos de nivel académico.
    Diseñado para escalar dinámicamente desde 2 hasta 36+ candidatos sin romper la interfaz.
    """
    print(f"1. Cargando datos desde: {ruta_csv}")
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{ruta_csv}'.")
        return

    # Mapear los identificadores de candidatos a nombres formales y completos en español
    if 'Candidato' in df.columns:
        df['Candidato'] = df['Candidato'].map(MAPEO_CANDIDATOS).fillna(df['Candidato'])
    elif 'Archivo_Entrevista_Origen' in df.columns:
        df['Candidato'] = df['Archivo_Entrevista_Origen'].apply(
            lambda x: "_".join(x.replace('_atomico.json', '').replace('.json', '').split('_')[2:]).replace('_', ' ').title()
            if len(x.replace('_atomico.json', '').replace('.json', '').split('_')) >= 3 else "Desconocido"
        )
        df['Candidato'] = df['Candidato'].apply(lambda name: MAPEO_CANDIDATOS.get(name.lower().replace(" ", "_"), name))
        
    num_candidatos = df['Candidato'].nunique()
    print(f"   -> Encontrados {len(df)} registros para {num_candidatos} candidato(s) distintos.")
    
    if guardar_como_png and not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    # Configuración base de estilo académico
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 🎨 PALETA DE COLORES ESTRICTA (Semántica Política)
    # Verde: Equivalencia | Amarillo: Falsa | Rojo: Novedad | Negro/Gris: Contradicción
    colores_logica = {
        'Equivalence': '#2ecc71',        # Verde esmeralda
        'False Equivalence': '#f1c40f',  # Amarillo girasol
        'Absolute Novelty': '#e74c3c',    # Rojo suave
        'Contradiction': '#2c3e50',       # Negro azulado/Gris oscuro
        'Error': '#95a5a6'                # Gris
    }

    # Altura dinámica para gráficos que involucren a todos los candidatos
    altura_dinamica = max(6, num_candidatos * 0.5) 

    print("\n2. Generando Gráfico 1: Perfil de Coherencia Lógica (COLORES SEMÁNTICOS)...")
    fig1, ax1 = plt.subplots(figsize=(12, altura_dinamica))
    
    # Mapear las categorías lógicas
    df['LLM_Relacion_Logica_Label'] = df['LLM_Relacion_Logica'].map(MAPEO_LOGICA).fillna(df['LLM_Relacion_Logica'])
    crosstab_logica = pd.crosstab(df['Candidato'], df['LLM_Relacion_Logica_Label'], normalize='index') * 100
    
    # Asegurar orden y colores correctos
    columnas_presentes = [c for c in ['Equivalence', 'False Equivalence', 'Absolute Novelty', 'Contradiction', 'Error'] if c in crosstab_logica.columns]
    crosstab_logica = crosstab_logica[columnas_presentes]
    colores_lista = [colores_logica[c] for c in columnas_presentes]

    if 'Equivalence' in crosstab_logica.columns:
        crosstab_logica = crosstab_logica.sort_values(by='Equivalence', ascending=True)

    crosstab_logica.plot(kind='barh', stacked=True, ax=ax1, color=colores_lista)
    ax1.set_title('Logical Coherence Profile (LLM Audit)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Percentage of Proposals (%)', fontsize=12)
    ax1.set_ylabel('Candidate', fontsize=12)
    plt.legend(title='Logical Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if guardar_como_png:
        fig1.savefig(os.path.join(dir_salida, '1_perfil_logica_colores.png'), dpi=300, bbox_inches='tight')
    if mostrar_inline:
        plt.show()
    plt.close(fig1)

    print("\n2b. Generando Nuevo Gráfico: Coherencia Matemática (Clasificación Coseno)...")
    # Este gráfico analiza la columna 'Nivel_Score' (o 'Clasificacion' como fallback)
    columna_math = None
    if 'Nivel_Score' in df.columns:
        columna_math = 'Nivel_Score'
    elif 'Clasificacion' in df.columns:
        columna_math = 'Clasificacion'
        
    if columna_math:
        fig1b, ax1b = plt.subplots(figsize=(12, altura_dinamica))
        
        # Mapear categorías matemáticas a inglés
        MAPEO_MATH = {
            'ALTO': 'HIGH',
            'MEDIO': 'MEDIUM',
            'BAJO': 'LOW',
            'SIN_MATCH': 'NO_MATCH',
            'COHERENTE_DIRECTO': 'DIRECT_COHERENT',
            'COHERENTE_POR_LEXICO': 'LEXICAL_COHERENT',
            'RELACIONADA_GENERICA': 'GENERIC_RELATED',
            'EMERGENTE_NUEVA': 'NEW_EMERGING',
            'SIN_MATCH_TEMATICO': 'NO_THEMATIC_MATCH'
        }
        df['Math_Label'] = df[columna_math].map(MAPEO_MATH).fillna(df[columna_math])
        crosstab_math = pd.crosstab(df['Candidato'], df['Math_Label'], normalize='index') * 100
        
        # Escala de colores similar (Familia Verde/Azul para coherentes, Rojos para emergentes)
        colores_math = {
            # Categorías para Nivel_Score
            'HIGH': '#1e8449',         # Verde Esmeralda profundo
            'MEDIUM': '#2ecc71',        # Verde Vivo
            'LOW': '#f1c40f',         # Amarillo Girasol
            'NO_MATCH': '#c0392b',    # Rojo Sangre / Oscuro
            
            # Categorías legacy para Clasificacion
            'DIRECT_COHERENT': '#1e8449',
            'LEXICAL_COHERENT': '#2ecc71',
            'GENERIC_RELATED': '#3498db',
            'NEW_EMERGING': '#e74c3c',
            'NO_THEMATIC_MATCH': '#c0392b'
        }
        
        orden_categorias = ['HIGH', 'MEDIUM', 'LOW', 'NO_MATCH'] if columna_math == 'Nivel_Score' else ['DIRECT_COHERENT', 'LEXICAL_COHERENT', 'GENERIC_RELATED', 'NEW_EMERGING', 'NO_THEMATIC_MATCH']
        
        cols_math_pres = [c for c in orden_categorias if c in crosstab_math.columns]
        crosstab_math = crosstab_math[cols_math_pres]
        colores_math_lista = [colores_math[c] for c in cols_math_pres]

        crosstab_math.plot(kind='barh', stacked=True, ax=ax1b, color=colores_math_lista)
        
        titulo_grafico = 'Mathematical Coherence Profile (Score Level)' if columna_math == 'Nivel_Score' else 'Mathematical Coherence Profile (Vector Distance)'
        ax1b.set_title(titulo_grafico, fontsize=14, fontweight='bold')
        ax1b.set_xlabel('Percentage of Proposals (%)', fontsize=12)
        ax1b.set_ylabel('Candidate', fontsize=12)
        plt.legend(title='Similarity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if guardar_como_png:
            fig1b.savefig(os.path.join(dir_salida, '1b_perfil_matematico.png'), dpi=300, bbox_inches='tight')
        if mostrar_inline:
            plt.show()
        plt.close(fig1b)

    print("3. Generando Matrices de Inflación por Candidato...")
    # GRAFICO 2: Matriz de Asimetría de Detalle vs Lógica (INDIVIDUAL POR CANDIDATO)
    for cand_name in df['Candidato'].unique():
        # Mapear columnas de detalle y lógica
        df_cand = df[
            (df['Candidato'] == cand_name) &
            (df['LLM_Delta_Detalle'] != 'NO_APLICA') &
            (df['LLM_Delta_Detalle'] != 'ERROR')
        ].copy()
        
        if df_cand.empty:
            continue
        
        df_cand['Relacion_Logica_Label'] = df_cand['LLM_Relacion_Logica'].map(MAPEO_LOGICA).fillna(df_cand['LLM_Relacion_Logica'])
        df_cand['Delta_Detalle_Label'] = df_cand['LLM_Delta_Detalle'].map(MAPEO_DETALLE).fillna(df_cand['LLM_Delta_Detalle'])

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        heatmap_data = pd.crosstab(df_cand['Relacion_Logica_Label'], df_cand['Delta_Detalle_Label'])
        
        cats_logica = ['Equivalence', 'False Equivalence', 'Absolute Novelty', 'Contradiction']
        cats_detalle = ['Interview More Detailed', 'Plan More Detailed', 'Similar Detail']
        
        filas_validas = [c for c in cats_logica if c in heatmap_data.index]
        cols_validas = [c for c in cats_detalle if c in heatmap_data.columns]
        heatmap_data = heatmap_data.reindex(index=filas_validas, columns=cols_validas, fill_value=0)
        
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Oranges', ax=ax2, cbar_kws={'label': 'Number of Proposals'})
        ax2.set_title(f'Inflation Matrix: {cand_name}\n(Detail Asymmetry vs Logic)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Detail Asymmetry', fontsize=12)
        ax2.set_ylabel('Logical Relation', fontsize=12)
        plt.tight_layout()
        
        if guardar_como_png:
            safe_name = _limpiar_nombre_archivo(cand_name)
            fig2.savefig(os.path.join(dir_salida, f'2_matriz_inflacion_{safe_name}.png'), dpi=300, bbox_inches='tight')
        
        if mostrar_inline:
            plt.show()
            
        plt.close(fig2)

    print("4. Generando Gráfico 3: Coherencia Temática...")
    # Definición de categorías válidas para limpieza
    valid_topics = {
        "SEGURIDAD_CIUDADANA", "ECONOMIA_Y_EMPLEO", "JUSTICIA_Y_ANTICORRUPCION", "SALUD_Y_BIENESTAR", 
        "EDUCACION_Y_CIENCIA", "INFRAESTRUCTURA_Y_TERRITORIO", "MINERIA_Y_AMBIENTE", "REFORMA_ESTADO"
    }
    
    def limpiar_topico(texto):
        if not texto or pd.isna(texto): return "Others"
        # Tomamos el primer tópico si hay varios (separados por coma)
        primera_parte = str(texto).split(',')[0].strip()
        # Quitamos subtópicos (limpieza después de :)
        t_base = primera_parte.split(':')[0].strip().upper()
        return MAPEO_TOPICOS.get(t_base, "Others")

    fig3, ax3 = plt.subplots(figsize=(14, altura_dinamica))
    df_equiv = df.copy()
    if 'LLM_Relacion_Logica' in df.columns:
        df_equiv['Es_Equivalente'] = (df_equiv['LLM_Relacion_Logica'] == 'EQUIVALENCIA').astype(int)
        # Aplicamos la limpieza de categoría macro
        df_equiv['Topico_Limpio'] = df_equiv['Topico_Macro'].apply(limpiar_topico)
        
        tema_coherencia = df_equiv.groupby(['Candidato', 'Topico_Limpio'])['Es_Equivalente'].mean().unstack() * 100
        tema_coherencia = tema_coherencia.fillna(0)
        
        # Asegurar orden de filas (Ejes) y transponer para poner candidatos en el eje horizontal (X)
        orden_ejes = [c for c in ["Citizen Security", "Economy and Employment", "Justice and Anti-Corruption", "Health and Well-being", "Education and Science", "Infrastructure and Territory", "Mining and Environment", "State Reform", "Others"] if c in tema_coherencia.columns]
        tema_coherencia = tema_coherencia[orden_ejes].T # Transposición: Filas=Ejes, Columnas=Candidatos
        
        sns.heatmap(tema_coherencia, annot=True, fmt='.0f', cmap='YlGn', ax=ax3, cbar_kws={'label': '% Equivalence with the Plan'})
        ax3.set_title('Heatmap: % of Proposals Faithful to the Plan by Thematic Axis', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Candidate', fontsize=12)
        ax3.set_ylabel('Thematic Axis (Canonical Taxonomy)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if guardar_como_png:
            fig3.savefig(os.path.join(dir_salida, '3_coherencia_tematica.png'), dpi=300, bbox_inches='tight')
        if mostrar_inline:
            plt.show()
    plt.close(fig3)

    print("4b. Generando Gráfico 3b: Densidad y Coherencia Temática (Burbujas)...")
    if 'LLM_Relacion_Logica' in df.columns:
        # Preparamos datos: Media para color, Conteo para tamaño
        stats = df_equiv.groupby(['Candidato', 'Topico_Limpio']).agg(
            Coherencia=('Es_Equivalente', 'mean'),
            Volumen=('Es_Equivalente', 'count')
        ).reset_index()
        stats['Coherencia_Pct'] = stats['Coherencia'] * 100
        
        fig3b, ax3b = plt.subplots(figsize=(14, altura_dinamica))
        
        # Mapeo de categorías para asegurar el orden en el eje Y
        stats['Topico_Limpio'] = pd.Categorical(stats['Topico_Limpio'], categories=orden_ejes[::-1], ordered=True)
        
        scatter = ax3b.scatter(
            x=stats['Candidato'], 
            y=stats['Topico_Limpio'], 
            s=stats['Volumen'] * 40, # Escalado del tamaño un poco más agresivo para visibilidad
            c=stats['Coherencia_Pct'], 
            cmap='YlGn', 
            alpha=0.7, 
            edgecolors='grey',
            linewidth=0.5
        )
        
        # Añadir barra de color
        cbar = plt.colorbar(scatter, ax=ax3b)
        cbar.set_label('% Equivalence with the Plan')
        
        ax3b.set_title('Thematic Density: Volume vs Coherence\n(Size = No. of Proposals | Color = % Fidelity)', fontsize=14, fontweight='bold')
        ax3b.set_xlabel('Candidate', fontsize=12)
        ax3b.set_ylabel('Thematic Axis', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if guardar_como_png:
            fig3b.savefig(os.path.join(dir_salida, '3b_densidad_burbujas.png'), dpi=300, bbox_inches='tight')
        if mostrar_inline:
            plt.show()
        plt.close(fig3b)


    print("5. Generando Gráfico 4: Distribución Matemática vs Lógica...")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    df['Relacion_Logica_Label'] = df['LLM_Relacion_Logica'].map(MAPEO_LOGICA).fillna(df['LLM_Relacion_Logica'])
    orden_violin = ['Equivalence', 'False Equivalence', 'Absolute Novelty', 'Contradiction']
    orden_violin_pres = [c for c in orden_violin if c in df['Relacion_Logica_Label'].unique()]
    
    sns.violinplot(data=df, x='Relacion_Logica_Label', y='Score_Semantico', order=orden_violin_pres, palette='muted', ax=ax4, inner="quartile")
    sns.stripplot(data=df, x='Relacion_Logica_Label', y='Score_Semantico', order=orden_violin_pres, size=3, color="black", alpha=0.3, ax=ax4)
    
    # Línea de referencia (Umbral Matemático)
    ax4.axhline(0.45, color='red', linestyle='--', linewidth=2, label='Threshold (0.45)')
    
    ax4.set_title('Methodological Justification: Cosine (Machine) vs Logic (LLM)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Final Logical Classification (LLM)', fontsize=12)
    ax4.set_ylabel('Cosine Similarity (Vector Distance)', fontsize=12)
    ax4.legend()
    plt.tight_layout()
    if guardar_como_png:
        fig4.savefig(os.path.join(dir_salida, '4_distribucion_coseno.png'), dpi=300, bbox_inches='tight')
    if mostrar_inline:
        plt.show()
    plt.close(fig4)


def _build_corpus(carpeta_atomico: str, carpeta_chunks: str) -> dict:
    entrevistas = {}
    for filename in os.listdir(carpeta_atomico):
        if not filename.endswith('.json'): continue
        base = filename.replace('_atomico.json', '').replace('.json', '')
        partes = base.split('_')
        if len(partes) < 2: continue
        cid_cand = "_".join(partes[1:])
        try:
            with open(os.path.join(carpeta_atomico, filename), 'r', encoding='utf-8') as f:
                props = json.load(f)
            entrevistas.setdefault(cid_cand, []).extend(props)
        except: pass
    
    planes = {}
    for filename in os.listdir(carpeta_chunks):
        if not filename.endswith('.json'): continue
        cid_cand = filename.replace('.json', '')
        try:
            with open(os.path.join(carpeta_chunks, filename), 'r', encoding='utf-8') as f:
                planes[cid_cand] = json.load(f)
        except: pass
    
    corpus = {}
    for cid, props_entrevista in entrevistas.items():
        nombre_display = MAPEO_CANDIDATOS.get(cid, " ".join(cid.split('_')[1:]).title())
        corpus[nombre_display] = {'entrevista': props_entrevista, 'plan': planes.get(cid, [])}
    return corpus


def _extraer_tags(propuestas: list, campo: str, mapear_topicos: bool = False) -> pd.Series:
    valid_topics = {
        "SEGURIDAD_CIUDADANA", "ECONOMIA_Y_EMPLEO", "JUSTICIA_Y_ANTICORRUPCION", "SALUD_Y_BIENESTAR", 
        "EDUCACION_Y_CIENCIA", "INFRAESTRUCTURA_Y_TERRITORIO", "MINERIA_Y_AMBIENTE", "REFORMA_ESTADO"
    }
    valores = []
    for prop in propuestas:
        v = prop.get('tags', {}).get(campo, None)
        if not v: continue
        items = v if isinstance(v, list) else [v]
        for x in items:
            t = str(x).strip().upper()
            if mapear_topicos and campo == 'topics':
                t_base = t.split(":")[0].strip().upper()
                if t_base in valid_topics: valores.append(t_base)
                else: valores.append("OTROS")
            else:
                valores.append(t)
    return pd.Series(valores).value_counts() if valores else pd.Series(dtype=int)


def grafico_sentimientos_por_candidato(carpeta_atomico: str, carpeta_chunks: str, top_n: int = 3, dir_salida: str = "Resultados", guardar_como_png: bool = False, mostrar_inline: bool = True):
    corpus = _build_corpus(carpeta_atomico, carpeta_chunks)
    if not corpus: return
    plt.style.use('seaborn-v0_8-whitegrid')
    COLOR_PLAN, COLOR_ENT = '#e67e22', '#3498db'

    for candidato, datos in corpus.items():
        s_ent_pct = (_extraer_tags(datos['entrevista'], 'sentiment_tone'))
        s_plan_pct = (_extraer_tags(datos['plan'], 'sentiment_tone'))
        s_ent_pct = s_ent_pct / s_ent_pct.sum() * 100 if not s_ent_pct.empty else pd.Series(dtype=float)
        s_plan_pct = s_plan_pct / s_plan_pct.sum() * 100 if not s_plan_pct.empty else pd.Series(dtype=float)
        
        # Traducir sentimientos si es necesario
        MAPEO_SENTIMIENTO = {
            'POSITIVO': 'POSITIVE',
            'NEUTRO': 'NEUTRAL',
            'NEGATIVO': 'NEGATIVE'
        }
        s_plan_pct.index = s_plan_pct.index.map(lambda x: MAPEO_SENTIMIENTO.get(str(x).upper(), x))
        s_ent_pct.index = s_ent_pct.index.map(lambda x: MAPEO_SENTIMIENTO.get(str(x).upper(), x))

        top_plan, top_ent = s_plan_pct.head(top_n).index, s_ent_pct.head(top_n).index
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Speech Tone: {candidato}', fontsize=13, fontweight='bold')
        
        for ax, top, title in zip([ax1, ax2], [top_plan, top_ent], ['Top PLAN', 'Top INTERVIEW']):
            y = range(len(top))
            v_p, v_e = s_plan_pct.reindex(top, fill_value=0), s_ent_pct.reindex(top, fill_value=0)
            ax.barh([i+0.17 for i in y], v_p, height=0.35, color=COLOR_PLAN, label='Plan')
            ax.barh([i-0.17 for i in y], v_e, height=0.35, color=COLOR_ENT, label='Interview')
            ax.set_yticks(y)
            ax.set_yticklabels(top)
            ax.set_title(title)
            ax.legend()
        plt.tight_layout()
        if guardar_como_png:
            os.makedirs(dir_salida, exist_ok=True)
            cand_clean = _limpiar_nombre_archivo(candidato)
            fig.savefig(os.path.join(dir_salida, f'5_sentimientos_{cand_clean}.png'))
        if mostrar_inline: plt.show()
        plt.close(fig)


def grafico_topicos_por_candidato(carpeta_atomico: str, carpeta_chunks: str, top_n: int = 9, dir_salida: str = "Resultados", guardar_como_png: bool = False, mostrar_inline: bool = True):
    """ Muestra el balance discursivo entre las 9 categorías canónicas (8 + OTROS). """
    corpus = _build_corpus(carpeta_atomico, carpeta_chunks)
    if not corpus: return
    categorias_ordenadas = ["Citizen Security", "Economy and Employment", "Justice and Anti-Corruption", "Health and Well-being", "Education and Science", "Infrastructure and Territory", "Mining and Environment", "State Reform", "Others"]

    for candidato, datos in corpus.items():
        t_ent = _extraer_tags(datos['entrevista'], 'topics', mapear_topicos=True)
        t_plan = _extraer_tags(datos['plan'], 'topics', mapear_topicos=True)
        
        # Traducir los índices al inglés en caso de que vengan en formato raw
        t_ent.index = t_ent.index.map(lambda x: MAPEO_TOPICOS.get(str(x).upper(), x))
        t_plan.index = t_plan.index.map(lambda x: MAPEO_TOPICOS.get(str(x).upper(), x))

        t_ent_pct = (t_ent / t_ent.sum() * 100).reindex(categorias_ordenadas, fill_value=0) if not t_ent.empty else pd.Series(0, index=categorias_ordenadas)
        t_plan_pct = (t_plan / t_plan.sum() * 100).reindex(categorias_ordenadas, fill_value=0) if not t_plan.empty else pd.Series(0, index=categorias_ordenadas)

        fig, ax = plt.subplots(figsize=(12, 7))
        y = np.arange(len(categorias_ordenadas))
        ax.barh(y + 0.2, t_ent_pct, height=0.4, label='Interview', color='#2980b9')
        ax.barh(y - 0.2, t_plan_pct, height=0.4, label='Plan', color='#e74c3c')
        
        for i, (ve, vp) in enumerate(zip(t_ent_pct, t_plan_pct)):
            if ve > 0.1: ax.text(ve+0.3, i+0.2, f'{ve:.1f}%', va='center', fontsize=8)
            if vp > 0.1: ax.text(vp+0.3, i-0.2, f'{vp:.1f}%', va='center', fontsize=8)
            
        ax.set_yticks(y)
        ax.set_yticklabels(categorias_ordenadas)
        ax.set_title(f'Discursive Balance: {candidato}')
        ax.legend()
        plt.tight_layout()
        if guardar_como_png:
            os.makedirs(dir_salida, exist_ok=True)
            cand_clean = _limpiar_nombre_archivo(candidato)
            fig.savefig(os.path.join(dir_salida, f'6_topicos_{cand_clean}.png'))
        if mostrar_inline: plt.show()
        plt.close(fig)
