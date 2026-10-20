# -*- coding: utf-8 -*-
"""
Script de Reporte Gráfico Reducido
==================================
Lee la matriz de resultados auditada final (Resultados/Matriz_Final.csv)
y genera de forma independiente tres gráficos académicos: el perfil de coherencia,
la densidad tematica de burbujas y la comparacion de score vs logica, sin títulos
y con tópicos sin subguiones en formato Title Case.
"""

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# ============================================================================
# CONFIGURACIÓN DE CANDIDATOS Y COLORES CONSTANTES
# ============================================================================

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
    'CONTRADICCION': 'Contradiction',
    'NOVEDAD_ABSOLUTA': 'Invention',
    'FALSA_EQUIVALENCIA': 'Invention'
}

# ============================================================================
# FUNCIONES DE FORMATEO
# ============================================================================

def limpiar_topico_burbuja(texto: str) -> str:
    """Normaliza y mapea el nombre del tópico para el gráfico de burbujas."""
    if not texto or pd.isna(texto):
        return "Otros"
    # Tomar la primera categoría si hay varias
    primera_parte = str(texto).split(',')[0].strip()
    t_base = primera_parte.split(':')[0].strip().upper()
    t_base = t_base if t_base in MAPEO_TOPICOS else "OTROS"
    return MAPEO_TOPICOS[t_base]


# ============================================================================
# FUNCIÓN PRINCIPAL DE REPORTES
# ============================================================================

def generar_reporte_graficos_reducido(
    ruta_csv: str = "Resultados/Matriz_Final.csv",
    dir_salida: str = "graficos"
):
    """
    Carga la matriz auditada y genera exactamente los Gráficos 1, 3b y 4 en el directorio indicado.
    Ninguno de los gráficos contiene título en su cabecera.
    """
    print(f"Cargando datos desde '{ruta_csv}'...")
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encontró el archivo '{ruta_csv}'.")
        return

    df = pd.read_csv(ruta_csv)
    os.makedirs(dir_salida, exist_ok=True)

    # 0. Cargar Score_Principal desde la Matriz original si no está presente
    if 'Score_Principal' not in df.columns:
        ruta_matriz_original = os.path.join(os.path.dirname(ruta_csv), "Matriz.csv")
        if os.path.exists(ruta_matriz_original):
            try:
                print(f"Cargando Score_Principal desde '{ruta_matriz_original}'...")
                df_orig = pd.read_csv(ruta_matriz_original)
                if 'Score_Principal' in df_orig.columns:
                    df_score = df_orig[['ID_Entrevista', 'ID_Plan_Gobierno', 'Score_Principal']].drop_duplicates()
                    df_score = df_score.groupby(['ID_Entrevista', 'ID_Plan_Gobierno'])['Score_Principal'].max().reset_index()
                    df = pd.merge(df, df_score, on=['ID_Entrevista', 'ID_Plan_Gobierno'], how='left')
            except Exception as e:
                print(f"   Aviso: No se pudo fusionar con la matriz original: {e}")
        
        if 'Score_Principal' not in df.columns:
            print("   Aviso: Columna 'Score_Principal' no disponible. Usando cálculo híbrido por defecto (70% Coseno + 30% Léxico).")
            df['Score_Principal'] = 0.7 * df['Score_Semantico'] + 0.3 * df['Score_Lexico']

    # 1. Normalizar nombres de Candidatos
    if 'Candidato' in df.columns:
        df['Candidato_Label'] = df['Candidato'].map(MAPEO_CANDIDATOS).fillna(df['Candidato'])
    else:
        df['Candidato_Label'] = "Desconocido"

    num_candidatos = df['Candidato_Label'].nunique()
    altura_dinamica = max(6, num_candidatos * 0.5)

    # Configuración de estilo académico
    plt.style.use('seaborn-v0_8-whitegrid')

    # ============================================================================
    # GRÁFICO 1: Perfil de Coherencia Lógica (Colores Semánticos Agrupados)
    # ============================================================================
    print("\nGenerando Gráfico 1: Perfil de Coherencia Lógica...")
    
    colores_logica = {
        'Equivalence': '#2ecc71',    # Verde
        'Invention': '#f1c40f',       # Amarillo
        'Contradiction': '#e74c3c'    # Rojo
    }

    fig1, ax1 = plt.subplots(figsize=(12, altura_dinamica))
    
    # Mapear y agrupar las categorías lógicas
    df['LLM_Relacion_Logica_Label'] = df['LLM_Relacion_Logica'].map(MAPEO_LOGICA).fillna(df['LLM_Relacion_Logica'])
    
    crosstab_logica = pd.crosstab(df['Candidato_Label'], df['LLM_Relacion_Logica_Label'], normalize='index') * 100
    
    # Orden deseado en las columnas apiladas
    columnas_ordenadas = [c for c in ['Equivalence', 'Invention', 'Contradiction'] if c in crosstab_logica.columns]
    crosstab_logica = crosstab_logica[columnas_ordenadas]
    colores_lista = [colores_logica[c] for c in columnas_ordenadas]

    # Ordenar candidatos de menor a mayor equivalencia
    if 'Equivalence' in crosstab_logica.columns:
        crosstab_logica = crosstab_logica.sort_values(by='Equivalence', ascending=True)

    crosstab_logica.plot(kind='barh', stacked=True, ax=ax1, color=colores_lista, width=0.6)
    
    # Calcular cantidades absolutas alineadas
    crosstab_counts = pd.crosstab(df['Candidato_Label'], df['LLM_Relacion_Logica_Label'])
    crosstab_counts = crosstab_counts.reindex(index=crosstab_logica.index, columns=crosstab_logica.columns).fillna(0)
    
    # Agregar etiquetas de porcentaje y cantidad dentro de las barras (solo si > 3.0% para evitar amontonamientos)
    for i, c in enumerate(ax1.containers):
        col_name = crosstab_logica.columns[i]
        cantidades = crosstab_counts[col_name].values
        labels = [f'{pct:.1f}%\n({int(qty)})' if pct > 3.0 else '' for pct, qty in zip(c.datavalues, cantidades)]
        ax1.bar_label(c, labels=labels, label_type='center', fontsize=10, color='white', fontweight='bold')
    
    # Se remueve el título por requerimiento visual del usuario
    ax1.set_xlabel('Percentage of Proposals (%)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Candidate', fontsize=12, labelpad=10)
    plt.legend(title='Logical Category', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    
    path_g1 = os.path.join(dir_salida, 'reporte_1_perfil_logica.png')
    fig1.savefig(path_g1, dpi=300, bbox_inches='tight')
    print(f"   -> Gráfico 1 guardado en: {path_g1}")
    plt.close(fig1)

    # ============================================================================
    # GRÁFICO 3b: Densidad y Coherencia Temática (Burbujas)
    # ============================================================================
    if 'LLM_Relacion_Logica' in df.columns:
        print("\nGenerando Gráfico 3b: Densidad y Coherencia Temática (Burbujas)...")
        
        df_equiv = df.copy()
        df_equiv['Es_Equivalente'] = (df_equiv['LLM_Relacion_Logica'] == 'EQUIVALENCIA').astype(int)
        df_equiv['Topico_Limpio'] = df_equiv['Topico_Macro'].apply(limpiar_topico_burbuja)

        # Agrupar estadísticas: Porcentaje de Coherencia y Volumen
        stats = df_equiv.groupby(['Candidato_Label', 'Topico_Limpio']).agg(
            Coherencia=('Es_Equivalente', 'mean'),
            Volumen=('Es_Equivalente', 'count')
        ).reset_index()
        stats['Coherencia_Pct'] = stats['Coherencia'] * 100

        fig3b, ax3b = plt.subplots(figsize=(7.2, 5.2))

        # Categorías ordenadas limpias
        orden_ejes = [
            "Citizen Security", "Economy and Employment", "Justice and Anti-Corruption", 
            "Health and Well-being", "Education and Science", "Infrastructure and Territory", 
            "Mining and Environment", "State Reform", "Others"
        ]
        
        stats['Topico_Limpio'] = pd.Categorical(stats['Topico_Limpio'], categories=orden_ejes[::-1], ordered=True)

        scatter = ax3b.scatter(
            x=stats['Candidato_Label'], 
            y=stats['Topico_Limpio'], 
            s=stats['Volumen'] * 20,
            c=stats['Coherencia_Pct'], 
            cmap='YlGn', 
            alpha=0.7, 
            edgecolors='grey',
            linewidth=0.5
        )

        cbar = plt.colorbar(scatter, ax=ax3b, shrink=0.7)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('% Equivalence with the Plan', fontsize=9, labelpad=5)

        # Se remueve el título por requerimiento visual del usuario
        ax3b.set_xlabel('Candidate', fontsize=9.5, fontweight='bold', labelpad=4)
        ax3b.set_ylabel('Thematic Axis', fontsize=9.5, fontweight='bold', labelpad=4)
        plt.xticks(rotation=35, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()

        path_g3b = os.path.join(dir_salida, 'reporte_3b_densidad_burbujas.png')
        fig3b.savefig(path_g3b, dpi=300, bbox_inches='tight')
        print(f"   -> Gráfico 3b guardado en: {path_g3b}")
        plt.close(fig3b)

    # ============================================================================
    # GRÁFICO 4: Distribución de Métricas RAG vs Lógica (Panel Triple)
    # ============================================================================
    print("\nGenerando Gráfico 4: Panel Triple de Distribución (Semántico, Léxico e Híbrido)...")
    
    fig4, (ax_sem, ax_lex, ax_comb) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    
    # Orden para el eje X
    orden_eje_x = [c for c in ['Equivalence', 'Invention', 'Contradiction'] if c in df['LLM_Relacion_Logica_Label'].unique()]
    paleta_violin = {c: colores_logica.get(c, '#95a5a6') for c in orden_eje_x}

    # Subplot 1: Semántico (Coseno)
    sns.violinplot(
        data=df, 
        x='LLM_Relacion_Logica_Label', 
        y='Score_Semantico', 
        order=orden_eje_x,
        palette=paleta_violin, 
        ax=ax_sem, 
        inner="quartile"
    )
    sns.stripplot(
        data=df, 
        x='LLM_Relacion_Logica_Label', 
        y='Score_Semantico', 
        order=orden_eje_x,
        size=3, 
        color="black", 
        alpha=0.3, 
        ax=ax_sem
    )
    ax_sem.set_title("a)", fontsize=12, fontweight='bold', pad=10)
    ax_sem.set_xlabel('Final Logical Classification (LLM)', fontsize=11, labelpad=10)
    ax_sem.set_ylabel('Semantic Similarity (Cosine)', fontsize=11, labelpad=10)

    # Subplot 2: Léxico (Jaccard / BM25)
    sns.violinplot(
        data=df, 
        x='LLM_Relacion_Logica_Label', 
        y='Score_Lexico', 
        order=orden_eje_x,
        palette=paleta_violin, 
        ax=ax_lex, 
        inner="quartile"
    )
    sns.stripplot(
        data=df, 
        x='LLM_Relacion_Logica_Label', 
        y='Score_Lexico', 
        order=orden_eje_x,
        size=3, 
        color="black", 
        alpha=0.3, 
        ax=ax_lex
    )
    ax_lex.set_title("b)", fontsize=12, fontweight='bold', pad=10)
    ax_lex.set_xlabel('Final Logical Classification (LLM)', fontsize=11, labelpad=10)
    ax_lex.set_ylabel('Lexical Similarity (Jaccard / BM25)', fontsize=11, labelpad=10)

    # Subplot 3: Combinado / Híbrido (Score Principal)
    sns.violinplot(
        data=df, 
        x='LLM_Relacion_Logica_Label', 
        y='Score_Principal', 
        order=orden_eje_x,
        palette=paleta_violin, 
        ax=ax_comb, 
        inner="quartile"
    )
    sns.stripplot(
        data=df, 
        x='LLM_Relacion_Logica_Label', 
        y='Score_Principal', 
        order=orden_eje_x,
        size=3, 
        color="black", 
        alpha=0.3, 
        ax=ax_comb
    )
    ax_comb.set_title("c)", fontsize=12, fontweight='bold', pad=10)
    ax_comb.set_xlabel('Final Logical Classification (LLM)', fontsize=11, labelpad=10)
    ax_comb.set_ylabel('Final Hybrid Ranking Metric', fontsize=11, labelpad=10)

    plt.tight_layout()

    path_g4 = os.path.join(dir_salida, 'reporte_4_distribucion_coseno.png')
    fig4.savefig(path_g4, dpi=300, bbox_inches='tight')
    print(f"   -> Gráfico 4 (Panel Triple) guardado en: {path_g4}")
    plt.close(fig4)

    # ============================================================================
    # GRÁFICO 7: Mapa de Navegación Discursiva (Adherencia vs Innovación)
    # ============================================================================
    print("\nGenerando Gráfico 7: Mapa de Navegación Discursiva...")
    
    # Calcular métricas por candidato
    df_metrics = []
    for cand_name in df['Candidato_Label'].unique():
        df_cand = df[df['Candidato_Label'] == cand_name]
        total = len(df_cand)
        
        # Conteo de categorías mapeadas
        equivalencias = df_cand[df_cand['LLM_Relacion_Logica_Label'] == 'Equivalence']
        invenciones = df_cand[df_cand['LLM_Relacion_Logica_Label'] == 'Invention']
        contradicciones = df_cand[df_cand['LLM_Relacion_Logica_Label'] == 'Contradiction']
        
        pct_equiv = (len(equivalencias) / total) * 100
        pct_inven = (len(invenciones) / total) * 100
        pct_contra = (len(contradicciones) / total) * 100
        
        df_metrics.append({
            'Candidato': cand_name,
            'Adherencia': pct_equiv,
            'Innovacion': pct_inven,
            'Contradiccion': pct_contra,
            'Total': total
        })
        
    df_metrics = pd.DataFrame(df_metrics)
    
    fig7, ax7 = plt.subplots(figsize=(11, 8))
    
    # Paleta consistente de colores por candidato
    PALETA_CANDIDATOS = {
        'Keiko Fujimori': '#e65100',       # Naranja
        'Roberto Sánchez': '#2e7d32',      # Verde
        'Vladimir Cerrón': '#c62828',      # Rojo
        'César Acuña': '#1565c0',          # Azul
        'Rafael López Aliaga': '#0288d1',  # Celeste
        'Jorge Nieto': '#fbc02d',          # Amarillo
        'Carlos Álvarez': '#757575',       # Gris
        'A. López-Chau': '#d81b60'         # Rosado
    }
    
    # Graficar cada candidato como un círculo
    # Configuración de etiquetas para evitar superposiciones y colisiones
    CONFIG_ETIQUETAS = {
        'Rafael López Aliaga': {'dx': 0.0, 'dy': -3.2, 'ha': 'center', 'va': 'top'},
        'César Acuña': {'dx': 1.6, 'dy': 0.0, 'ha': 'left', 'va': 'center'},
        'Vladimir Cerrón': {'dx': -1.6, 'dy': 0.5, 'ha': 'right', 'va': 'center'},
        'A. López-Chau': {'dx': 0.0, 'dy': -3.2, 'ha': 'center', 'va': 'top'},
        'Carlos Álvarez': {'dx': 0.0, 'dy': 1.6, 'ha': 'center', 'va': 'bottom'},
        'Jorge Nieto': {'dx': 0.0, 'dy': 1.6, 'ha': 'center', 'va': 'bottom'},
        'Keiko Fujimori': {'dx': 0.0, 'dy': 1.6, 'ha': 'center', 'va': 'bottom'},
        'Roberto Sánchez': {'dx': 0.0, 'dy': 1.6, 'ha': 'center', 'va': 'bottom'}
    }

    for idx, row in df_metrics.iterrows():
        cand = row['Candidato']
        color = PALETA_CANDIDATOS.get(cand, '#7f8c8d')
        # Tamaño proporcional a las contradicciones (mínimo 100 para visibilidad)
        size = max(100, row['Contradiccion'] * 35)
        
        ax7.scatter(
            x=row['Adherencia'], 
            y=row['Innovacion'], 
            s=size, 
            color=color, 
            alpha=0.8, 
            edgecolors='black', 
            linewidth=1.2,
            label=f"{cand} (Contra.: {row['Contradiccion']:.1f}%)"
        )
        
        # Etiqueta de texto para cada candidato
        cfg = CONFIG_ETIQUETAS.get(cand, {'dx': 0.0, 'dy': 1.5, 'ha': 'center', 'va': 'bottom'})
        ax7.text(
            x=row['Adherencia'] + cfg['dx'], 
            y=row['Innovacion'] + cfg['dy'], 
            s=cand, 
            fontsize=10, 
            fontweight='bold', 
            ha=cfg['ha'],
            va=cfg['va']
        )
        
    # Dibujar líneas medias (promedio de los ejes)
    mean_x = df_metrics['Adherencia'].mean()
    mean_y = df_metrics['Innovacion'].mean()
    
    ax7.axvline(mean_x, color='#7f8c8d', linestyle='--', linewidth=1)
    ax7.axhline(mean_y, color='#7f8c8d', linestyle='--', linewidth=1)
    
    # Límites fijos para asegurar legibilidad consistente y evitar colisiones
    ax7.set_xlim(10, 60)
    ax7.set_ylim(25, 83)
    
    # Anotaciones de los Cuadrantes
    xlim = ax7.get_xlim()
    ylim = ax7.get_ylim()
    
    # Esquinas del plano para colocar etiquetas de cuadrante
    ax7.text(xlim[1] - 1, ylim[1] - 2, 'Mixed Discourse\n(Partial Alignment with New Proposals)', fontsize=10, color='#27ae60', fontweight='bold', ha='right', va='top')
    ax7.text(xlim[1] - 1, ylim[0] + 1, 'Strict Fidelity\n(Adhering to the Written Plan)', fontsize=10, color='#2980b9', fontweight='bold', ha='right', va='bottom')
    ax7.text(xlim[0] + 1, ylim[1] - 2, 'Emerging Agenda\n(Discourse Outside the Plan)', fontsize=10, color='#d35400', fontweight='bold', ha='left', va='top')
    ax7.text(xlim[0] + 1, ylim[0] + 1, 'Programmatic Dissonance\n(Inconsistency and Contradiction)', fontsize=10, color='#c0392b', fontweight='bold', ha='left', va='bottom')
    
    # Estilos del gráfico
    ax7.set_xlabel('Plan Adherence Index (% Equivalence)', fontsize=12, labelpad=10)
    ax7.set_ylabel('Agenda Innovation Index (% Invention)', fontsize=12, labelpad=10)
    ax7.grid(True, linestyle=':', alpha=0.6)
    
    # Leyenda para explicar el tamaño de los círculos
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Bubble size represents % of Contradictions',
                   markerfacecolor='#7f8c8d', markersize=10, markeredgecolor='black')
    ]
    ax7.legend(handles=legend_elements, bbox_to_anchor=(0.0, 1.01), loc='lower left', frameon=True)
    plt.tight_layout()
    
    path_g7 = os.path.join(dir_salida, 'reporte_7_navegacion_discursiva.png')
    fig7.savefig(path_g7, dpi=300, bbox_inches='tight')
    print(f"   -> Gráfico 7 guardado en: {path_g7}")
    plt.close(fig7)


if __name__ == "__main__":
    generar_reporte_graficos_reducido(
        ruta_csv="Resultados/Matriz_Final.csv",
        dir_salida="graficos"
    )
