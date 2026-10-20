import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generar_grafico_apilado_niveles(ruta_csv: str = r"Resultados\Matriz.csv", output_dir: str = "graficos"):
    """
    Lee la matriz de resultados y genera un grafico de barras apiladas 
    mostrando la distribucion porcentual y absoluta de las propuestas 
    clasificadas en 3 niveles de ajuste (Bajo, Medio, Alto) recalculados 
    dinamicamente para cada candidato sobre sus matches principales (Rank 1).
    Usa una escala de color morado premium (claro a oscuro), sin titulo y con leyenda simplificada.
    """
    print(f"Cargando datos desde '{ruta_csv}'...")
    if not os.path.exists(ruta_csv):
        print(f"Error: No se encontro el archivo '{ruta_csv}'.")
        return

    df_full = pd.read_csv(ruta_csv)
    
    # 1. Filtrar solo por el match principal (Rank 1) para representar la propuesta real
    if 'Rank_Match' in df_full.columns:
        df = df_full[df_full['Rank_Match'] == 1].copy()
        print("Filtrado por Rank_Match == 1 para matches de primer nivel.")
    else:
        df = df_full.copy()
        print("Columna 'Rank_Match' no encontrada. Utilizando todas las filas del dataset.")

    if df.empty:
        print("Error: El dataset filtrado esta vacio.")
        return

    # 2. Mapeo Academico de Nombres de Candidatos
    MAPEO_CANDIDATOS = {
        '1_keiko': 'Keiko Fujimori',
        '2_rla': 'R. López Aliaga',
        '3_carlosal': 'Carlos Álvarez',
        '4_chau': 'A. López-Chau',
        '5_jorgeni': 'Jorge Nieto',
        '6_robertosa': 'Roberto Sánchez',
        '7_cesara': 'César Acuña',
        '8_cerron': 'Vladimir Cerrón'
    }

    def formatear_candidato(cand_val):
        cand_str = str(cand_val).strip()
        if cand_str in MAPEO_CANDIDATOS:
            return MAPEO_CANDIDATOS[cand_str]
        
        # Fallback de limpieza basica
        cand_str = cand_str.split('_')[-1]
        return cand_str.capitalize()

    # Si existe la columna 'Candidato', la usamos. Si no, usamos 'Archivo_Entrevista_Origen'
    if 'Candidato' in df.columns:
        df['Candidato_Label'] = df['Candidato'].apply(formatear_candidato)
    elif 'Archivo_Entrevista_Origen' in df.columns:
        def extraer_cid_cand(filename):
            partes = str(filename).split('_')
            if len(partes) >= 3:
                return f"{partes[1]}_{partes[2]}"
            return filename
        df['Candidato_Label'] = df['Archivo_Entrevista_Origen'].apply(extraer_cid_cand).apply(formatear_candidato)
    else:
        df['Candidato_Label'] = "Candidato Desconocido"

    # 3. Clasificacion Dinamica Basada en Cuantiles Estadisticos (Percentiles 33 y 66)
    score_col = 'Score_Principal' if 'Score_Principal' in df.columns else ('Score_Semantico' if 'Score_Semantico' in df.columns else None)
    
    if score_col is None:
        print("Error: No se encontro ninguna columna de score en el archivo.")
        return

    scores_validos = df[score_col].dropna()
    p33 = scores_validos.quantile(0.33)
    p66 = scores_validos.quantile(0.66)
    
    print(f"Límites calculados dinámicamente:")
    print(f"  - Nivel Bajo: Score < {p33:.6f}")
    print(f"  - Nivel Medio: {p33:.6f} <= Score < {p66:.6f}")
    print(f"  - Nivel Alto: Score >= {p66:.6f}")

    def clasificar_dinamica(score):
        if pd.isna(score):
            return 'Low'
        if score < p33:
            return 'Low'
        elif score < p66:
            return 'Medium'
        else:
            return 'High'

    df['Nivel_Ajuste'] = df[score_col].apply(clasificar_dinamica)

    # 4. Tabulacion Cruzada por Candidato
    tabla = pd.crosstab(df['Candidato_Label'], df['Nivel_Ajuste'])
    
    for col in ['Low', 'Medium', 'High']:
        if col not in tabla.columns:
            tabla[col] = 0

    tabla = tabla[['Low', 'Medium', 'High']]
    tabla['Total'] = tabla['Low'] + tabla['Medium'] + tabla['High']
    tabla = tabla.sort_values('Total', ascending=False)

    os.makedirs(output_dir, exist_ok=True)

    tabla_csv_path = os.path.join(output_dir, 'proporcion_ajuste_tabular.csv')
    tabla.to_csv(tabla_csv_path, encoding='utf-8-sig')
    print(f"Tabla de proporciones exportada a '{tabla_csv_path}'.")

    # 5. Configuración Visual del Grafico Apilado
    tabla_plot = tabla[['Low', 'Medium', 'High']]
    
    # Paleta monocromática de morados/violetas premium:
    # Low (Malo): Morado muy oscuro (#311b92)
    # Medium: Morado intermedio (#7e57c2)
    # High (Bueno): Lavanda claro (#d1c4e9)
    colores = ['#311b92', '#7e57c2', '#d1c4e9']
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Graficar barras apiladas con mayor ancho (width=0.8) y menor espacio
    tabla_plot.plot(kind='bar', stacked=True, color=colores, edgecolor='white', linewidth=0.8, alpha=0.9, ax=ax, width=0.8)

    # 6. Insercion de Etiquetas Internas (Valores absolutos y porcentajes)
    for c_idx, (idx, row) in enumerate(tabla_plot.iterrows()):
        total = tabla.loc[idx, 'Total']
        val_bajo = row['Low']
        val_medio = row['Medium']
        val_alto = row['High']
        
        # Posiciones acumulativas en el eje Y
        y_bajo = val_bajo / 2
        y_medio = val_bajo + (val_medio / 2)
        y_alto = val_bajo + val_medio + (val_alto / 2)
        
        # Etiqueta Low (Fondo Oscuro -> Texto Blanco)
        if val_bajo > 0:
            pct_bajo = (val_bajo / total) * 100
            ax.text(c_idx, y_bajo, f"{int(val_bajo)}\n({pct_bajo:.1f}%)", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8)
            
        # Etiqueta Medium (Fondo Medio -> Texto Blanco)
        if val_medio > 0:
            pct_medio = (val_medio / total) * 100
            ax.text(c_idx, y_medio, f"{int(val_medio)}\n({pct_medio:.1f}%)", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8)
            
        # Etiqueta High (Fondo Claro -> Texto Oscuro para contraste óptimo)
        if val_alto > 0:
            pct_alto = (val_alto / total) * 100
            ax.text(c_idx, y_alto, f"{int(val_alto)}\n({pct_alto:.1f}%)", 
                    ha='center', va='center', color='#2c3e50', fontweight='bold', fontsize=8)

    # 7. Estilizado Academico
    # (Se ha quitado la llamada a set_title para remover el titulo del grafico)
    
    ax.set_xlabel('Presidential Candidate', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Interview Proposals', fontsize=12, fontweight='bold', labelpad=10)
    
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Remover bordes superior y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Anadir lineas de guia horizontales tenues
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='grey')
    ax.set_axisbelow(True)

    # Leyenda simplificada: Solo "Low", "Medium", "High"
    ax.legend(labels=["Low", "Medium", "High"], 
              bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, frameon=True, facecolor='whitesmoke')
    
    plt.tight_layout()

    # Guardado de la imagen
    nombre_imagen = os.path.join(output_dir, "grafico_niveles_ajuste.png")
    plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')
    print(f"Grafico exportado exitosamente a '{nombre_imagen}'.")
    
    plt.close()

if __name__ == "__main__":
    generar_grafico_apilado_niveles()
