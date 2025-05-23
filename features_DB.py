"""
AUTOR:          JOSE MIGUEL GONZALEZ ZARAGOZA
                631145-IRSI-8VO_SEMESTRE
CLIENTE:        ANTONIO MARTINEZ TORTEYA
INSTITUCION:    UNIVERSIDAD DE MONTERREY (UDEM)
TITULO:         'features_DB.py'
DESCRIPCION:    CODIGO DEDICADO A LA GENERACION DE UNA BASE DE 
                DATOS COMPLETA EN UN SOLO ARCHIVO
"""

import os
import argparse
import pandas as pd
import numpy as np
from warnings import warn


"""
Script para consolidar archivos features.csv de múltiples pacientes en una única base de datos.

Este script procesa archivos features.csv generados por análisis de radiómica, eliminando columnas
y filas específicas no deseadas, y consolidando los datos en un único archivo CSV con formato:
- Cada fila representa un paciente
- Cada columna representa una característica de una estructura anatómica específica
- Una columna final 'Target' con el valor especificado (0 o 1)

Uso:
    python consolidar_features.py /ruta/a/carpeta/maestra 1  # Para Target=1
    python consolidar_features.py /ruta/a/carpeta/maestra 0  # Para Target=0
"""

def validate_value(value):
    """Valida y normaliza un valor, manejando casos especiales."""
    if pd.isna(value) or value == '':
        return 'N/A'
    try:
        # Manejar números complejos
        if isinstance(value, (complex, np.complex128, np.complex64)):
            warn(f"Valor complejo encontrado: {value}")
            return 'N/A'
        # Convertir a float y verificar si es finito
        float_val = float(value)
        return float_val if np.isfinite(float_val) else 'N/A'
    except (TypeError, ValueError):
        return 'N/A'
    

def process_features_file(file_path, patient_name, include_headers=False):
    """
    Procesa un archivo features.csv individual para convertirlo en una fila de la base de datos final.

    Args:
        file_path (str): Ruta al archivo features.csv a procesar
        patient_name (str): Identificador del paciente
        include_headers (bool): Si es True, incluye los nombres de columnas en el resultado

    Returns:
        tuple: (column_names, row_values) donde:
            - column_names: Lista de nombres de columnas (solo si include_headers=True)
            - row_values: Lista de valores para la fila del paciente

    El proceso:
        1. Elimina columnas no deseadas predefinidas
        2. Elimina filas correspondientes a estructuras anatómicas no deseadas
        3. Transforma los datos de formato ancho a largo
        4. Genera nombres de columnas combinando estructura y variable
    """
    columns_to_remove = [
        'ColorId','diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum',
        'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis',
        'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation',
        'original_firstorder_Skewness', 'original_firstorder_Uniformity', 'original_firstorder_Variance',
        'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency',
        'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy',
        'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn',
        'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage',
        'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability',
        'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_gldm_GrayLevelVariance',
        'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformityNormalized',
        'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis',
        'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis',
        'original_glszm_LowGrayLevelZoneEmphasis', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity',
        'original_ngtdm_Contrast', 'original_ngtdm_Strength', 'original_firstorder_10Percentile', 'original_firstorder_90Percentile',
        'original_firstorder_Maximum', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum',
        'original_firstorder_RootMeanSquared', 'diagnostics_Mask-original_VoxelNum', 'original_gldm_GrayLevelNonUniformity',
        'original_glszm_GrayLevelNonUniformity', 'original_firstorder_TotalEnergy', 'original_gldm_LargeDependenceHighGrayLevelEmphasis',
        'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis',
        'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis',
        'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis',
        'original_shape_MeshVolume', 'RGB', 'diagnostics_Image-original_Hash', 'diagnostics_Image-original_Dimensionality',
        'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size', 'diagnostics_Mask-original_Hash',
        'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox',
        'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass'
    ]

    filas_a_eliminar = [
        'ctx-rh-parsorbitalis',
        'ctx-lh-rostralanteriorcingulate',
        'ctx-rh-medialorbitofrontal',
        'ctx-rh-superiorfrontal',
        'ctx-lh-temporalpole',
        'ctx-lh-medialorbitofrontal',
        'ctx-rh-parahippocampal',
        'ctx-lh-parahippocampal',
        'ctx-lh-frontalpole',
        'ctx-rh-lateralorbitofrontal',
        'ctx-rh-rostralmiddlefrontal',
        'ctx-rh-supramarginal',
        'ctx-lh-parsorbitalis',
        'ctx-rh-precuneus',
        'ctx-lh-entorhinal',
        'ctx-rh-caudalmiddlefrontal',
        'ctx-rh-superiorparietal',
        'ctx-rh-frontalpole',
        'Left-Pallidum',
        'Optic-Chiasm',
        'Right-Pallidum',
        'CC_Central'
    ]

    try:
        # Leer el archivo con manejo de errores
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            warn(f"Error leyendo {file_path}: {str(e)}")
            return None, None
        
        # Verificar columnas mínimas requeridas
        if 'StructName' not in df.columns:
            warn(f"Archivo {file_path} no tiene columna StructName")
            return None, None
            
        # Eliminar columnas no deseadas (tu lógica original)
        df = df.drop(columns=columns_to_remove, errors='ignore')
        df = df.drop(columns=['Color ID'], errors='ignore')
        
        # Eliminar filas no deseadas (tu lógica original)
        df = df[~df['StructName'].isin(filas_a_eliminar)]
        
        # Transformar a formato largo con validación de valores
        pivoted_values = df.set_index('StructName').stack().reset_index()
        pivoted_values.columns = ['StructName', 'Variable', 'Valor']
        pivoted_values['Columna'] = pivoted_values['StructName'] + '_' + pivoted_values['Variable']
        
        # Validar todos los valores
        pivoted_values['Valor'] = pivoted_values['Valor'].apply(validate_value)
        
        # Reportar valores inválidos
        invalid_values = pivoted_values[pivoted_values['Valor'] == 'N/A']
        for _, row in invalid_values.iterrows():
            if str(row['Valor']) != 'N/A':  # Solo reportar valores que no eran N/A originalmente
                warn(f"Valor inválido en {patient_name}, estructura {row['StructName']}, variable {row['Variable']}: {row['Valor']}")
        
        column_names = list(pivoted_values['Columna'])
        row_values = [patient_name] + list(pivoted_values['Valor'])
        
        if include_headers:
            return column_names, row_values
        else:
            return None, row_values
            
    except Exception as e:
        warn(f"Error grave procesando {file_path}: {str(e)}")
        return None, None
    


def main():
    """Función principal con las mejoras de consistencia en columnas."""
    parser = argparse.ArgumentParser(description="Generar base de datos consolidada desde archivos features.csv.")
    parser.add_argument("ruta_maestra", type=str, help="Ruta de la carpeta maestra que contiene los subdirectorios de pacientes.")
    parser.add_argument("target_value", type=int, choices=[0, 1], help="Valor para la columna Target (0 o 1)")
    args = parser.parse_args()
    
    output_csv = os.path.join(args.ruta_maestra, "features_consolidado.csv")
    all_data = []
    all_columns = set()
    
    print(f"Explorando la carpeta maestra: {args.ruta_maestra}")
    
    # Primera pasada: recolectar todas las columnas posibles
    print("\nFase 1: Recolectando estructura de columnas...")
    for subdir in os.listdir(args.ruta_maestra):
        subdir_path = os.path.join(args.ruta_maestra, subdir)
        stats_path = os.path.join(subdir_path, "stats")
        features_path = os.path.join(stats_path, "features.csv")
        
        if os.path.isdir(subdir_path) and os.path.isdir(stats_path) and os.path.exists(features_path):
            columns, _ = process_features_file(features_path, subdir, include_headers=True)
            if columns:
                all_columns.update(columns)
    
    if not all_columns:
        print("No se encontraron archivos válidos features.csv")
        return
    
    # Segunda pasada: procesar todos los archivos con estructura consistente
    print("\nFase 2: Procesando archivos con estructura consistente...")
    for subdir in os.listdir(args.ruta_maestra):
        subdir_path = os.path.join(args.ruta_maestra, subdir)
        stats_path = os.path.join(subdir_path, "stats")
        features_path = os.path.join(stats_path, "features.csv")
        
        if os.path.isdir(subdir_path) and os.path.isdir(stats_path) and os.path.exists(features_path):
            print(f"Procesando: {features_path}")
            _, row = process_features_file(features_path, subdir, include_headers=False)
            if row:
                # Crear diccionario con todas las columnas posibles
                patient_data = {'Paciente': row[0]}
                patient_data.update(dict(zip(all_columns, ['N/A']*len(all_columns))))
                
                # Llenar con los valores disponibles (row[1:] porque row[0] es el nombre)
                available_data = dict(zip(_, row[1:])) if _ else {}
                for col, val in available_data.items():
                    if col in patient_data:  # Solo si es una columna conocida
                        patient_data[col] = val
                
                patient_data['Target'] = args.target_value
                all_data.append(patient_data)
    
    if not all_data:
        print("No se encontraron datos válidos para consolidar")
        return
    
    # Crear DataFrame final
    df_final = pd.DataFrame(all_data)
    
    # Ordenar columnas
    column_order = ['Paciente'] + sorted(list(all_columns)) + ['Target']
    df_final = df_final[column_order]
    
    # Guardar resultados
    df_final.to_csv(output_csv, index=False)
    print(f"\nBase de datos consolidada guardada en: {output_csv}")
    print(f"Total pacientes: {len(df_final)}")
    print(f"Total variables: {len(df_final.columns)-2}")  # Restamos Paciente y Target

if __name__ == "__main__":
    main()