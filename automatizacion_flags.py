"""
AUTOR:          JOSE MIGUEL GONZALEZ ZARAGOZA
                631145-IRSI-8VO_SEMESTRE
CLIENTE:        ANTONIO MARTINEZ TORTEYA
INSTITUCION:    UNIVERSIDAD DE MONTERREY (UDEM)
TITULO:         'automatizacion_flags.py'
DESCRIPCION:    CODIGO DEDICADO A LA EXTRCACCION DE CARACTERISTICAS DE IMAGENES 
                CEREBRALES PARA VARIOS PACIENTES DENTRO DE UNA SOLA CARPETA
                CON EL FIN DE AUTOMATIZAR LA EXTRACCION DE BASES DE DATOS
"""


import os
import argparse
import warnings
import radiomics
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from nilearn import image
from helperFuns import presentSegments

def process_folder(folder_path):
    """Procesa los archivos de MRI de una carpeta de paciente y extrae características radiómicas.

    Args:
        folder_path (str): Ruta completa a la carpeta del paciente que contiene los archivos MRI.

    Returns:
        pd.DataFrame or None: DataFrame con las características radiómicas extraídas para cada segmento.
            Contiene columnas: 'ColorId', 'StructName', 'RGB' y características radiómicas.
            Retorna None si no se encuentran los archivos necesarios o ocurre un error.

    Raises:
        FileNotFoundError: Si no se encuentran los archivos 'aparc+aseg.mgz' o 'brain.mgz'.
        RuntimeError: Si ocurre un error durante el procesamiento de las imágenes.
    """


    print(f"Processing folder: {os.path.basename(folder_path)}")
    
    aseg_path = os.path.join(folder_path, 'mri', 'aparc+aseg.mgz')
    brain_path = os.path.join(folder_path, 'mri', 'brain.mgz')
    
    if not os.path.exists(aseg_path) or not os.path.exists(brain_path):
        print(f"Skipping {folder_path}: Required files not found.")
        return None
    
    try:
        brain = nib.load(brain_path)
        aseg = nib.load(aseg_path)
        img_data = aseg.get_fdata()
        df = presentSegments(img_data)
        
        # Asegurar copia explícita como en features.py original
        brain_sitk = sitk.GetImageFromArray(np.array(aseg.dataobj, dtype=np.int32, copy=True))
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        results = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, row in df.iterrows():
                if i > 0 and row['NVoxels'] > 1:
                    color_id, name, color = row[['Id', 'StructName', 'Color Array']]
                    mask = (img_data == color_id)
                    
                    # Mostrar progreso como en el original (cada ~5%)
                    if i / len(df) * 100 % 5 < 1:
                        print(f"Progress: {i / len(df) * 100:.0f}%")
                    
                    masked_img_data = np.where(mask, color_id, 0)
                    masked_img = image.new_img_like(aseg, masked_img_data.astype(np.int32))
                    segmentation_sitk = sitk.GetImageFromArray(np.array(masked_img.dataobj, dtype=np.int32))
                    
                    # Extraer características y filtrar como en el original
                    features = extractor.execute(brain_sitk, segmentation_sitk, color_id)
                    filtered_features = {
                        feature_name: features[feature_name] 
                        for feature_name in features.keys() 
                        if 'original' in feature_name
                    }
                    
                    results.append({
                        'ColorId': color_id, 
                        'StructName': name, 
                        'RGB': color, 
                        **filtered_features
                    })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"Error processing {folder_path}: {str(e)}")
        return None

def main():
    """Función principal que automatiza el procesamiento de múltiples carpetas con datos de MRI.

    Procesa recursivamente todas las subcarpetas dentro del directorio maestro especificado,
    extrayendo características radiómicas para cada paciente y guardando los resultados en archivos CSV.

    El sistema lleva un registro de las carpetas ya procesadas en 'processed_folders.csv' para evitar
    reprocesamiento.

    Args:
        No recibe argumentos directamente, pero espera un argumento de línea de comandos:
        master_folder (str): Ruta al directorio maestro que contiene las carpetas de pacientes.

    Returns:
        None

    Ejemplo de uso:
        python automatizacion_flags.py /ruta/a/carpeta_maestro

    Notas:
        - Crea una subcarpeta 'stats' en cada carpeta de paciente con los resultados.
        - Los archivos requeridos en cada carpeta son: 'mri/aparc+aseg.mgz' y 'mri/brain.mgz'.
    """


    parser = argparse.ArgumentParser(description='Extract features from MRI files in multiple folders.')
    parser.add_argument('master_folder', type=str, help='Path to the master folder containing subfolders with MRI data.')
    args = parser.parse_args()
    
    master_folder = args.master_folder
    processed_csv = os.path.join(master_folder, 'processed_folders.csv')
    
    if not os.path.isdir(master_folder):
        print(f"Error: {master_folder} is not a valid directory.")
        return
    
    processed_folders = set()
    if os.path.exists(processed_csv):
        processed_df = pd.read_csv(processed_csv)
        processed_folders = set(processed_df['Folder'].tolist())
    
    subfolders = [os.path.join(master_folder, d) for d in os.listdir(master_folder) if os.path.isdir(os.path.join(master_folder, d))]
    new_processed = []
    
    for folder in subfolders:
        if folder in processed_folders:
            print(f"Skipping {folder}: Already processed.")
            continue
        
        results_df = process_folder(folder)
        if results_df is not None:
            stats_path = os.path.join(folder, 'stats')
            os.makedirs(stats_path, exist_ok=True)
            results_df.to_csv(os.path.join(stats_path, 'features.csv'), index=False)
            print(f"Saved features for {folder}")
            new_processed.append(folder)
    
    if new_processed:
        updated_df = pd.DataFrame({'Folder': new_processed})
        if os.path.exists(processed_csv):
            updated_df.to_csv(processed_csv, mode='a', header=False, index=False)
        else:
            updated_df.to_csv(processed_csv, index=False)
    
if __name__ == "__main__":
    main()
