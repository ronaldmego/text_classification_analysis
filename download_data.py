import os
import requests
import pandas as pd
from pathlib import Path
import zipfile
import io

def download_mind_dataset(save_path="./data"):
    """
    Descarga el dataset MIND desde el repositorio de Microsoft Recommenders.
    
    Args:
        save_path (str): Directorio donde guardar los datos
    Returns:
        pandas.DataFrame: DataFrame con títulos y categorías
    """
    # Crear directorio si no existe
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # URL del dataset
    url = "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip"
    
    print(f"Descargando dataset desde {url}...")
    try:
        # Descargar el archivo
        response = requests.get(url, timeout=30)
        
        response.raise_for_status()
        print("Descarga exitosa. Procesando archivo...")
        
        # Extraer y procesar el archivo zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            news_file = next(
                (save_path / file for file in zip_ref.namelist() if file.endswith('news.tsv')),
                None
            )
            if news_file:
                zip_ref.extract(news_file.name, save_path)
                print(f"Archivo extraído: {news_file}")
            else:
                raise ValueError("No se encontró el archivo news.tsv en el ZIP")
        
        # Leer y procesar el TSV
        print("Leyendo archivo TSV...")
        df = pd.read_csv(
            news_file, 
            sep='\t',
            header=None,
            names=['id', 'category', 'subcategory', 'title', 'abstract', 
                   'url', 'title_entities', 'abstract_entities']
        )
        
        # Crear y guardar DataFrame simplificado
        classification_df = df[['title', 'category', 'subcategory']]
        output_file = save_path / 'mind_news_classification.csv'
        classification_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\nDataset procesado exitosamente. Shape: {classification_df.shape}")
        print(f"Datos guardados en: {output_file}")
        
        return classification_df
        
    except Exception as e:
        print(f"Error durante la descarga o procesamiento: {e}")
        print("\nPor favor, verifica que:")
        print("1. Tienes conexión a internet")
        print("2. El firewall no está bloqueando la conexión")
        print("3. Tienes permisos de escritura en el directorio de destino")
        return None

def main():
    df = download_mind_dataset()
    
    if df is not None:
        try:
            print("\nDistribución de categorías:")
            print(df['category'].value_counts())
            print("\nEjemplos de títulos por categoría:")
            for category in df['category'].unique():
                print(f"\n{category.upper()}:")
                print(df[df['category'] == category]['title'].iloc[0])
        except Exception as e:
            print(f"Error al mostrar estadísticas: {e}")

if __name__ == "__main__":
    main()