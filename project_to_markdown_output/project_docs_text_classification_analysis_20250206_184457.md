Project Documentation Summary
=========================
Root Directory: C:\Users\ronal\APPs\text_classification_analysis
Files Analyzed: 6
Total Size: 12.23 KB
Timestamp: 2025-02-06 18:44:57

Skipped Files Sample:
-------------------
- file too large (2299.4KB): data\mind_news_classification.csv
- unknown type (4 files):
  • data\mind_news_classification_100.csv
  • outputs\sentiment_analysis_20250206_183722.csv
  • outputs\sentiment_analysis_20250206_184015.csv
  • outputs\sentiment_analysis_20250206_184337.csv
- file too large (22578.5KB): data\news.tsv
- binary file: sentiment_analysis.log

Directory Structure
==================

text_classification_analysis
├── app.py
├── data/
│   ├── mind_news_classification.csv
│   ├── mind_news_classification_100.csv
│   └── news.tsv
├── download_data.py
├── outputs/
│   ├── sentiment_analysis_20250206_183722.csv
│   ├── sentiment_analysis_20250206_184015.csv
│   └── sentiment_analysis_20250206_184337.csv
├── readme.md
├── requirements.txt
├── sample_data.py
├── sentiment_analysis.log
└── sentiment_analyzer.py

File Contents
=============


================================================================================
File: app.py
================================================================================

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from sentiment_analyzer import SentimentAnalyzer
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="📊",
    layout="wide"
)

def create_output_dir():
    """Crea el directorio de outputs si no existe."""
    os.makedirs("outputs", exist_ok=True)

def generate_output_filename():
    """Genera un nombre de archivo único basado en la fecha y hora."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/sentiment_analysis_{timestamp}.csv"

def main():
    st.title("📊 Analizador de Sentimientos para Títulos")
    st.write("Carga un archivo CSV y analiza el sentimiento de los títulos usando modelos de lenguaje.")

    # Crear directorio de outputs
    create_output_dir()

    # Subida de archivo
    uploaded_file = st.file_uploader("Elige un archivo CSV", type=['csv'])

    if uploaded_file is not None:
        # Leer el CSV
        df = pd.read_csv(uploaded_file)
        total_rows = len(df)
        
        st.write(f"📄 Archivo cargado: {uploaded_file.name}")
        st.write(f"📊 Total de títulos: {total_rows:,}")

        # Opciones de análisis
        st.subheader("Configuración del Análisis")
        analysis_type = st.radio(
            "Selecciona el tipo de análisis:",
            ["Primeros 10 títulos", "Primeros 100 títulos", "Dataset completo"]
        )

        # Mapear selección a límite
        limit_map = {
            "Primeros 10 títulos": 10,
            "Primeros 100 títulos": 100,
            "Dataset completo": None
        }
        limit = limit_map[analysis_type]

        # Tamaño del batch
        batch_size = st.slider("Tamaño del batch", min_value=10, max_value=100, value=50, step=10)

        if st.button("🚀 Iniciar Análisis", type="primary"):
            # Inicializar el analizador
            analyzer = SentimentAnalyzer(batch_size=batch_size)
            
            # Contenedores para la visualización en tiempo real
            progress_container = st.empty()
            metrics_container = st.container()
            
            # Variables para el seguimiento
            processed = 0
            results = []
            start_time = datetime.now()

            # Procesar títulos
            total_to_process = limit if limit else total_rows
            progress_bar = st.progress(0)
            
            # Columnas para métricas en tiempo real
            col1, col2, col3 = metrics_container.columns(3)
            positive_metric = col1.metric("Positivos", 0)
            neutral_metric = col2.metric("Neutros", 0)
            negative_metric = col3.metric("Negativos", 0)

            # Procesar por lotes
            df_to_process = df.head(limit) if limit else df
            
            for start_idx in range(0, len(df_to_process), batch_size):
                end_idx = min(start_idx + batch_size, len(df_to_process))
                batch = df_to_process.iloc[start_idx:end_idx]
                
                for _, row in batch.iterrows():
                    # Tomar el texto de la primera columna
                    text = row.iloc[0]
                    sentiment_code, sentiment = analyzer.analyze_sentiment(text)
                    
                    results.append({
                        'text': text,
                        'sentiment_code': sentiment_code,
                        'sentiment': sentiment
                    })
                    
                    processed += 1
                    
                    # Actualizar progreso y métricas
                    progress = processed / total_to_process
                    progress_bar.progress(progress)
                    
                    # Actualizar métricas
                    current_results = pd.DataFrame(results)
                    if not current_results.empty:
                        sentiment_counts = current_results['sentiment'].value_counts()
                        positive_metric.metric("Positivos", sentiment_counts.get('POSITIVE', 0))
                        neutral_metric.metric("Neutros", sentiment_counts.get('NEUTRAL', 0))
                        negative_metric.metric("Negativos", sentiment_counts.get('NEGATIVE', 0))
                    
                    # Actualizar información de progreso
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    rate = processed / elapsed_time if elapsed_time > 0 else 0
                    
                    progress_container.info(
                        f"⏳ Progreso: {processed:,}/{total_to_process:,} títulos "
                        f"({progress:.1%}) | "
                        f"Velocidad: {rate:.1f} títulos/segundo"
                    )

            # Crear DataFrame con resultados
            results_df = pd.DataFrame(results)
            
            # Guardar resultados
            output_file = generate_output_filename()
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            
            # Mostrar resultados finales
            st.success(f"✅ Análisis completado! Archivo guardado en: {output_file}")
            
            # Visualizaciones
            st.subheader("📊 Resultados del Análisis")
            
            # Gráfico de distribución de sentimientos
            fig_pie = px.pie(
                results_df, 
                names='sentiment', 
                title='Distribución de Sentimientos',
                color='sentiment',
                color_discrete_map={
                    'POSITIVE': '#28a745',
                    'NEUTRAL': '#17a2b8',
                    'NEGATIVE': '#dc3545'
                }
            )
            st.plotly_chart(fig_pie)
            
            # Mostrar nombre de la columna analizada
            st.info(f"📝 Columna analizada: {df.columns[0]}")
            
            # Mostrar datos en tabla
            st.subheader("📋 Muestra de Resultados")
            st.dataframe(
                results_df.style.set_properties(**{'text-align': 'left'})
            )

if __name__ == "__main__":
    main()



================================================================================
File: download_data.py
================================================================================

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



================================================================================
File: readme.md
================================================================================

# Guia

## Setup

Crear entorno virtual
```bash
python -m venv venv
```
Activar el entorno virtual
```bash
venv\Scripts\activate
```

Instalar dependencias
```bash
pip install -r requirements.txt
```

## Data

Los datos son de [Open-Data-Azure](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets) y permiten hacer la extraccion de data para probar algoritmos de ML.

Los datos se descargan con el script `download_data.py`



================================================================================
File: requirements.txt
================================================================================

numpy==2.2.2
pandas==2.2.3
python-dateutil==2.9.0.post0
requests==2.32.3
langchain-ollama==0.2.2
python-dotenv==1.0.0
streamlit==1.42.0
plotly==6.0.0



================================================================================
File: sample_data.py
================================================================================

import pandas as pd

# Leer el CSV
df = pd.read_csv('data/mind_news_classification.csv')

# Tomar los primeros 100 registros
df_100 = df.head(100)

# Opcional: Guardar los 100 registros en un nuevo CSV
df_100.to_csv('data/mind_news_classification_100.csv', index=False)



================================================================================
File: sentiment_analyzer.py
================================================================================

from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    SENTIMENT_CODES = {
        'POSITIVE': 1,
        'NEUTRAL': 0,
        'NEGATIVE': -1
    }
    
    def __init__(self, batch_size=50):
        """Inicializa el analizador de sentimientos."""
        load_dotenv()
        
        self.llm = OllamaLLM(
            model=os.getenv('OLLAMA_MODEL', 'phi4'),
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            temperature=0.1
        )
        
        self.batch_size = batch_size
        self.prompt_template = PromptTemplate(
            input_variables=["title"],
            template="""Analyze the sentiment of this news title:
"{title}"

Classify it into one of these categories:
- POSITIVE: For positive, optimistic, or uplifting content
- NEUTRAL: For factual, balanced, or objective content
- NEGATIVE: For negative, critical, or concerning content

Respond with ONLY ONE WORD: POSITIVE, NEUTRAL, or NEGATIVE."""
        )
        
    def analyze_sentiment(self, title: str) -> tuple[int, str]:
        """Analiza el sentimiento de un título y retorna (código, sentimiento)."""
        try:
            prompt = self.prompt_template.format(title=title)
            response = self.llm.invoke(prompt).strip().upper()
            
            # Validar respuesta
            if response in self.SENTIMENT_CODES:
                logger.info(f"Título: '{title[:100]}...' → Sentimiento: {response}")
                return self.SENTIMENT_CODES[response], response
            
            # Por defecto, retornar NEUTRAL
            logger.warning(f"[RESPUESTA INVÁLIDA] '{response}' para título: '{title[:100]}...' → Asignando: NEUTRAL")
            return self.SENTIMENT_CODES['NEUTRAL'], 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Error analizando título '{title}': {str(e)}")
            return self.SENTIMENT_CODES['NEUTRAL'], 'NEUTRAL'

