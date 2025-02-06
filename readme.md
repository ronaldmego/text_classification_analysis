# Análisis de Sentimientos con LLM Local

Sistema de análisis de sentimientos para títulos de noticias utilizando modelos de lenguaje locales a través de Ollama.

## 🚀 Características

- Análisis de sentimientos en tiempo real usando LLM local (phi4 por defecto)
- Interface web interactiva con Streamlit
- Procesamiento por lotes configurable
- Visualización de resultados con gráficos interactivos
- Exportación automática de resultados a CSV
- Soporte para datasets grandes con procesamiento por batches

## 📋 Prerequisitos

- Python 3.8+
- [Ollama](https://ollama.ai/) instalado y corriendo localmente
- Modelo phi4 descargado en Ollama (u otro modelo compatible)

## 🛠️ Guía

### Setup

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

### Data

Los datos son de [Open-Data-Azure](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets) y permiten hacer la extraccion de data para probar algoritmos de ML.

Los datos se descargan con el script `download_data.py`

## 📊 Uso

1. Descargar datos de ejemplo
```bash
python download_data.py
```

2. Iniciar la aplicación
```bash
streamlit run app.py
```

3. Abrir en el navegador: `http://localhost:8501`

## 📁 Estructura del Proyecto

```
text_classification_analysis/
├── app.py                  # Aplicación principal Streamlit
├── sentiment_analyzer.py   # Clase para análisis de sentimientos
├── download_data.py        # Script para descargar dataset MIND
├── sample_data.py          # Utilidad para crear muestras de datos
├── data/                   # Directorio para datasets
├── outputs/               # Resultados del análisis
└── requirements.txt       # Dependencias del proyecto
```

## 🔧 Configuración

El sistema permite configurar:
- Tamaño de batch para procesamiento
- Cantidad de títulos a analizar
- Modelo de Ollama a utilizar
- URL base de Ollama

## 📈 Resultados

Los resultados se guardan en:
- CSV en la carpeta `outputs/` con timestamp
- Visualizaciones en tiempo real en la interface
- Logs detallados en `sentiment_analysis.log`