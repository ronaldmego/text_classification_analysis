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
    st.title("📊 Local LLM - Analizador de Sentimientos")
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

        if st.button("🚀 Iniciar Análisis", type="primary"):
            # Inicializar el analizador
            analyzer = SentimentAnalyzer()
            
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

            # Procesar títulos
            df_to_process = df.head(limit) if limit else df
            
            for _, row in df_to_process.iterrows():
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