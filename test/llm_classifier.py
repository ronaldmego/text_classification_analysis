from dotenv import load_dotenv
import os
import pandas as pd
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
    
    def process_titles(self, csv_path: str, limit=None):
        """Procesa los títulos del CSV y genera análisis de sentimientos."""
        start_time = datetime.now()
        try:
            # Leer el CSV
            df = pd.read_csv(csv_path)
            total_available = len(df)
            
            if limit:
                df = df.head(limit)
                logger.info(f"Limitando análisis a {limit} títulos de {total_available} disponibles")
            
            total_titles = len(df)
            logger.info(f"Iniciando análisis de {total_titles:,} títulos")
            
            results = []
            last_progress = 0

            for start_idx in range(0, total_titles, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_titles)
                batch = df.iloc[start_idx:end_idx]
                
                for _, row in batch.iterrows():
                    sentiment_code, sentiment = self.analyze_sentiment(row['title'])
                    
                    results.append({
                        'title': row['title'],
                        'category': row['category'],
                        'subcategory': row['subcategory'],
                        'sentiment_code': sentiment_code,
                        'sentiment': sentiment
                    })
                
                # Mostrar progreso
                current_progress = (end_idx / total_titles) * 100
                if current_progress - last_progress >= 5 or end_idx == total_titles:
                    elapsed_time = datetime.now() - start_time
                    rate = end_idx / elapsed_time.total_seconds()
                    estimated_total_time = total_titles / rate
                    remaining_time = estimated_total_time - elapsed_time.total_seconds()
                    
                    logger.info(
                        f"Progreso: {current_progress:.1f}% "
                        f"({end_idx:,}/{total_titles:,} títulos) "
                        f"Velocidad: {rate:.1f} títulos/segundo "
                        f"Tiempo restante: {remaining_time/60:.1f} minutos"
                    )
                    last_progress = current_progress
            
            # Crear DataFrame con resultados
            results_df = pd.DataFrame(results)
            
            # Guardar resultados
            output_file = 'sentiment_analysis_results.csv'
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"\nResultados guardados en: {output_file}")
            
            # Estadísticas finales
            sentiment_stats = results_df['sentiment'].value_counts()
            sentiment_percentages = (sentiment_stats / len(results_df) * 100).round(2)
            
            logger.info("\nEstadísticas finales:")
            logger.info(f"Tiempo total de ejecución: {datetime.now() - start_time}")
            logger.info(f"Total títulos procesados: {total_titles:,}")
            logger.info("\nDistribución de sentimientos:")
            for sentiment in sentiment_stats.index:
                logger.info(f"{sentiment}: {sentiment_stats[sentiment]:,} títulos ({sentiment_percentages[sentiment]}%)")
            
            # Análisis por categoría
            logger.info("\nDistribución por categoría:")
            category_sentiment = pd.crosstab(
                results_df['category'], 
                results_df['sentiment'], 
                normalize='index'
            ) * 100
            logger.info(category_sentiment.round(2))
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error en el proceso: {str(e)}")
            raise

def main():
    try:
        import argparse
        
        # Configurar el parser de argumentos
        parser = argparse.ArgumentParser(description='Análisis de sentimientos para títulos de noticias')
        parser.add_argument('--limit', type=int, help='Número de títulos a procesar (opcional)')
        parser.add_argument('--batch', type=int, default=50, help='Tamaño del batch (default: 50)')
        parser.add_argument('--csv', type=str, default='data/mind_news_classification.csv', 
                          help='Ruta al archivo CSV (default: data/mind_news_classification.csv)')
        
        args = parser.parse_args()
        
        analyzer = SentimentAnalyzer(batch_size=args.batch)
        logger.info(f"Iniciando análisis con {'todos los títulos' if args.limit is None else f'límite de {args.limit} títulos'}")
        
        analyzer.process_titles(args.csv, limit=args.limit)
        
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {str(e)}")

if __name__ == "__main__":
    main()