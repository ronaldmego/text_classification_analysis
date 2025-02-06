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