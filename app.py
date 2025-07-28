from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, DetectorFactory # Importa a função detect e DetectorFactory

# Garante que a detecção de idioma seja consistente
DetectorFactory.seed = 0

app = Flask(__name__)

# --- Configuração do NLTK para português ---
# Carregamos as stopwords para os idiomas mais comuns que vamos suportar.
# É importante que esses pacotes NLTK estejam baixados no ambiente do Render.
# (Lembre-se do comando 'python -c "import nltk; nltk.data.path.append(...); nltk.download('all-nltk', download_dir=...)"')

# Dicionário para armazenar as stopwords por idioma
STOPWORDS_BY_LANG = {}
SUPPORTED_LANGUAGES = ['portuguese', 'english', 'spanish']

for lang in SUPPORTED_LANGUAGES:
    try:
        STOPWORDS_BY_LANG[lang] = set(nltk.corpus.stopwords.words(lang))
    except LookupError:
        # Se as stopwords não estiverem baixadas, tenta baixar.
        # Em produção (Render), isso deve ser feito no build command.
        # Para evitar loops infinitos ou problemas de permissão, é melhor garantir
        # que 'all-nltk' seja baixado no build command do Render.
        # Esta parte é mais para desenvolvimento local.
        nltk.download('stopwords', quiet=True) # quiet=True para não imprimir no console
        STOPWORDS_BY_LANG[lang] = set(nltk.corpus.stopwords.words(lang))

# --- Rota Principal (Endpoint do Servidor Web) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Função auxiliar para analisar uma única URL ---
def analyze_single_url(url):
    """
    Analisa uma única URL, realiza web scraping e TF-IDF.
    Retorna (erro, keywords) ou (None, keywords_formatadas).
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status() # Lança exceção para erros HTTP (4xx ou 5xx)
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- REMOÇÃO DE BOILERPLATE APRIMORADA ---
        # Remove elementos HTML que geralmente não contêm texto relevante
        # ou que podem poluir a análise (scripts JavaScript, estilos CSS,
        # cabeçalhos, rodapés, menus de navegação, formulários, botões, iframes, etc.).
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button', 'input', 'select', 'textarea', 'iframe', 'noscript', 'img']):
            tag.decompose() # Remove o elemento e seu conteúdo
        # --- FIM DA REMOÇÃO DE BOILERPLATE APRIMORADA ---

        # Extrai todo o texto visível da página HTML.
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip() # Substitui múltiplos espaços por um único
        text = re.sub(r'[^\w\s]', '', text) # Remove caracteres que não são letras, números ou espaços

        # Processamento de Linguagem Natural (PLN)
        # Tenta detectar o idioma do texto limpo
        detected_lang_code = 'unknown'
        try:
            if len(text) > 50: # langdetect precisa de um mínimo de texto para funcionar bem
                detected_lang_code = detect(text)
            else:
                detected_lang_code = 'unknown' # Texto muito curto para detectar
        except Exception:
            detected_lang_code = 'unknown' # Falha na detecção

        # Mapeia o código do idioma detectado para o nome completo do NLTK
        nltk_lang_name = 'unknown'
        if detected_lang_code == 'pt':
            nltk_lang_name = 'portuguese'
        elif detected_lang_code == 'en':
            nltk_lang_name = 'english'
        elif detected_lang_code == 'es':
            nltk_lang_name = 'spanish'
        
        # Seleciona as stopwords com base no idioma detectado
        current_stopwords = STOPWORDS_BY_LANG.get(nltk_lang_name, set()) # Pega as stopwords ou um set vazio se não suportado

        # Usa o idioma detectado para tokenização, com 'english' como fallback
        words = nltk.word_tokenize(text.lower(), language=nltk_lang_name if nltk_lang_name != 'unknown' else 'english')
        filtered_words = [word for word in words if word.isalpha() and word not in current_stopwords and len(word) > 2]
        
        processed_text = " ".join(filtered_words)

        if not processed_text.strip():
            return {"error": "Conteúdo textual insuficiente para análise."}, None

        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words=list(current_stopwords), # Passa as stopwords do idioma detectado
            max_features=200,
            ngram_range=(1, 2) # Considera palavras únicas e pares de palavras
        )

        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        word_tfidf_scores = list(zip(feature_names, tfidf_scores))
        top_keywords = sorted(word_tfidf_scores, key=lambda x: x[1], reverse=True)[:20]

        if not top_keywords:
            return {"error": "Não foi possível extrair palavras-chave relevantes suficientes."}, None
        
        # Formata os resultados para ter a palavra e seu score (arredondado para 4 casas decimais)
        top_keywords_formatted = [(word, round(score, 4)) for word, score in top_keywords]
        
        # Retorna o idioma detectado junto com as palavras-chave
        return None, {"keywords": top_keywords_formatted, "detected_language": nltk_lang_name} 

    # --- Tratamento de Erros Específicos para analyze_single_url ---
    except requests.exceptions.Timeout:
        return {"error": "Tempo limite excedido ao acessar a URL. O site pode estar lento ou inacessível."}, None
    except requests.exceptions.ConnectionError:
        return {"error": "Não foi possível conectar à URL. Verifique sua conexão ou se o site está online."}, None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 404:
            return {"error": f"URL não encontrada (Erro {status_code}). Verifique se o endereço está correto."}, None
        elif status_code == 403:
            return {"error": f"Acesso negado (Erro {status_code}). O site pode estar bloqueando a análise."}, None
        elif status_code == 500:
            return {"error": f"Erro interno do servidor (Erro {status_code}) ao acessar a URL. Tente novamente mais tarde."}, None
        else:
            return {"error": f"Erro HTTP {status_code} ao acessar a URL. Detalhes: {e}"}, None
    except requests.exceptions.RequestException:
        return {"error": "Ocorreu um erro de rede inesperado ao acessar a URL. Verifique a URL e sua conexão."}, None
    except Exception as e:
        return {"error": f"Ocorreu um erro interno no servidor: {e}"}, None

# --- Rota de Análise Principal (Endpoint da API) ---
@app.route('/analyze', methods=['POST'])
def analyze():
    urls_to_analyze = request.json.get('urls') 
    
    if not urls_to_analyze or not isinstance(urls_to_analyze, list) or not all(isinstance(url, str) for url in urls_to_analyze):
        return jsonify({"error": "Lista de URLs não fornecida ou formato inválido."}), 400

    individual_results = []
    all_keywords_for_comparison = []

    for url_item in urls_to_analyze:
        error_message, analysis_data = analyze_single_url(url_item)
        if error_message:
            individual_results.append({"url": url_item, "status": "error", "message": error_message["error"]})
        else:
            individual_results.append({
                "url": url_item, 
                "status": "success", 
                "top_keywords": analysis_data["keywords"],
                "detected_language": analysis_data["detected_language"]
            })
            all_keywords_for_comparison.append(set(k[0] for k in analysis_data["keywords"])) 

    common_keywords = set()
    unique_keywords_per_url = []

    if all_keywords_for_comparison:
        if len(all_keywords_for_comparison) > 1:
            common_keywords = set.intersection(*all_keywords_for_comparison)
        elif len(all_keywords_for_comparison) == 1:
            common_keywords = all_keywords_for_comparison[0]

        for i, current_set in enumerate(all_keywords_for_comparison):
            unique_to_current = current_set.copy()
            for j, other_set in enumerate(all_keywords_for_comparison):
                if i != j:
                    unique_to_current.difference_update(other_set)
            unique_keywords_per_url.append({"url": urls_to_analyze[i], "keywords": list(unique_to_current)})
    
    return jsonify({
        "individual_results": individual_results,
        "comparison": {
            "common_keywords": list(common_keywords),
            "unique_keywords_per_url": unique_keywords_per_url
        }
    })

# --- Execução do Aplicativo Flask ---
if __name__ == '__main__':
    # Em produção no Render, o Gunicorn gerencia a execução do Flask.
    # O debug=True é útil apenas para desenvolvimento local.
    app.run(debug=True)
