from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# --- Configuração do NLTK para português ---
stop_words_pt = set(nltk.corpus.stopwords.words('portuguese'))

# --- Rota Principal (Endpoint do Servidor Web) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Função auxiliar para analisar uma única URL ---
def analyze_single_url(url, stop_words_pt):
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

        # Remove elementos HTML que geralmente não contêm texto relevante
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose()

        # Extrai todo o texto visível da página HTML.
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip() # Substitui múltiplos espaços por um único
        text = re.sub(r'[^\w\s]', '', text) # Remove caracteres que não são letras, números ou espaços

        # Processamento de Linguagem Natural (PLN)
        words = nltk.word_tokenize(text.lower(), language='portuguese')
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words_pt and len(word) > 2]
        processed_text = " ".join(filtered_words)

        # Se não houver texto processado, retorna erro
        if not processed_text.strip():
            return {"error": "Conteúdo textual insuficiente para análise."}, None

        # Inicializa o TfidfVectorizer
        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words=list(stop_words_pt), # Converte para lista
            max_features=200,
            ngram_range=(1, 2) # Considera palavras únicas e pares de palavras
        )

        # Ajusta o vetorizador ao texto e o transforma em uma matriz TF-IDF
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Cria lista de tuplas (palavra, score TF-IDF)
        word_tfidf_scores = list(zip(feature_names, tfidf_scores))
        top_keywords = sorted(word_tfidf_scores, key=lambda x: x[1], reverse=True)[:20] # Pega as 20 principais

        # Se não houver palavras-chave relevantes, retorna erro
        if not top_keywords:
            return {"error": "Não foi possível extrair palavras-chave relevantes suficientes."}, None
        
        # Formata os resultados para ter a palavra e seu score (arredondado para 4 casas decimais)
        top_keywords_formatted = [(word, round(score, 4)) for word, score in top_keywords]
        return None, top_keywords_formatted # Retorna (None, keywords) em caso de sucesso

    # --- Tratamento de Erros Específicos para analyze_single_url ---
    except requests.exceptions.Timeout:
        return {"error": "Tempo limite excedido ao acessar a URL."}, None
    except requests.exceptions.ConnectionError:
        return {"error": "Não foi possível conectar à URL. Verifique se o site está online."}, None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 404:
            return {"error": f"URL não encontrada (Erro {status_code})."}, None
        elif status_code == 403:
            return {"error": f"Acesso negado (Erro {status_code}). O site pode estar bloqueando a análise."}, None
        elif status_code == 500:
            return {"error": f"Erro interno do servidor (Erro {status_code}) ao acessar a URL."}, None
        else:
            return {"error": f"Erro HTTP {status_code} ao acessar a URL."}, None
    except requests.exceptions.RequestException:
        return {"error": "Ocorreu um erro de rede inesperado ao acessar a URL."}, None
    except Exception as e:
        return {"error": f"Ocorreu um erro interno ao analisar a URL: {e}."}, None

# --- Rota de Análise Principal (Endpoint da API) ---
@app.route('/analyze', methods=['POST'])
def analyze():
    # Agora espera uma LISTA de URLs do frontend
    urls_to_analyze = request.json.get('urls') 
    
    # Validação inicial da entrada
    if not urls_to_analyze or not isinstance(urls_to_analyze, list) or not all(isinstance(url, str) for url in urls_to_analyze):
        return jsonify({"error": "Lista de URLs não fornecida ou formato inválido."}), 400

    individual_results = [] # Para armazenar os resultados de cada URL
    all_keywords_for_comparison = [] # Para armazenar conjuntos de palavras-chave (apenas as palavras) para a lógica de comparação

    # Analisa cada URL individualmente
    for url_item in urls_to_analyze:
        error_message, keywords_data = analyze_single_url(url_item, stop_words_pt)
        if error_message:
            individual_results.append({"url": url_item, "status": "error", "message": error_message["error"]})
        else:
            individual_results.append({"url": url_item, "status": "success", "top_keywords": keywords_data})
            # Adiciona apenas as palavras-chave (sem os scores) para a comparação
            all_keywords_for_comparison.append(set(k[0] for k in keywords_data)) 

    # --- Lógica de Comparação entre as URLs ---
    common_keywords = set()
    unique_keywords_per_url = []

    if all_keywords_for_comparison:
        # 1. Encontra palavras-chave comuns a TODAS as URLs (interseção)
        if len(all_keywords_for_comparison) > 1:
            common_keywords = set.intersection(*all_keywords_for_comparison)
        elif len(all_keywords_for_comparison) == 1:
            # Se houver apenas uma URL analisada com sucesso, as "comuns" são as dela próprias
            common_keywords = all_keywords_for_comparison[0]

        # 2. Encontra palavras-chave únicas para cada URL
        for i, current_set in enumerate(all_keywords_for_comparison):
            unique_to_current = current_set.copy() # Começa com todas as palavras da URL atual
            for j, other_set in enumerate(all_keywords_for_comparison):
                if i != j:
                    # Remove palavras que aparecem em outras URLs
                    unique_to_current.difference_update(other_set)
            # Adiciona a URL original (do input) e suas palavras-chave únicas
            unique_keywords_per_url.append({"url": urls_to_analyze[i], "keywords": list(unique_to_current)})
    
    # Retorna os resultados individuais de cada URL e a comparação geral
    return jsonify({
        "individual_results": individual_results, # Resultados detalhados de cada URL
        "comparison": {
            "common_keywords": list(common_keywords), # Palavras-chave comuns a todas
            "unique_keywords_per_url": unique_keywords_per_url # Palavras-chave únicas para cada URL
        }
    })

# --- Execução do Aplicativo Flask ---
if __name__ == '__main__':
    app.run(debug=True)

