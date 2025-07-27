from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import nltk
from collections import Counter # Counter não será mais usado diretamente para TF-IDF, mas pode ser útil para depuração ou futuras funcionalidades
import re # Para expressões regulares
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# --- Configuração do NLTK para português ---
# Carrega as stopwords em português.
# Presume que os recursos 'punkt' e 'stopwords' já foram baixados durante o processo de build do Render.
stop_words_pt = set(nltk.corpus.stopwords.words('portuguese'))

# --- Rota Principal (Endpoint do Servidor Web) ---
# Quando alguém acessar a URL base do seu aplicativo (ex: http://127.0.0.1:5000/),
# esta função será executada e vai carregar a interface (frontend).
@app.route('/')
def index():
    return render_template('index.html') # Vai procurar 'index.html' na pasta 'templates'

# --- Rota de Análise (Endpoint da API) ---
# Esta rota é para o frontend enviar a URL do concorrente.
# Ela só aceita requisições POST, pois está "recebendo" dados para processar.
@app.route('/analyze', methods=['POST'])
def analyze():
    # Pega a URL do corpo da requisição JSON (enviada pelo frontend)
    url = request.json.get('url')
    if not url:
        # Se a URL não for fornecida, retorna um erro 400 (Bad Request)
        return jsonify({"error": "URL não fornecida"}), 400

    try:
        # --- 1. Coleta de Dados (Web Scraping) ---
        # Faz uma requisição HTTP GET para a URL fornecida.
        # timeout=10: Define um tempo máximo de 10 segundos para a requisição, evita que o aplicativo trave.
        response = requests.get(url, timeout=10)
        # raise_for_status(): Lança uma exceção se a requisição retornar um erro HTTP (4xx ou 5xx).
        response.raise_for_status()
        
        # Cria um objeto BeautifulSoup para analisar o conteúdo HTML da página.
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- 2. Pré-processamento do Conteúdo ---
        # Remove elementos HTML que geralmente não contêm texto relevante
        # ou que podem poluir a análise (ex: scripts JavaScript, estilos CSS, cabeçalhos, rodapés, menus de navegação).
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose() # Remove o elemento e seu conteúdo

        # Extrai todo o texto visível da página HTML.
        text = soup.get_text()

        # Limpa o texto:
        # re.sub(r'\s+', ' ', text): Substitui múltiplos espaços (incluindo quebras de linha e tabulações) por um único espaço.
        # .strip(): Remove espaços em branco do início e do fim do texto.
        text = re.sub(r'\s+', ' ', text).strip()
        # re.sub(r'[^\w\s]', '', text): Remove todos os caracteres que não são letras, números ou espaços.
        # Isso ajuda a limpar pontuações e símbolos.
        text = re.sub(r'[^\w\s]', '', text) 

        # --- 3. Processamento de Linguagem Natural (PLN) ---
        # Converte todo o texto para minúsculas para padronização.
        # nltk.word_tokenize(): Divide o texto em uma lista de palavras (tokens).
        # language='portuguese': Indica que o texto está em português para uma tokenização mais precisa.
        words = nltk.word_tokenize(text.lower(), language='portuguese')

        # Filtra as palavras:
        # word.isalpha(): Garante que a palavra é composta apenas por letras (remove números e símbolos restantes).
        # word not in stop_words_pt: Remove as palavras comuns da lista de stopwords.
        # len(word) > 2: Remove palavras muito curtas (ex: "a", "o", "e", que podem ter escapado).
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words_pt and len(word) > 2]

        # --- 3. Processamento de Linguagem Natural (PLN) com TF-IDF ---
        # O TfidfVectorizer precisa de uma string de texto, não uma lista de palavras.
        # Juntamos as palavras filtradas de volta em uma string.
        processed_text = " ".join(filtered_words)

        # Se não houver texto processado, não podemos calcular TF-IDF
        if not processed_text.strip():
            # Retorna um erro 400 se não houver texto suficiente para análise
            return jsonify({"error": "Não foi possível extrair conteúdo textual relevante suficiente desta URL para análise."}), 400

        # Inicializa o TfidfVectorizer.
        # analyzer='word': Analisa por palavras.
        # stop_words=stop_words_pt: Usa nossas stopwords em português.
        # max_features=200: Considera no máximo 200 features (palavras) mais relevantes para a análise.
        # ngram_range=(1,2): Considera palavras únicas (unigrams) e pares de palavras (bigrams).
        # Isso pode capturar termos como "marketing digital".
        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words=list(stop_words_pt), # Convertemos para lista, pois o TfidfVectorizer espera uma lista ou None
            max_features=200,
            ngram_range=(1, 2)
        )

        # Ajusta o vetorizador ao texto e o transforma em uma matriz TF-IDF.
        # É importante que o texto seja passado como uma lista de documentos, mesmo que seja apenas um.
        tfidf_matrix = vectorizer.fit_transform([processed_text])

        # Pega os nomes das features (as palavras/ngramas)
        feature_names = vectorizer.get_feature_names_out()

        # Pega os scores TF-IDF para o nosso documento (primeira linha da matriz)
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Cria uma lista de tuplas (palavra, score TF-IDF)
        word_tfidf_scores = list(zip(feature_names, tfidf_scores))

        # Ordena as palavras pelos scores TF-IDF em ordem decrescente
        # Pega as 20 palavras/ngramas com os scores TF-IDF mais altos.
        top_keywords = sorted(word_tfidf_scores, key=lambda x: x[1], reverse=True)[:20]

        # --- VERIFICAÇÃO ADICIONAL PARA POUCAS PALAVRAS-CHAVE RELEVANTES ---
        if not top_keywords: # Se a lista de palavras-chave top_keywords estiver vazia
            return jsonify({
                "error": "Não foi possível extrair palavras-chave relevantes suficientes desta URL. O conteúdo pode ser muito curto, não textual, ou não estar em português."
            }), 400 # Retorna um erro 400 (Bad Request)

        # Formata os resultados para ter a palavra e seu score (arredondado para 4 casas decimais)
        # O score é um float, e não uma contagem inteira.
        top_keywords_formatted = [(word, round(score, 4)) for word, score in top_keywords]

        # --- 4. Retorno dos Resultados ---
        # Retorna os resultados em formato JSON para o frontend.
        # jsonify() converte dicionários Python para JSON.
        return jsonify({"url": url, "top_keywords": top_keywords_formatted})

    # --- Tratamento de Erros ---
    except requests.exceptions.RequestException as e:
        # Captura erros relacionados a problemas de rede ou HTTP (ex: site não encontrado, timeout).
        return jsonify({"error": f"Erro ao acessar a URL: {e}. Verifique se a URL está correta e acessível."}), 500
    except Exception as e:
        # Captura qualquer outro erro inesperado.
        return jsonify({"error": f"Ocorreu um erro inesperado: {e}"}), 500

# --- Execução do Aplicativo Flask ---
if __name__ == '__main__':
    # Roda o servidor Flask.
    # debug=True: Ativa o modo de depuração. Isso recarrega o servidor automaticamente a cada mudança no código
    #             e mostra mensagens de erro detalhadas. Ótimo para desenvolvimento, mas DESATIVE em produção.
    app.run(debug=True)
