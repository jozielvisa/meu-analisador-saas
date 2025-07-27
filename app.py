from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import nltk
from collections import Counter
import re # Para expressões regulares

app = Flask(__name__)

# --- Configuração do NLTK para português ---
# Baixa os recursos 'punkt' e 'stopwords' do NLTK.
# 'punkt' é para tokenização (dividir texto em palavras/frases).
# 'stopwords' é para uma lista de palavras comuns que queremos ignorar (ex: "e", "de", "para").
# A linha abaixo vai tentar baixar se não encontrar. Já fizemos isso no passo anterior, mas é bom ter para garantir.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Carrega as stopwords em português
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
        # ou que podem poluir a análise (scripts JavaScript, estilos CSS, cabeçalhos, rodapés, menus de navegação).
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

        # Conta a frequência de cada palavra filtrada.
        word_counts = Counter(filtered_words)

        # Pega as 20 palavras mais comuns com suas contagens.
        top_keywords = word_counts.most_common(20)

        # --- 4. Retorno dos Resultados ---
        # Retorna os resultados em formato JSON para o frontend.
        # jsonify() converte dicionários Python para JSON.
        return jsonify({"url": url, "top_keywords": top_keywords})

    # --- Tratamento de Erros ---
    except requests.exceptions.RequestException as e:
        # Captura erros relacionados a problemas de rede ou HTTP (ex: site não encontrado, timeout).
        return jsonify({"error": f"Erro ao acessar a URL: {e}"}), 500
    except Exception as e:
        # Captura qualquer outro erro inesperado.
        return jsonify({"error": f"Ocorreu um erro inesperado: {e}"}), 500

# --- Execução do Aplicativo Flask ---
if __name__ == '__main__':
    # Roda o servidor Flask.
    # debug=True: Ativa o modo de depuração. Isso recarrega o servidor automaticamente a cada mudança no código
    #             e mostra mensagens de erro detalhadas. Ótimo para desenvolvimento, mas DESATIVE em produção.
    app.run(debug=True)