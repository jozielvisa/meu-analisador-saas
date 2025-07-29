from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, DetectorFactory
import firebase_admin
from firebase_admin import credentials, auth
from dotenv import load_dotenv
load_dotenv() # Carrega as variáveis do .env

# Garante que a detecção de idioma seja consistente
DetectorFactory.seed = 0

app = Flask(__name__)

# --- Configuração do Firebase Admin SDK ---
FIREBASE_SERVICE_ACCOUNT_CONFIG = None # OU {} (um dicionário vazio)

# Inicializa o Firebase Admin SDK
try:
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_CONFIG)
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK inicializado com sucesso!")
except ValueError as e:
    print(f"Erro ao inicializar Firebase Admin SDK: {e}. Verifique se FIREBASE_SERVICE_ACCOUNT_CONFIG está correto.")
except Exception as e:
    print(f"Erro inesperado na inicialização do Firebase: {e}")


# --- Configuração do NLTK para português ---
STOPWORDS_BY_LANG = {}
SUPPORTED_LANGUAGES = ['portuguese', 'english', 'spanish']

for lang in SUPPORTED_LANGUAGES:
    try:
        STOPWORDS_BY_LANG[lang] = set(nltk.corpus.stopwords.words(lang))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        STOPWORDS_BY_LANG[lang] = set(nltk.corpus.stopwords.words(lang))

# --- Rota Principal (Endpoint do Servidor Web) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Rotas de Autenticação ---
@app.route('/signup', methods=['POST'])
def signup():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"error": "Email e senha são obrigatórios."}), 400

    try:
        user = auth.create_user(email=email, password=password)
        # Opcional: Gerar um token de ID para login automático após o registro
        # custom_token = auth.create_custom_token(user.uid)
        return jsonify({"message": "Usuário criado com sucesso!", "uid": user.uid}), 201
    except Exception as e:
        return jsonify({"error": f"Erro ao criar usuário: {e}"}), 400

@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"error": "Email e senha são obrigatórios."}), 400

    try:
        # Firebase Admin SDK não tem uma função direta de "login" com email/senha.
        # Ele é usado para gerenciar usuários (criar, desabilitar, etc.) e verificar tokens.
        # O login real é feito no frontend com o SDK do Firebase Client.
        # Aqui, podemos apenas verificar se o usuário existe ou gerar um token personalizado
        # para o frontend usar. Para simplicidade, vamos apenas retornar sucesso se
        # o usuário existir (isso é mais para um cenário de Admin SDK).
        # Para um login real, o frontend enviaria as credenciais diretamente para o Firebase Auth.

        # Para simular um "login" no backend com Admin SDK, você pode:
        # 1. Tentar obter o usuário pelo email
        user = auth.get_user_by_email(email)
        # 2. Se o usuário existe, você pode gerar um custom token para o frontend
        custom_token = auth.create_custom_token(user.uid).decode('utf-8')
        return jsonify({"message": "Login bem-sucedido!", "uid": user.uid, "customToken": custom_token}), 200
    except auth.UserNotFoundError:
        return jsonify({"error": "Usuário não encontrado."}), 404
    except Exception as e:
        return jsonify({"error": f"Erro ao fazer login: {e}"}), 400

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
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button', 'input', 'select', 'textarea', 'iframe', 'noscript', 'img']):
            tag.decompose()

        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s]', '', text) 

        detected_lang_code = 'unknown'
        try:
            if len(text) > 50:
                detected_lang_code = detect(text)
            else:
                detected_lang_code = 'unknown'
        except Exception:
            detected_lang_code = 'unknown'

        nltk_lang_name = 'unknown'
        if detected_lang_code == 'pt':
            nltk_lang_name = 'portuguese'
        elif detected_lang_code == 'en':
            nltk_lang_name = 'english'
        elif detected_lang_code == 'es':
            nltk_lang_name = 'spanish'
        
        current_stopwords = STOPWORDS_BY_LANG.get(nltk_lang_name, set())

        words = nltk.word_tokenize(text.lower(), language=nltk_lang_name if nltk_lang_name != 'unknown' else 'english')
        filtered_words = [word for word in words if word.isalpha() and word not in current_stopwords and len(word) > 2]
        
        processed_text = " ".join(filtered_words)

        if not processed_text.strip():
            return {"error": "Conteúdo textual insuficiente para análise."}, None

        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words=list(current_stopwords),
            max_features=200,
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        word_tfidf_scores = list(zip(feature_names, tfidf_scores))
        top_keywords = sorted(word_tfidf_scores, key=lambda x: x[1], reverse=True)[:20]

        if not top_keywords:
            return {"error": "Não foi possível extrair palavras-chave relevantes suficientes."}, None
        
        top_keywords_formatted = [(word, round(score, 4)) for word, score in top_keywords]
        
        return None, {"keywords": top_keywords_formatted, "detected_language": nltk_lang_name} 

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
    app.run(debug=True)
