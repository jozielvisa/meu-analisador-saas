from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, DetectorFactory
import firebase_admin
from firebase_admin import credentials, auth, firestore # Importa firestore
import os
import json
from datetime import datetime, timedelta, timezone # Para gerenciar datas e limites

# Garante que a detecção de idioma seja consistente
DetectorFactory.seed = 0

app = Flask(__name__)

# --- Configuração do Firebase Admin SDK ---
FIREBASE_SERVICE_ACCOUNT_CONFIG = None
try:
    firebase_config_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if firebase_config_json:
        FIREBASE_SERVICE_ACCOUNT_CONFIG = json.loads(firebase_config_json)
    else:
        print("Variável de ambiente 'FIREBASE_SERVICE_ACCOUNT_JSON' não encontrada.")
except json.JSONDecodeError as e:
    print(f"Erro ao decodificar JSON da variável de ambiente FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
except Exception as e:
    print(f"Erro inesperado ao carregar credenciais do Firebase de variável de ambiente: {e}")

# Inicializa o Firebase Admin SDK e Firestore
db = None # Inicializa db como None
if FIREBASE_SERVICE_ACCOUNT_CONFIG:
    try:
        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_CONFIG)
        firebase_admin.initialize_app(cred)
        db = firestore.client() # Inicializa o cliente Firestore
        print("Firebase Admin SDK e Firestore inicializados com sucesso!")
    except ValueError as e:
        print(f"Erro ao inicializar Firebase Admin SDK: {e}. Verifique se FIREBASE_SERVICE_ACCOUNT_CONFIG está correto.")
    except Exception as e:
        print(f"Erro inesperado na inicialização do Firebase: {e}")
else:
    print("Firebase Admin SDK e Firestore NÃO inicializados: Credenciais não foram carregadas.")


# --- Configuração do NLTK para português ---
STOPWORDS_BY_LANG = {}
SUPPORTED_LANGUAGES = ['portuguese', 'english', 'spanish']

for lang in SUPPORTED_LANGUAGES:
    try:
        STOPWORDS_BY_LANG[lang] = set(nltk.corpus.stopwords.words(lang))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        STOPWORDS_BY_LANG[lang] = set(nltk.corpus.stopwords.words(lang))

# --- Limites de Uso ---
MAX_FREE_ANALYSES_PER_DAY = 5 # Limite de análises por dia para usuários gratuitos

# --- Rota Principal (Endpoint do Servidor Web) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Rotas de Autenticação (EXISTENTES) ---
@app.route('/signup', methods=['POST'])
def signup():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"error": "Email e senha são obrigatórios."}), 400

    if not firebase_admin._apps:
        return jsonify({"error": "Serviço de autenticação não disponível. Contate o suporte."}), 503

    try:
        user = auth.create_user(email=email, password=password)
        # Cria um documento de usuário no Firestore com dados iniciais
        user_ref = db.collection('users').document(user.uid)
        user_ref.set({
            'email': user.email,
            'is_premium': False, # Novo campo para status premium
            'analysis_count_today': 0,
            'last_analysis_date': datetime.now(timezone.utc) # Armazena a data da última análise (UTC)
        })
        return jsonify({"message": "Usuário criado com sucesso!", "uid": user.uid}), 201
    except Exception as e:
        error_message = str(e)
        if "EMAIL_ALREADY_EXISTS" in error_message:
            return jsonify({"error": "Este email já está em uso."}), 409
        elif "WEAK_PASSWORD" in error_message:
            return jsonify({"error": "A senha é muito fraca. Use pelo menos 6 caracteres."}), 400
        return jsonify({"error": f"Erro ao criar usuário: {error_message}"}), 400

@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({"error": "Email e senha são obrigatórios."}), 400

    if not firebase_admin._apps:
        return jsonify({"error": "Serviço de autenticação não disponível. Contate o suporte."}), 503

    try:
        user = auth.get_user_by_email(email)
        custom_token = auth.create_custom_token(user.uid).decode('utf-8')
        return jsonify({"message": "Login bem-sucedido!", "uid": user.uid, "customToken": custom_token}), 200
    except auth.UserNotFoundError:
        return jsonify({"error": "Usuário não encontrado."}), 404
    except Exception as e:
        return jsonify({"error": f"Erro ao fazer login: {e}"}), 400

# --- Middleware para verificar autenticação e status do usuário ---
# Esta função será chamada antes de processar a análise
@app.before_request
def check_auth_and_usage():
    # Apenas para a rota /analyze
    if request.path == '/analyze' and request.method == 'POST':
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Autenticação necessária. Faça login para usar o serviço."}), 401
        
        id_token = auth_header.split('Bearer ')[1]
        
        try:
            # Verifica o token de ID do Firebase
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            
            # Anexa o UID do usuário à requisição para uso posterior
            request.user_uid = uid

            # --- Lógica de Limite de Uso e Status Premium ---
            if not db: # Se o Firestore não foi inicializado
                return jsonify({"error": "Serviço de banco de dados não disponível. Contate o suporte."}), 503

            user_ref = db.collection('users').document(uid)
            user_doc = user_ref.get()

            if not user_doc.exists:
                # Se o documento do usuário não existe (deveria ter sido criado no signup), cria um padrão
                user_ref.set({
                    'email': decoded_token.get('email', 'N/A'),
                    'is_premium': False,
                    'analysis_count_today': 0,
                    'last_analysis_date': datetime.now(timezone.utc)
                })
                user_data = {'is_premium': False, 'analysis_count_today': 0, 'last_analysis_date': datetime.now(timezone.utc)}
            else:
                user_data = user_doc.to_dict()

            is_premium = user_data.get('is_premium', False)
            analysis_count_today = user_data.get('analysis_count_today', 0)
            last_analysis_date = user_data.get('last_analysis_date')

            # Resetar a contagem se for um novo dia (UTC)
            today_utc = datetime.now(timezone.utc).date()
            if last_analysis_date and last_analysis_date.date() < today_utc:
                analysis_count_today = 0
                user_ref.update({
                    'analysis_count_today': 0,
                    'last_analysis_date': datetime.now(timezone.utc)
                })
            
            if not is_premium and analysis_count_today >= MAX_FREE_ANALYSES_PER_DAY:
                return jsonify({
                    "error": f"Limite de análises diárias atingido ({MAX_FREE_ANALYSES_PER_DAY}). Faça upgrade para análises ilimitadas.",
                    "limit_reached": True # Adiciona um flag para o frontend identificar
                }), 429 # Too Many Requests
            
            # Anexa os dados do usuário à requisição para uso posterior na rota /analyze
            request.user_data = user_data
            request.user_ref = user_ref

        except auth.InvalidIdTokenError:
            return jsonify({"error": "Token de autenticação inválido ou expirado. Faça login novamente."}), 401
        except Exception as e:
            print(f"Erro na verificação de autenticação/limite: {e}")
            return jsonify({"error": f"Erro de autenticação ou limite: {e}"}), 401
    # Se não for a rota /analyze, continua normalmente
    return None

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

    # O limite de URLs por análise é o número de URLs na lista
    num_urls_in_request = len(urls_to_analyze)

    # Verifica o limite de análises por requisição para usuários gratuitos
    # Se o usuário não for premium e a requisição tiver mais URLs do que o permitido em uma única análise
    # (Ex: um usuário gratuito pode fazer 5 análises por dia, mas cada análise só pode ter 1 URL)
    # Por enquanto, estamos limitando o número total de análises por dia, não por requisição.
    # Esta lógica pode ser adicionada aqui se você quiser um limite por requisição também.

    individual_results = []
    all_keywords_for_comparison = []

    # Incrementa a contagem de análises para o usuário (se não for premium)
    # Isso é feito APÓS a verificação do limite no check_auth_and_usage
    # e antes de iniciar o processamento das URLs.
    user_data = request.user_data # Dados do usuário anexados pelo @app.before_request
    user_ref = request.user_ref # Referência ao documento do usuário anexada pelo @app.before_request

    if not user_data.get('is_premium', False):
        new_analysis_count = user_data.get('analysis_count_today', 0) + 1
        user_ref.update({
            'analysis_count_today': new_analysis_count,
            'last_analysis_date': datetime.now(timezone.utc)
        })
        # Opcional: Anexar o novo limite restante à resposta para o frontend
        remaining_analyses = MAX_FREE_ANALYSES_PER_DAY - new_analysis_count
        request.remaining_analyses = remaining_analyses # Anexa para uso na resposta final

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
    
    response_data = {
        "individual_results": individual_results,
        "comparison": {
            "common_keywords": list(common_keywords),
            "unique_keywords_per_url": unique_keywords_per_url
        }
    }
    # Adiciona informações de limite à resposta se for um usuário gratuito
    if not user_data.get('is_premium', False):
        response_data['user_limit_info'] = {
            'is_premium': False,
            'remaining_analyses': request.remaining_analyses,
            'max_free_analyses': MAX_FREE_ANALYSES_PER_DAY
        }
    else:
        response_data['user_limit_info'] = {
            'is_premium': True,
            'remaining_analyses': 'Unlimited', # Ou um valor grande
            'max_free_analyses': 'Unlimited'
        }

    return jsonify(response_data)

# --- Execução do Aplicativo Flask ---
if __name__ == '__main__':
    app.run(debug=True)
