import os
import io
import re
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURAÇÃO DA PÁGINA (STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="Gerenciador de OS & Relatórios IA",
    page_icon="🛠️",
    layout="wide"
)

st.markdown("""
    <style>
    /* Estilos do Dashboard Executivo */
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }

    /* --- AJUSTES DA BARRA LATERAL (SEM ESPAÇOS EXCEDENTES) --- */
    [data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    
    /* Reduz o espaço padrão entre os componentes da barra lateral */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }

    /* Reduz o espaçamento gerado pelas linhas divisórias (st.markdown("---")) */
    [data-testid="stSidebar"] hr {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Responsividade para celulares e telas menores */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem !important;
        }
        .sub-header {
            font-size: 1rem !important;
            margin-bottom: 10px;
        }
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.25rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        
        .stSelectbox, .stTextInput, .stFileUploader {
            width: 100% !important;
        }

        .stDataFrame {
            width: 100% !important;
            overflow-x: auto !important;
        }
    }
    </style>
""", unsafe_allow_html=True)
# ==========================================
# SISTEMA DE LOGIN MÚLTIPLO (CAIXA REDUZIDA)
# ==========================================
def verificar_autenticacao():
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False
        st.session_state.usuario_atual = ""

    if not st.session_state.autenticado:
        # Cria colunas para centralizar e limitar a largura da caixa de login
        col_esq, col_centro, col_dir = st.columns([1, 1.2, 1])
        
        with col_centro:
            st.markdown("## 🔐 Autenticação")
            st.markdown("Insira suas credenciais para acessar o painel.")
            
            with st.form("form_login"):
                usuario_input = st.text_input("Usuário").strip()
                senha_input = st.text_input("Senha", type="password")
                # use_container_width=True faz o botão ocupar a largura da caixa menor
                submit = st.form_submit_button("Entrar", use_container_width=True)
                
                if submit:
                    usuarios_cadastrados = {}
                    try:
                        if hasattr(st, "secrets") and "SENHA_APP" in st.secrets:
                            usuarios_cadastrados = dict(st.secrets["SENHA_APP"])
                    except Exception:
                        pass
                    
                    # Valida se o usuário existe e se a senha está correta
                    if usuario_input in usuarios_cadastrados and senha_input == usuarios_cadastrados[usuario_input]:
                        st.session_state.autenticado = True
                        st.session_state.usuario_atual = usuario_input
                        st.rerun()
                    else:
                        st.error("❌ Usuário ou senha incorretos.")
        return False
    return True

# Executa a verificação de login antes de carregar o app
if not verificar_autenticacao():
    st.stop()

# ==========================================
# FUNÇÕES DE AUXÍLIO PARA TRATAMENTO DE DADOS
# ==========================================
def formatar_moeda_br(valor):
    try:
        val_str = f"{float(valor):,.2f}"
        val_str = val_str.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {val_str}"
    except Exception:
        return "R$ 0,00"

def limpar_coluna_monetaria(serie):
    if serie is None:
        return pd.Series([0.0])
    
    if serie.dtype in ['float64', 'int64']:
        return serie.fillna(0.0)
    
    def converter_valor(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).strip()
        val_limpo = re.sub(r'[^\d,.-]', '', val_str)
        if not val_limpo or val_limpo in ['-', '.', ',']:
            return 0.0
        
        try:
            if '.' in val_limpo and ',' in val_limpo:
                val_limpo = val_limpo.replace('.', '').replace(',', '.')
            elif ',' in val_limpo:
                val_limpo = val_limpo.replace(',', '.')
            return float(val_limpo)
        except Exception:
            return 0.0

    return serie.apply(converter_valor).astype(float).fillna(0.0)

# ==========================================
# MÓDULO DE NLP & CLASSIFICAÇÃO DE TEXTO
# ==========================================
def classificar_texto_nlp(df: pd.DataFrame, coluna_texto: str) -> pd.DataFrame:
    df_cls = df.copy()
    if coluna_texto not in df_cls.columns:
        return df_cls
    
    textos = df_cls[coluna_texto].fillna("").astype(str).str.lower()
    
    def categorizar_por_regras(texto):
        texto = str(texto).lower().strip()

        if any(w in texto for w in [
            "vazamento", "vazando", "vaza", "água", "agua",
            "cano", "tubulação", "tubulacao", "tubo",
            "esgoto", "ralo", "torneira", "registro",
            "válvula", "valvula", "hydra", "descarga",
            "bacia sanitária", "vaso sanitário", "vaso sanitario",
            "pia", "chuveiro", "bebedouro", "caixa d'água",
            "caixa dagua", "reservatório", "reservatorio", "caixa dagua",
            "bomba d'água", "bomba dagua", "boia", "boia de nível",
            "pressão da água", "pressao da agua", "entupimento",
            "entupido", "infiltração hidráulica", "infiltracao hidraulica"
        ]):
            return "Hidráulica / Saneamento"

        elif any(w in texto for w in [
            "lâmpada", "lampada", "iluminação", "iluminacao",
            "luminária", "luminaria", "tomada", "interruptor",
            "disjuntor", "quadro elétrico", "quadro eletrico",
            "painel elétrico", "painel eletrico",
            "fio", "cabo elétrico", "cabo eletrico",
            "curto circuito", "curto-circuito",
            "falta de energia", "sem energia",
            "queda de energia", "energia elétrica",
            "energia eletrica", "contator", "relé", "rele",
            "fusível", "fusivel", "barramento",
            "aterramento", "dr", "dps", "inversor",
            "distribuição elétrica", "distribuicao eletrica"
        ]):
            return "Elétrica"

        elif any(w in texto for w in [
            "ar condicionado", "ar-condicionado",
            "split", "cassete", "vrf", "vrv",
            "climatização", "climatizacao",
            "refrigeração", "refrigeracao",
            "evaporadora", "condensadora",
            "compressor", "gás refrigerante", "gas refrigerante",
            "fluido refrigerante", "serpentina",
            "filtro do ar", "filtro de ar",
            "temperatura do ambiente", "não gela", "nao gela",
            "não refrigera", "nao refrigera",
            "geladeira", "freezer", "frigobar",
            "bebedouro refrigerado", "ventilador",
            "exaustor", "exaustão", "exaustao"
        ]):
            return "Climatização / Refrigeração"

        elif any(w in texto for w in [
            "parede", "reboco", "alvenaria", "argamassa",
            "cimento", "concreto", "piso", "revestimento",
            "cerâmica", "ceramica", "azulejo",
            "forro", "teto", "telhado", "cobertura",
            "calha", "rufo", "fachada", "infiltração", "infiltracao", 
             "trinca", "rachadura", "fissura",
            "porta", "janela", "batente", "fechadura",
            "dobradiça", "dobradica", "vidro", "umidade",
            "espelho", "divisória", "divisoria",
            "drywall", "granito", "mármore",
            "marcenaria", "carpintaria", "serralheria"
        ]):
            return "Civil / Edificações"

        elif any(w in texto for w in [
            "acabamento", "tinta", "massa corrida", "massa acrílica",
            "pintura", "textura", "gesso", "mofo"
        ]):
            return "Pintura / acabamento"

        elif any(w in texto for w in [
            "motor", "redutor", "engrenagem", "rolamento",
            "correia", "polia", "eixo", "acoplamento",
            "bucha", "mancal", "corrente mecânica",
            "lubrificação", "lubrificante",
            "graxa", "óleo", "oleo",
            "vibração mecânica", "vibracao mecanica",
            "desgaste mecânico", "desgaste mecanico",
            "mecânica", "mecanica",
            "pneumática", "pneumatica",
            "compressor de ar", "cilindro pneumático",
            "atuador", "válvula pneumática", "valvula pneumatica"
        ]):
            return "Mecânica"

        elif any(w in texto for w in [
            "equipamento", "máquina", "maquina",
            "forno", "microondas", "micro-ondas",
            "cafeteira", "liquidificador",
            "televisão", "televisao", "tv",
            "projetor", "datashow",
            "impressora", "scanner",
            "computador", "monitor",
            "notebook", "desktop",
            "equipamento de cozinha",
            "equipamento industrial",
            "equipamento técnico", "equipamento tecnico"
        ]):
            return "Equipamentos"

        elif any(w in texto for w in [
            "elevador", "elevadores",
            "cabina", "cabine do elevador",
            "porta do elevador",
            "comando do elevador",
            "painel do elevador",
            "nivelamento do elevador",
            "ascensor",
            "plataforma elevatória",
            "plataforma elevatoria"
        ]):
            return "Elevadores / Transporte Vertical"

        elif any(w in texto for w in [
            "extintor", "hidrante", "sprinkler",
            "sprinklers", "alarme de incêndio",
            "alarme de incendio", "detector de fumaça",
            "detector de fumaca", "central de incêndio",
            "central de incendio", "mangueira de incêndio",
            "mangueira de incendio", "brigada",
            "bomba de incêndio", "bomba de incendio",
            "sistema de incêndio", "sistema de incendio",
            "porta corta-fogo", "porta corta fogo",
            "iluminação de emergência",
            "iluminacao de emergencia",
            "saída de emergência", "saida de emergencia"
        ]):
            return "Prevenção e Combate a Incêndio"

        elif any(w in texto for w in [
            "spda", "para-raios", "para raios",
            "pára-raios", "captor", "descida do spda",
            "malha de aterramento", "malha de terra",
            "haste de aterramento", "aterramento predial",
            "equipotencialização", "equipotencializacao",
            "DPS", "proteção contra descargas",
            "descarga atmosférica", "descarga atmosferica"
        ]):
            return "SPDA / Aterramento"

        elif any(w in texto for w in [
            "internet", "wi-fi", "wifi", "rede",
            "cabeamento estruturado", "cabo de rede",
            "ponto de rede", "rack", "switch",
            "roteador", "access point", "ap",
            "fibra óptica", "fibra optica",
            "conector rj45", "patch panel",
            "telefonia", "telefone", "ramal",
            "cftv", "câmera", "camera",
            "controle de acesso", "catraca",
            "interfone"
        ]):
            return "Telecomunicações / TI"

        elif any(w in texto for w in [
            "gás", "gas", "glp", "gn",
            "botijão", "botijao",
            "central de gás", "central de gas",
            "tubulação de gás", "tubulacao de gas",
            "vazamento de gás", "vazamento de gas",
            "regulador de gás", "regulador de gas"
        ]):
            return "Gás / GLP"

        elif any(w in texto for w in [
            "cozinha industrial", "fogão industrial",
            "fogao industrial", "forno industrial",
            "coifa", "exaustão de cozinha",
            "exaustao de cozinha", "fritadeira",
            "chapa", "caldeirão", "caldeirao",
            "câmara fria", "camara fria",
            "bancada refrigerada", "processador de alimentos",
            "lavadora de louças", "lavadora de loucas"
        ]):
            return "Cozinha / Gastronomia"

        elif any(w in texto for w in [
            "piscina", "casa de máquinas da piscina",
            "casa de maquinas da piscina",
            "filtro da piscina", "bomba da piscina",
            "clorador", "dosador de cloro",
            "tratamento da piscina",
            "aspiração da piscina", "aspiracao da piscina"
        ]):
            return "Piscina / Tratamento de Água"

        elif any(w in texto for w in [
            "jardim", "jardinagem", "grama",
            "gramado", "árvore", "arvore",
            "poda", "paisagismo", "canteiro",
            "irrigação", "irrigacao",
            "plantio", "mato", "vegetação", "vegetacao"
        ]):
            return "Paisagismo / Áreas Externas"

        elif any(w in texto for w in [
            "limpeza", "higienização", "higienizacao",
            "lavagem", "desinfecção", "desinfeccao",
            "sanitização", "sanitizacao",
            "lixo", "resíduo", "residuo",
            "entulho", "sujeira", "conservação",
            "conservacao", "dedetização", "dedetizacao",
            "pragas", "insetos", "baratas", "cupins"
        ]):
            return "Limpeza / Conservação"

        elif any(w in texto for w in [
            "hóspede", "hospede",
            "hóspedes", "hospedes",
            "quarto", "quartos",
            "suíte", "suite",
            "apartamento", "hospedagem",
            "hotel", "hotelaria",
            "frigobar do quarto",
            "tv do quarto",
            "cama", "colchão", "colchao",
            "enxoval", "rouparia"
        ]):
            return "Hotelaria / Hospedagem"

        elif any(w in texto for w in [
            "acessibilidade", "pessoa com deficiência",
            "pessoa com deficiencia", "pcd",
            "rampa", "corrimão", "corrimao",
            "piso tátil", "piso tatil",
            "banheiro acessível", "banheiro acessivel",
            "barra de apoio", "plataforma acessível",
            "plataforma acessivel"
        ]):
            return "Acessibilidade"

        elif any(w in texto for w in [
            "cadeira", "mesa", "armário", "armario",
            "gaveta", "estante", "balcão", "balcao",
            "banco", "mobiliário", "mobiliario",
            "móvel", "movel", "escrivaninha",
            "arquivo", "prateleira"
        ]):
            return "Mobiliário"

        elif any(w in texto for w in [
            "segurança patrimonial", "seguranca patrimonial",
            "alarme", "sensor de presença",
            "sensor de presenca", "cerca elétrica",
            "cerca eletrica", "portão eletrônico",
            "portao eletronico", "fechamento eletrônico",
            "fechamento eletronico"
        ]):
            return "Segurança Patrimonial"

        else:
            return "Outros / Diversos"

    df_cls["CATEGORIA_NLP"] = textos.apply(categorizar_por_regras)
    
    try:
        mask_outros = df_cls["CATEGORIA_NLP"] == "Outros / Diversos"
        if mask_outros.sum() > 10:
            vectorizer = TfidfVectorizer(max_features=500, stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'na', 'por'])
            X = vectorizer.fit_transform(textos[mask_outros])
            n_clusters = min(3, max(1, int(mask_outros.sum() / 10)))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X)
                cluster_map = {0: "Geral / Manutenção Corretiva", 1: "Solicitação / Atendimento", 2: "Inspeção / Chamado Técnico"}
                df_cls.loc[mask_outros, "CATEGORIA_NLP"] = [cluster_map.get(c, "Outros / Diversos") for c in clusters]
    except Exception:
        pass

    return df_cls

# ==========================================
# CONFIGURAÇÃO DA IA (GROQ)
# ==========================================
def obter_modelo_ativo(client, api_key: str) -> str:
    if not api_key:
        return "llama-3.1-8b-instant"
    try:
        modelos_disponiveis = client.models.list()
        for m in modelos_disponiveis.data:
            m_id = m.id.lower()
            if "llama-3.1-8b" in m_id or "8b" in m_id:
                return m.id
        for m in modelos_disponiveis.data:
            m_id = m.id.lower()
            if any(nome in m_id for nome in ["llama-3.1", "qwen"]) and not any(ign in m_id for ign in ["guard", "safeguard", "whisper", "vision", "embed", "gpt-oss"]):
                return m.id
    except Exception:
        pass
    return "llama-3.1-8b-instant"

def gerar_relatorio_ia(termo: str, lista_observacoes: list, api_key: str) -> str:
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral."
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}"

    modelo_ativo = obter_modelo_ativo(client, api_key)
    
    df_obs = pd.DataFrame({"obs": [str(o).strip().lower() for o in lista_observacoes]})
    top_falhas = df_obs["obs"].value_counts().head(15).reset_index()
    top_falhas.columns = ["Descricao_Falha", "Frequencia"]
    
    ranking_texto = "\n".join([f"- {row['Descricao_Falha']} (Ocorrências: {row['Frequencia']})" for _, row in top_falhas.iterrows()])

    prompt = f"""Você é um especialista em manutenção de Ordens de Serviço (OS).
Foram encontrados {len(lista_observacoes)} registros no total para o termo '{termo.upper()}'. 
Abaixo está o ranking real das descrições de falhas mais repetidas na coluna de observação:
{ranking_texto}

Escreva um relatório executivo detalhado em português contendo:
1. Resumo dos Principais Problemas Relatados (Inclua obrigatoriamente uma tabela em Markdown com colunas 'Descrição da Falha / Problema' e 'Frequência')
2. Padrões ou Causas Recorrentes baseados nas repetições reais
3. Recomendações e Ações Preventivas direcionadas a esses problemas"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo_ativo,
            temperature=0.2,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro ao gerar relatório com a IA: {e}"

def responder_pergunta_livre_com_todo_arquivo(pergunta_usuario: str, df: pd.DataFrame, api_key: str) -> str:
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral."
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}"

    modelo_ativo = obter_modelo_ativo(client, api_key)
    total_linhas = len(df)
    
    resumo_status = dict(df["STATUS"].value_counts().head(5)) if "STATUS" in df.columns else "Não disponível"
    col_setor = "SETOR" if "SETOR" in df.columns else ("MÁQUINA" if "MÁQUINA" in df.columns else None)
    resumo_setor = dict(df[col_setor].value_counts().head(5)) if col_setor else "Não disponível"
    resumo_nlp = dict(df["CATEGORIA_NLP"].value_counts().head(5)) if "CATEGORIA_NLP" in df.columns else "Não disponível"
    
    periodo_dados = "Não disponível"
    col_data_ref = "ABERTO EM" if "ABERTO EM" in df.columns else ("DATA" if "DATA" in df.columns else None)
    if col_data_ref:
        try:
            dt_min = pd.to_datetime(df[col_data_ref], errors='coerce').min().strftime('%d/%m/%Y')
            dt_max = pd.to_datetime(df[col_data_ref], errors='coerce').max().strftime('%d/%m/%Y')
            periodo_dados = f"De {dt_min} até {dt_max}"
        except Exception:
            pass

    pergunta_lower = pergunta_usuario.lower()
    palavras_chave = [p for p in pergunta_lower.split() if len(p) > 3]
    
    df_relevante = df
    col_obs = "OBSERVAÇÃO ABERTURA" if "OBSERVAÇÃO ABERTURA" in df.columns else df.columns[0]
    if palavras_chave:
        filtro = df[col_obs].astype(str).str.lower().apply(lambda x: any(p in x for p in palavras_chave))
        df_filtrado_ia = df[filtro]
        if len(df_filtrado_ia) > 0:
            df_relevante = df_filtrado_ia

    df_amostra = df_relevante.head(5).copy()
    if col_obs in df_amostra.columns:
        df_amostra[col_obs] = df_amostra[col_obs].astype(str).str.slice(0, 60)

    colunas_visiveis = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "MÁQUINA", "CATEGORIA_NLP", col_obs] if c in df_amostra.columns]
    amostra_relevante = df_amostra[colunas_visiveis].to_string(index=False)

    prompt = f"""Você é um analista especialista em gestão de manutenção. A planilha foi processada com NLP e possui os seguintes dados consolidados:

--- ESTATÍSTICAS GLOBAIS DA PLANILHA ({total_linhas} registros) ---
- Total de registros: {total_linhas}
- Período coberto: {periodo_dados}
- Top Categorias NLP: {resumo_nlp}
- Distribuição por Status: {resumo_status}
- Top Setores/Máquinas: {resumo_setor}
------------------------------------------------------------

--- AMOSTRA RELEVANTE ({len(df_relevante)} correspondências) ---
{amostra_relevante}
------------------------------------------------------------

Responda objetivamente e de forma precisa em português:
Pergunta: {pergunta_usuario}"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo_ativo,
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro ao responder pergunta: {e}"

def extrair_dados_tabela_markdown(relatorio_texto: str) -> pd.DataFrame:
    try:
        tabelas = pd.read_html(io.StringIO(relatorio_texto))
        if tabelas:
            df_tabela = tabelas[0]
            col_freq, col_cat = None, None
            for col in df_tabela.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ["frequência", "frequencia", "ocorrências", "ocorrencias", "quantidade"]):
                    col_freq = col
                elif any(k in col_lower for k in ["área", "area", "tipo", "problema", "descrição", "descricao"]):
                    if not col_cat:
                        col_cat = col
            if col_freq and col_cat:
                df_tabela["Valor_Numerico"] = df_tabela[col_freq].astype(str).str.extract(r'(\d+)').astype(float).fillna(1)
                df_resultado = df_tabela[[col_cat, "Valor_Numerico"]].dropna()
                df_resultado.columns = ["Categoria", "Frequência"]
                return df_resultado.set_index("Categoria")["Frequência"]
    except Exception:
        pass
    return None


# ==========================================
# CARREGAMENTO DO ARQUIVO (ONEDRIVE / MANUAL) + TRATAMENTO DE VAZIOS
# ==========================================
caminho_padrao_onedrive = os.path.expanduser(r"~\OneDrive\base_os.csv")
df = None

if os.path.exists(caminho_padrao_onedrive):
    try:
        df = pd.read_csv(caminho_padrao_onedrive, sep=None, engine='python')
        st.sidebar.success("📁 Base carregada do OneDrive!")
    except Exception:
        pass

if df is None:
    st.sidebar.markdown("### 📂 Carregar Dados Manualmente")
    uploaded_file = st.sidebar.file_uploader("Envie sua base (CSV / Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            st.stop()
    else:
        st.warning("⚠️ Envie a base de dados na barra lateral para iniciar.")
        st.stop()

# Padroniza as colunas da base
df.columns = [str(c).strip().upper() for c in df.columns]

# 👉 AJUSTE CRUCIAL: Preenche as observações vazias para não descartar os Checklists
coluna_comentario = "OBSERVAÇÃO ABERTURA" if "OBSERVAÇÃO ABERTURA" in df.columns else (df.columns[0] if len(df.columns) > 0 else "")
if coluna_comentario in df.columns:
    df[coluna_comentario] = df[coluna_comentario].fillna("Sem observação de abertura")

df = classificar_texto_nlp(df, coluna_comentario)


df = classificar_texto_nlp(df, coluna_comentario)
# ==========================================
# INTERFACE GRÁFICA (BARRA LATERAL)
# ==========================================
st.sidebar.title("⚙️ Configurações")

# --- INFORMAÇÕES DO USUÁRIO E LOGOUT ---
st.sidebar.markdown(f"👤 Logado como: **{st.session_state.get('usuario_atual', 'Usuário')}**")

if st.sidebar.button("🚪 Sair", use_container_width=True, type="secondary"):
    # Limpa as variáveis de autenticação da sessão
    st.session_state.autenticado = False
    st.session_state.usuario_atual = ""
    # Recarrega a aplicação para voltar à tela de login
    st.rerun()

st.sidebar.markdown("---")
# ----------------------------------------

groq_api_key_input = ""
try:
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_api_key_input = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key_input and os.path.exists("groq_api_key.txt"):
    try:
        with open("groq_api_key.txt", "r", encoding="utf-8") as f:
            groq_api_key_input = f.read().strip()
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Seleção de Unidade")

# Lista de unidades disponíveis
lista_unidades = [
    "HOTEL BARREIRA ROXA",
    "CHECKLIST - DIRETORIA - HOTEL",
    "SENAC CENTRO", 
    "SENAC ALECRIM", 
    "SENAC ZONA NORTE", 
    "SENAC ZONA SUL", 
    "SENAC MOSSORÓ", 
    "SENAC CAICÓ", 
    "SENAC ASSÚ",  
    "CEP BARREIRA ROXA", 
    "CARRETA GASTRONOMIA", 
    "CARRETA MODA E BELEZA", 
    "CARRETA INFORMÁTICA"
    
]

unidade_escolhida = st.sidebar.selectbox("Escolha a Unidade", lista_unidades)

st.sidebar.markdown("---")
arquivo_carregado = st.sidebar.file_uploader("Enviar manualmente ", type=["csv", "xls", "xlsx"])
# ==========================================
# CORPO PRINCIPAL DO APLICATIVO
# ==========================================
st.markdown('<p class="main-header">🛠️ Gestão da Manutenção - Visão Geral & Painel Analítico</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Unidade Selecionada: <b>{unidade_escolhida}</b> | Classificação NLP, KPIs e Chat com IA.</p>', unsafe_allow_html=True)

if 'termo_pesquisa' not in st.session_state:
    st.session_state.termo_pesquisa = ""

if 'mensagens_chat' not in st.session_state:
    st.session_state.mensagens_chat = []

def limpar_pesquisa():
    if "relatorio_gerado" in st.session_state:
        del st.session_state.relatorio_gerado
    if "termo_relatorio" in st.session_state:
        del st.session_state.termo_relatorio

# Carregamento do arquivo (Via Secrets mapeando por Unidade -> Upload Manual -> Arquivo Local)
df = None
coluna_comentario = "OBSERVAÇÃO ABERTURA"

# Busca a URL específica da unidade nos segredos (ex: st.secrets["UNIDADES"]["SENAC CENTRO"])
url_unidade = ""
try:
    if hasattr(st, "secrets") and "UNIDADES" in st.secrets and unidade_escolhida in st.secrets["UNIDADES"]:
        url_unidade = st.secrets["UNIDADES"][unidade_escolhida]
except Exception:
    pass

if url_unidade:
    try:
        link_download = url_unidade
        if "?" in link_download and "download=1" not in link_download:
            link_download += "&download=1"
        elif "?" not in link_download:
            link_download += "?download=1"

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(link_download, headers=headers)
        
        if response.status_code == 200:
            if ".csv" in link_download.lower():
                df = pd.read_csv(io.BytesIO(response.content))
            else:
                df = pd.read_excel(io.BytesIO(response.content))
            st.success(f"✅ Dados da unidade **{unidade_escolhida}** carregados com sucesso ({len(df):,} registros)!")
        else:
            st.error(f"❌ Erro HTTP {response.status_code} ao acessar o link do OneDrive para `{unidade_escolhida}`.")
    except Exception as e:
        st.error(f"❌ Erro ao baixar dados da unidade: {e}")
elif arquivo_carregado is not None:
    try:
        if arquivo_carregado.name.endswith('.csv'):
            df = pd.read_csv(arquivo_carregado)
        else:
            df = pd.read_excel(arquivo_carregado)
        st.success(f"✅ Arquivo `{arquivo_carregado.name}` carregado com sucesso ({len(df):,} registros)!")
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo enviado: {e}")
elif os.path.exists(f"{unidade_escolhida}.csv"):
    try:
        df = pd.read_csv(f"{unidade_escolhida}.csv")
        st.info(f"ℹ️ Usando arquivo local: `{unidade_escolhida}.csv` ({len(df):,} registros)")
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo local: {e}")
elif os.path.exists(f"{unidade_escolhida}.xls"):
    try:
        df = pd.read_excel(f"{unidade_escolhida}.xls")
        st.info(f"ℹ️ Usando arquivo local: `{unidade_escolhida}.xls` ({len(df):,} registros)")
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo local: {e}")
else:
    st.warning(f"⚠️ Configure os links das unidades em `secrets.toml` ou envie uma planilha para a unidade **{unidade_escolhida}**.")

if df is not None:
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    if coluna_comentario not in df.columns:
        st.error(f"❌ A coluna obrigatória **'{coluna_comentario}'** não foi encontrada na planilha.")
        st.write(f"📋 **Colunas disponíveis na planilha:** `{list(df.columns)}`")
    else:
        df = df.dropna(subset=[coluna_comentario]).copy()

        if "CATEGORIA_NLP" not in df.columns:
            with st.spinner(f"Processando algoritmo de classificação NLP em {len(df):,} registros..."):
                df = classificar_texto_nlp(df, coluna_comentario)

        col_maquinas_ref = "MÁQUINAS" if "MÁQUINAS" in df.columns else ("MAQUINAS" if "MAQUINAS" in df.columns else None)
        
        if col_maquinas_ref:
            df["TEXTO_BUSCA"] = (
                df[coluna_comentario].astype(str).str.lower().fillna("") + " " + 
                df[col_maquinas_ref].astype(str).str.lower().fillna("")
            )
        else:
            df["TEXTO_BUSCA"] = df[coluna_comentario].astype(str).str.lower().fillna("")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Filtros Dashboard")
        
        df_filtrado_dashboard = df.copy()
        
    if "STATUS" in df.columns:
        lista_status = ["Todos"] + list(df["STATUS"].dropna().unique())
        status_escolhido = st.sidebar.selectbox("Filtrar por Status", lista_status)
        if status_escolhido != "Todos":
            df_filtrado_dashboard = df_filtrado_dashboard[df_filtrado_dashboard["STATUS"] == status_escolhido]

    # ==========================================
    # FILTRO COMPLETO E EXATO POR TIPO DE MANUTENÇÃO
    # ==========================================
    col_tipo_servico = None
    for c in df.columns:
        c_limpo = c.strip().upper()
        if "TIPO DE MANUT" in c_limpo or c_limpo == "TIPO":
            col_tipo_servico = c
            break

    if col_tipo_servico:
        tipos_desejados = ["Corretiva Programada", "Corretiva Emergencial", "Checklist", "Roteiro", "Melhoria"]
        lista_tipos = ["Todos"] + tipos_desejados
        tipo_escolhido = st.sidebar.selectbox("Filtrar por Tipo", lista_tipos)
        
        if tipo_escolhido != "Todos":
            termo = tipo_escolhido.strip().lower()
            coluna_str = df_filtrado_dashboard[col_tipo_servico].astype(str).str.strip().str.lower()
            
            if termo == "checklist":
                # Captura exatamente o valor "checklist" presente na base
                df_filtrado_dashboard = df_filtrado_dashboard[coluna_str == "checklist"]
            elif "corretiva programada" in termo:
                df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("corretiva", na=False) & coluna_str.str.contains("program", na=False)]
            elif "corretiva emergencial" in termo:
                df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("corretiva", na=False) & coluna_str.str.contains("emerg|urg", na=False)]
            elif "roteiro" in termo:
                # Captura todas as variações de roteiro, como "ROTEIRO - PERIÓDICA"
                df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("roteiro|rot", na=False)]
            elif "melhoria" in termo:
                df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("melhoria|melhor", na=False)]
    else:
        st.sidebar.info("Coluna de Tipo de Manutenção não encontrada na base.")
        st.markdown("---")
        
           # ==========================================
# NAVEGAÇÃO POR BOTÕES (SUBSTITUINDO AS ABAS)
# ==========================================

# Inicializa a aba ativa no session_state se não existir
if "aba_ativa" not in st.session_state:
    st.session_state.aba_ativa = "dash"

# Cria colunas para dispor os botões lado a lado no topo
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    if st.button("📊 Dashboard Geral", use_container_width=True, type="primary" if st.session_state.aba_ativa == "dash" else "secondary"):
        st.session_state.aba_ativa = "dash"
        st.rerun()

with col_b2:
    if st.button("🔍 Pesquisa & Relatórios", use_container_width=True, type="primary" if st.session_state.aba_ativa == "busca" else "secondary"):
        st.session_state.aba_ativa = "busca"
        st.rerun()

with col_b3:
    if st.button("🏷️ Classificação (NLP)", use_container_width=True, type="primary" if st.session_state.aba_ativa == "nlp" else "secondary"):
        st.session_state.aba_ativa = "nlp"
        st.rerun()

with col_b4:
    if st.button("💬 Chat Inteligente", use_container_width=True, type="primary" if st.session_state.aba_ativa == "chat" else "secondary"):
        st.session_state.aba_ativa = "chat"
        st.rerun()

st.markdown("---")

# ==========================================
# CONTEÚDO DE CADA SEÇÃO (BASEADO NO BOTÃO CLICADO)
# ==========================================

if st.session_state.aba_ativa == "dash":
    st.subheader(f"📈 Visão Executiva Completa da Manutenção - {unidade_escolhida}")
    
    # Inicializa o estado do filtro dos KPIs se não existir
    if "filtro_kpi_ativo" not in st.session_state:
        st.session_state.filtro_kpi_ativo = "geral"

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    total_os = len(df_filtrado_dashboard)
    
    if "CUSTO COM MATERIAIS" in df_filtrado_dashboard.columns:
        val_mat = limpar_coluna_monetaria(df_filtrado_dashboard["CUSTO COM MATERIAIS"]).sum()
        custo_materiais = formatar_moeda_br(val_mat)
    else:
        custo_materiais = "R$ 0,00"
        val_mat = 0.0

    if "MÃO DE OBRA EXTERNA" in df_filtrado_dashboard.columns:
        val_moe = limpar_coluna_monetaria(df_filtrado_dashboard["MÃO DE OBRA EXTERNA"]).sum()
        mao_de_obra = formatar_moeda_br(val_moe)
    else:
        mao_de_obra = "R$ 0,00"
        val_moe = 0.0
    
    total_os_str = f"{total_os:,}".replace(",", ".")
    custo_total_val = val_mat + val_moe

    with col_kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">TOTAL DE OS</h4>
                <h2 style="color: #1E3A8A; font-size: 26px; margin: 0;">{total_os_str}</h2>
            </div>
        """, unsafe_allow_html=True)
        if st.button("📋 Detalhar OS", key="btn_kpi_os", use_container_width=True):
            st.session_state.filtro_kpi_ativo = "todas_os"
            st.rerun()

    with col_kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">CUSTO MATERIAIS</h4>
                <h2 style="color: #059669; font-size: 26px; margin: 0;">{custo_materiais}</h2>
            </div>
        """, unsafe_allow_html=True)
        if st.button("📦 Ver Materiais", key="btn_kpi_mat", use_container_width=True):
            st.session_state.filtro_kpi_ativo = "materiais"
            st.rerun()

    with col_kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">MÃO DE OBRA EXTERNA</h4>
                <h2 style="color: #D97706; font-size: 26px; margin: 0;">{mao_de_obra}</h2>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🛠️ Ver Mão de Obra", key="btn_kpi_moe", use_container_width=True):
            st.session_state.filtro_kpi_ativo = "mao_de_obra"
            st.rerun()

    with col_kpi4:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">CUSTO TOTAL</h4>
                <h2 style="color: #DC2626; font-size: 26px; margin: 0;">{formatar_moeda_br(custo_total_val)}</h2>
            </div>
        """, unsafe_allow_html=True)
        if st.button("💰 Ver Custos Detalhados", key="btn_kpi_custo", use_container_width=True):
            st.session_state.filtro_kpi_ativo = "custos_totais"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.filtro_kpi_ativo != "geral":
        st.markdown("---")
        if st.session_state.filtro_kpi_ativo == "todas_os":
            st.info("📌 **Filtro Ativo:** Exibindo todas as Ordens de Serviço da unidade.")
            st.dataframe(df_filtrado_dashboard, use_container_width=True, hide_index=True)
        elif st.session_state.filtro_kpi_ativo == "materiais":
            st.info("📌 **Filtro Ativo:** Exibindo OS com foco em Custo de Materiais.")
            cols_mat = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "CUSTO COM MATERIAIS"] if c in df_filtrado_dashboard.columns]
            st.dataframe(df_filtrado_dashboard[cols_mat], use_container_width=True, hide_index=True)
        elif st.session_state.filtro_kpi_ativo == "mao_de_obra":
            st.info("📌 **Filtro Ativo:** Exibindo OS com foco em Mão de Obra Externa.")
            cols_moe = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "MÃO DE OBRA EXTERNA"] if c in df_filtrado_dashboard.columns]
            st.dataframe(df_filtrado_dashboard[cols_moe], use_container_width=True, hide_index=True)
        elif st.session_state.filtro_kpi_ativo == "custos_totais":
            st.info("📌 **Filtro Ativo:** Exibindo comparativo financeiro completo.")
            cols_custos = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "CUSTO COM MATERIAIS", "MÃO DE OBRA EXTERNA"] if c in df_filtrado_dashboard.columns]
            st.dataframe(df_filtrado_dashboard[cols_custos], use_container_width=True, hide_index=True)
        
        if st.button("🔄 Limpar Filtro do KPI"):
            st.session_state.filtro_kpi_ativo = "geral"
            st.rerun()

   # ==========================================
    # NOVOS INDICADORES EXECUTIVOS (CARDS ADICIONAIS)
    # ==========================================
    col_ind1, col_ind2, col_ind3 = st.columns(3)
    
    if "STATUS" in df_filtrado_dashboard.columns:
        status_txt = df_filtrado_dashboard["STATUS"].astype(str).str.lower()
        concluidas_count = status_txt.str.contains("concluíd|concluid|fechad|encerrad|finalizad").sum()
        conformidade_pct = (concluidas_count / total_os * 100) if total_os > 0 else 0.0
        conf_str = f"{conformidade_pct:.1f}".replace(".", ",") + "%"
        
        backlog_count = total_os - concluidas_count
        backlog_str = f"{backlog_count:,}".replace(",", ".")
    else:
        conf_str = "N/A"
        backlog_str = "N/A"

    tempo_medio_str = "N/A"
    col_data_abertura = "ABERTO EM" if "ABERTO EM" in df_filtrado_dashboard.columns else ("DATA" if "DATA" in df_filtrado_dashboard.columns else None)
    col_data_fechamento = "FECHADO EM" if "FECHADO EM" in df_filtrado_dashboard.columns else ("DATA FECHAMENTO" if "DATA FECHAMENTO" in df_filtrado_dashboard.columns else None)
    
    if col_data_abertura and col_data_fechamento:
        try:
            dt_ini = pd.to_datetime(df_filtrado_dashboard[col_data_abertura], errors='coerce')
            dt_fim = pd.to_datetime(df_filtrado_dashboard[col_data_fechamento], errors='coerce')
            diff_dias = (dt_fim - dt_ini).dt.total_seconds() / 86400
            media_dias = diff_dias[diff_dias >= 0].mean()
            if not pd.isna(media_dias):
                tempo_medio_str = f"{media_dias:.1f} dias".replace(".", ",")
        except Exception:
            pass

    with col_ind1:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">% CONFORMIDADE GERAL</h4>
                <h2 style="color: #2563EB; font-size: 24px; margin: 0;">{conf_str}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col_ind2:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">BACKLOG DE MANUTENÇÃO</h4>
                <h2 style="color: #9333EA; font-size: 24px; margin: 0;">{backlog_str} OS</h2>
            </div>
        """, unsafe_allow_html=True)
    with col_ind3:
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">TEMPO MÉDIO DE ATENDIMENTO</h4>
                <h2 style="color: #0D9488; font-size: 24px; margin: 0;">{tempo_medio_str}</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # GARANTINDO A DEFINIÇÃO DAS VARIÁVEIS DE DATA
    # ==========================================
    col_data = "ABERTO EM" if "ABERTO EM" in df_filtrado_dashboard.columns else ("DATA" if "DATA" in df_filtrado_dashboard.columns else None)
    col_data_fechamento = "FECHADO EM" if "FECHADO EM" in df_filtrado_dashboard.columns else ("DATA FECHAMENTO" if "DATA FECHAMENTO" in df_filtrado_dashboard.columns else None)

    # ==========================================
    # 1. EVOLUÇÃO TEMPORAL EMPILHADA (Abertas, Pendentes, Programadas, Finalizadas)
    # ==========================================
    st.markdown("### 📊 Evolução Temporal das OS por Status (Finalizadas, Pendentes, Programadas)")
    if col_data and "STATUS" in df_filtrado_dashboard.columns:
        df_evolucao = df_filtrado_dashboard.copy()
        df_evolucao["MES_ANO"] = pd.to_datetime(df_evolucao[col_data], errors='coerce').dt.to_period("M").astype(str)
        
        # Categorização detalhada para o gráfico de barras
        def categorizar_status_detalhado(val):
            v = str(val).lower()
            if any(term in v for term in ["concluíd", "concluid", "fechad", "encerrad", "finalizad"]):
                return "Finalizadas"
            elif any(term in v for term in ["program", "planejad"]):
                return "Programadas"
            else:
                return "Pendentes / Abertas"
                
        df_evolucao["STATUS_DETALHE"] = df_evolucao["STATUS"].apply(categorizar_status_detalhado)
        df_group_evol = df_evolucao.groupby(["MES_ANO", "STATUS_DETALHE"]).size().reset_index(name="Quantidade")
        
        fig_evolucao_bar = px.bar(
            df_group_evol, 
            x="MES_ANO", 
            y="Quantidade", 
            color="STATUS_DETALHE", 
            text="Quantidade",
            barmode="stack",
            color_discrete_map={"Finalizadas": "#059669", "Programadas": "#2563EB", "Pendentes / Abertas": "#DC2626"}
        )
        fig_evolucao_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380, separators=',.', legend_title="Status das OS")
        st.plotly_chart(fig_evolucao_bar, use_container_width=True)
    else:
        st.info("Colunas de data ou status indisponíveis para gerar a evolução temporal detalhada.")

    st.markdown("---")

    # ==========================================
    # 2. Gráficos: OS Abertas x Concluídas & Categoria NLP
    # ==========================================
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("### 🔄 OS Abertas x Concluídas")
        if "STATUS" in df_filtrado_dashboard.columns:
            s_low = df_filtrado_dashboard["STATUS"].astype(str).str.lower()
            cond_conc = s_low.str.contains("concluíd|concluid|fechad|encerrad|finalizad")
            df_filtrado_dashboard["STATUS_AGRUPADO"] = "Concluídas / Fechadas"
            df_filtrado_dashboard.loc[~cond_conc, "STATUS_AGRUPADO"] = "Pendentes / Abertas"
            
            df_status_comp = df_filtrado_dashboard["STATUS_AGRUPADO"].value_counts().reset_index()
            df_status_comp.columns = ["Status", "Quantidade"]
            df_status_comp["Texto_Fmt"] = df_status_comp["Quantidade"].apply(lambda x: f"{x:,}".replace(",", "."))
            
            fig_abertas_conc = px.bar(df_status_comp, x="Status", y="Quantidade", text="Texto_Fmt", color="Status", color_discrete_sequence=["#059669", "#DC2626"])
            fig_abertas_conc.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, showlegend=False, separators=',.')
            st.plotly_chart(fig_abertas_conc, use_container_width=True)
        else:
            st.info("Coluna de status não disponível para comparação.")

    with col_g2:
        st.markdown("### 🏷️ Pendências por Categoria (NLP)")
        if "CATEGORIA_NLP" in df_filtrado_dashboard.columns:
            df_cat = df_filtrado_dashboard["CATEGORIA_NLP"].value_counts().reset_index()
            df_cat.columns = ["Categoria", "Quantidade"]
            df_cat["Texto_Fmt"] = df_cat["Quantidade"].apply(lambda x: f"{x:,}".replace(",", "."))
            
            fig_cat = px.bar(df_cat, x="Quantidade", y="Categoria", orientation="h", text="Texto_Fmt", color="Quantidade", color_continuous_scale="Tealgrn")
            fig_cat.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, yaxis={'categoryorder':'total ascending'}, separators=',.')
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Dado de categoria NLP não disponível.")

    st.markdown("---")

    # ==========================================
    # 3. Gráficos: Linha Temporal & Pendências por Setor (Com Diagrama de Pareto Interativo)
    # ==========================================
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.markdown("### 📈 Evolução de Ocorrências ao Longo do Tempo (Linhas)")
        if col_data:
            try:
                df_temp = df_filtrado_dashboard.copy()
                df_temp["DATA_FORMATADA"] = pd.to_datetime(df_temp[col_data], errors='coerce').dt.to_period("M").astype(str)
                df_line = df_temp["DATA_FORMATADA"].value_counts().sort_index().reset_index()
                df_line.columns = ["Mês/Ano", "Total OS"]
                fig_line = px.line(df_line, x="Mês/Ano", y="Total OS", markers=True, line_shape="linear")
                fig_line.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, separators=',.')
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception:
                st.info("Formato de data inválido para montagem do gráfico temporal.")
        else:
            st.info("Coluna de data não encontrada na base.")

    with col_g4:
        st.markdown("### 🏢 Pendências por Setor / Máquinas")
        col_setor = "SETOR" if "SETOR" in df_filtrado_dashboard.columns else ("MÁQUINA" if "MÁQUINA" in df_filtrado_dashboard.columns else None)
        
        if col_setor:
            df_setor = df_filtrado_dashboard[col_setor].value_counts().head(10).reset_index()
            df_setor.columns = ["Setor/Máquina", "Quantidade"]
            df_setor["Texto_Fmt"] = df_setor["Quantidade"].apply(lambda x: f"{x:,}".replace(",", "."))
            
            fig_horiz = px.bar(
                df_setor, 
                y="Setor/Máquina", 
                x="Quantidade", 
                orientation="h", 
                text="Texto_Fmt", 
                color="Quantidade", 
                color_continuous_scale="Blues"
            )
            fig_horiz.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), 
                height=350, 
                yaxis={'categoryorder':'total ascending'}, 
                separators=',.'
            )
            
            # Gráfico de barras com suporte a cliques
            evento_grafico = st.plotly_chart(
                fig_horiz, 
                width='stretch', 
                on_select="rerun", 
                key="grafico_barras_setor_pareto"
            )
            
            # Captura o clique na barra
            try:
                selection_points = evento_grafico.get("selection", {}).get("points", [])
                if selection_points:
                    st.session_state["maquina_pareto_ativa"] = selection_points[0].get("y")
            except Exception:
                pass

            # Alternativa via menu suspenso para garantir funcionamento imediato
            lista_maquinas_pareto = ["-- Ou selecione a máquina aqui --"] + sorted(df_filtrado_dashboard[col_setor].dropna().unique().astype(str))
            maquina_escolhida = st.selectbox(
                "📊 Seleção rápida para o Diagrama de Pareto:",
                options=lista_maquinas_pareto,
                key="select_maquina_pareto_direto"
            )
            
            if maquina_escolhida and maquina_escolhida != "-- Ou selecione a máquina aqui --":
                st.session_state["maquina_pareto_ativa"] = maquina_escolhida

        else:
            st.info("Coluna de Setor ou Máquina não encontrada.")

    # ==========================================
    # EXIBIÇÃO DO DIAGRAMA DE PARETO
    # ==========================================
    if "maquina_pareto_ativa" in st.session_state and st.session_state["maquina_pareto_ativa"]:
        maq_ativa = st.session_state["maquina_pareto_ativa"]
        st.markdown("---")
        
        col_t1, col_t2 = st.columns([4, 1])
        with col_t1:
            st.markdown(f"### 📊 Diagrama de Pareto de Falhas — Máquina: `{maq_ativa}`")
        with col_t2:
            if st.button("❌ Fechar Pareto", key="btn_fechar_pareto"):
                st.session_state["maquina_pareto_ativa"] = None
                st.rerun()

        col_maq_filtro = None
        for c in ["MÁQUINA", "MAQUINA", "MÁQUINAS", "MAQUINAS", "SETOR"]:
            if c in df_filtrado_dashboard.columns:
                col_maq_filtro = c
                break

        if col_maq_filtro:
            df_pareto_base = df_filtrado_dashboard[df_filtrado_dashboard[col_maq_filtro].astype(str) == str(maq_ativa)]
            
            col_falha = None
            for c in ["CATEGORIA_NLP", "FALHA", "PROBLEMA", coluna_comentario]:
                if c and c in df_pareto_base.columns:
                    col_falha = c
                    break
            
            if col_falha and not df_pareto_base.empty:
                df_pareto = df_pareto_base[col_falha].value_counts().reset_index()
                df_pareto.columns = ["Falha / Ocorrência", "Quantidade"]
                df_pareto["Acumulado_%"] = (df_pareto["Quantidade"].cumsum() / df_pareto["Quantidade"].sum()) * 100
                
                fig_pareto = px.bar(
                    df_pareto, 
                    x="Falha / Ocorrência", 
                    y="Quantidade", 
                    text="Quantidade", 
                    title=f"Análise de Pareto para {maq_ativa}"
                )
                fig_pareto.add_scatter(
                    x=df_pareto["Falha / Ocorrência"], 
                    y=df_pareto["Acumulado_%"], 
                    mode="lines+markers", 
                    name="Acumulado %", 
                    yaxis="y2"
                )
                fig_pareto.update_layout(
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(title="Quantidade de Falhas"),
                    yaxis2=dict(title="Acumulado (%)", overlaying="y", side="right", range=[0, 105]),
                    height=400,
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                st.plotly_chart(fig_pareto, width='stretch')
            else:
                st.warning(f"Não há dados detalhados suficientes de falhas para a máquina '{maq_ativa}'.")

elif st.session_state.aba_ativa == "busca":
    st.subheader("🔍 Pesquisa Textual & Geração de Relatórios por IA")
    
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        termo = st.text_input("Digite um termo para pesquisar nas observações (ex: ar condicionado, vazamento, bomba):", value=st.session_state.termo_pesquisa)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔎 Buscar"):
            st.session_state.termo_pesquisa = termo
            limpar_pesquisa()

    if st.session_state.termo_pesquisa:
        termo_busca = st.session_state.termo_pesquisa.lower().strip()
        df_resultado = df[df["TEXTO_BUSCA"].str.contains(termo_busca, na=False)]
        
        st.markdown(f"**Registros encontrados:** {len(df_resultado):,} de {len(df):,}".replace(",", "."))
        
        if len(df_resultado) > 0:
            col_r1, col_r2 = st.columns([1, 2])
            
            with col_r1:
                st.markdown("#### 📄 Ações com a Seleção")
                if st.button("🤖 Gerar Relatório Executivo com IA"):
                    with st.spinner("Analisando dados e gerando relatório executivo via Groq..."):
                        relatorio = gerar_relatorio_ia(termo_busca, df_resultado[coluna_comentario].tolist(), groq_api_key_input)
                        st.session_state.relatorio_gerado = relatorio
                        st.session_state.termo_relatorio = termo_busca

            if "relatorio_gerado" in st.session_state and st.session_state.get("termo_relatorio") == termo_busca:
                st.markdown("---")
                st.markdown(f"### 📋 Relatório IA para '{termo_busca.upper()}'")
                st.markdown(st.session_state.relatorio_gerado)
                
                serie_tabela = extrair_dados_tabela_markdown(st.session_state.relatorio_gerado)
                if serie_tabela is not None and not serie_tabela.empty:
                    st.markdown("#### 📊 Gráfico extraído do Relatório")
                    fig_rel = px.bar(x=serie_tabela.index, y=serie_tabela.values, labels={'x': 'Área/Problema', 'y': 'Frequência'})
                    fig_rel.update_layout(separators=',.')
                    st.plotly_chart(fig_rel, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📋 Listagem das OS Encontradas")
            cols_exibir = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "MÁQUINA", "CATEGORIA_NLP", coluna_comentario] if c in df_resultado.columns]
            st.dataframe(df_resultado[cols_exibir], use_container_width=True)
        else:
            st.warning("Nenhuma Ordem de Serviço encontrada com o termo informado.")

elif st.session_state.aba_ativa == "nlp":
    st.subheader("🔍 Detalhamento por Categoria e Equipamentos")

    if "CATEGORIA_NLP" in df.columns:
        categorias_disponiveis = ["-- Selecione uma categoria --", "Todas"] + sorted(df["CATEGORIA_NLP"].dropna().unique())
        
        cat_selecionada = st.selectbox(
            "Selecione uma Categoria para abrir e ver os itens:", 
            options=categorias_disponiveis, 
            index=0
        )
        
        if cat_selecionada and cat_selecionada != "-- Selecione uma categoria --":
            if cat_selecionada == "Todas":
                df_cat = df.copy()
                st.write(f"Exibindo registros de **Todas** as categorias: **{len(df_cat):,}**")
            else:
                df_cat = df[df["CATEGORIA_NLP"] == cat_selecionada]
                st.write(f"Total de registros na categoria **{cat_selecionada}**: **{len(df_cat):,}**")
            
            col_maq = None
            for c in ["MÁQUINAS", "MAQUINAS", "MÁQUINA", "MAQUINA"]:
                if c in df.columns:
                    col_maq = c
                    break
            
            if col_maq:
                st.markdown(f"##### ⚙️ Equipamentos / Ativos relacionados (Coluna: `{col_maq}`)")
                
                df_ranking_maq = df_cat[col_maq].value_counts().reset_index()
                df_ranking_maq.columns = ["Equipamento / Máquina", "Quantidade"]
                
                evento_tabela = st.dataframe(
                    df_ranking_maq, 
                    use_container_width=True, 
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun",
                    key=f"tabela_ranking_{cat_selecionada}"
                )
                
                linhas_selecionadas = evento_tabela.get("selection", {}).get("rows", [])
                
                maquina_via_tabela = None
                if linhas_selecionadas:
                    idx_linha = linhas_selecionadas[0]
                    maquina_via_tabela = str(df_ranking_maq.iloc[idx_linha]["Equipamento / Máquina"])

                st.markdown("---")
                st.markdown("#### 📋 Ordens de Serviço Específicas por Máquina/Equipamento")
                
                lista_maquinas_cat = ["-- Selecione uma Máquina/Equipamento --"] + sorted(df_cat[col_maq].dropna().unique().astype(str))
                
                index_inicial = 0
                if maquina_via_tabela and maquina_via_tabela in lista_maquinas_cat:
                    index_inicial = lista_maquinas_cat.index(maquina_via_tabela)

                maquina_selecionada = st.selectbox(
                    "Selecione uma Máquina/Equipamento para ver as Ordens de Serviço detalhadas:", 
                    lista_maquinas_cat,
                    index=index_inicial,
                    key=f"select_maq_{cat_selecionada}"
                )
                
                maquina_ativa = maquina_via_tabela if maquina_via_tabela else maquina_selecionada

                if maquina_ativa and maquina_ativa != "-- Selecione uma Máquina/Equipamento --":
                    df_os_maquina = df_cat[df_cat[col_maq].astype(str) == maquina_ativa]
                    
                    st.info(f"Exibindo **{len(df_os_maquina)}** Ordem(ns) de Serviço para o equipamento: **{maquina_ativa}**")
                    
                    col_exec_prevista = None
                    for c in ["EXEC. PREVISTA", "EXEC_PREVISTA", "DATA EXEC PREVISTA", "PREVISÃO", "PREVISAO"]:
                        if c in df.columns:
                            col_exec_prevista = c
                            break
                    
                    candidatos_cols = ["CÓDIGO", "ABERTO EM", col_exec_prevista, "STATUS", "SETOR", col_maq, coluna_comentario, "CUSTO COM MATERIAIS"]
                    cols_mostrar = [c for c in candidatos_cols if c and c in df_os_maquina.columns]
                    
                    st.dataframe(df_os_maquina[cols_mostrar], use_container_width=True, hide_index=True)
                else:
                    st.info("👈 Clique em uma linha na tabela acima ou selecione um equipamento no menu para carregar as Ordens de Serviço específicas.")

elif st.session_state.aba_ativa == "chat":
    st.subheader("💬 Chat Inteligente sobre a Base de Dados")
    st.write("Faça perguntas diretas em linguagem natural sobre qualquer informação da planilha.")
    
    for msg in st.session_state.mensagens_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    pergunta = st.chat_input("Ex: Qual o setor com mais chamados abertos? Quantos vazamentos tivemos?")
    
    if pergunta:
        st.session_state.mensagens_chat.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.write(pergunta)
        
        with st.chat_message("assistant"):
            with st.spinner("Consultando dados da planilha via Groq..."):
                resposta = responder_pergunta_livre_com_todo_arquivo(pergunta, df, groq_api_key_input)
                st.write(resposta)
                st.session_state.mensagens_chat.append({"role": "assistant", "content": resposta})
