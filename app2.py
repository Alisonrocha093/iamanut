import os
import io
import re
import json
import unicodedata
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots

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
    /* ---------- Tema escuro executivo ---------- */
    :root {
        --bg: #0B1220; --surface: #111A2E; --surface-2: #16213A;
        --border: #243049; --text: #E2E8F0; --text-2: #CBD5E1; --muted: #94A3B8;
        --accent: #60A5FA;
    }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #0E1628; border-right: 1px solid var(--border); }
    h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }
    p, li, label, [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] { color: var(--text-2); }
    hr { border-color: var(--border) !important; }
    [data-baseweb="input"], [data-baseweb="input"] input, [data-baseweb="select"] > div,
    [data-baseweb="textarea"], textarea {
        background-color: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important;
    }
    [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
    [data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }

    /* ---------- Componentes que ficavam brancos ---------- */
    button p { color: inherit !important; }
    [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"],
    [data-testid="stBaseButton-minimal"], .stDownloadButton button {
        background: var(--surface-2) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
    }
    [data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
        border-color: var(--accent) !important; color: #fff !important;
    }
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
        background: #2563EB !important; color: #fff !important; border: 1px solid #3B82F6 !important;
    }
    [data-testid="stAlert"] {
        background: rgba(96,165,250,.12) !important; border: 1px solid rgba(96,165,250,.30); border-radius: 10px;
    }
    [data-testid="stAlert"] * { color: var(--text) !important; }
    [data-baseweb="select"] * { color: var(--text) !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul {
        background: var(--surface) !important;
    }
    [data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: var(--text) !important; background: transparent !important; }
    [data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--surface-2) !important; }
    input::placeholder, textarea::placeholder { color: #7C8AA5 !important; }
    [data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; border: 1px dashed var(--border) !important; }
    [data-testid="stFileUploaderDropzone"] * { color: var(--text-2) !important; }
    [data-testid="stForm"] { border-color: var(--border) !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--text) !important; }
    code { background: var(--surface-2) !important; color: #BFDBFE !important; }

    /* ---------- Cabeçalho da página ---------- */
    .main-header { font-size: 2.2rem; color: #F1F5F9; font-weight: 700; margin-bottom: 0px; letter-spacing: -.01em; }
    .sub-header  { font-size: 1.1rem; color: var(--muted); margin-bottom: 20px; }

    /* ---------- Faixa executiva (hero) ---------- */
    .hero {
        background: linear-gradient(135deg, #0F2A5C 0%, #1D4ED8 100%);
        padding: 22px 28px; border-radius: 14px; color: #fff;
        display: flex; justify-content: space-between; align-items: center; gap: 16px;
        margin-bottom: 18px; border: 1px solid rgba(96,165,250,.35);
        box-shadow: 0 6px 22px rgba(0,0,0,.45);
    }
    .hero-title { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
    .hero-sub   { font-size: .95rem; opacity: .85; margin-top: 4px; }
    .hero-badge {
        background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.35);
        padding: 8px 16px; border-radius: 999px; font-weight: 600; white-space: nowrap;
    }

    /* ---------- Cards de KPI ---------- */
    .kpi-card {
        background: var(--surface); border: 1px solid var(--border); border-left: 5px solid var(--accent);
        border-radius: 12px; padding: 14px 16px; height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,.35);
        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,.5); background: var(--surface-2); }
    .kpi-title { font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
    .kpi-value { font-size: 1.65rem; font-weight: 700; margin: 4px 0 2px; line-height: 1.15; }
    .kpi-sub   { font-size: .78rem; color: var(--muted); min-height: 1.1em; }
    .kpi-delta { font-weight: 600; margin-left: 4px; }

    /* ---------- Seções e títulos de gráficos ---------- */
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #F1F5F9;
        margin: 28px 0 6px; padding-bottom: 6px; border-bottom: 2px solid var(--border);
    }
    .chart-title { font-size: .92rem; font-weight: 600; color: var(--text); margin: 8px 0 0; }
    .chart-note  { font-size: .78rem; color: #7C8AA5; margin: 0 0 4px; }

    /* ---------- Destaques automáticos ---------- */
    .insights { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 4px; }
    .chip {
        background: rgba(96,165,250,.12); border: 1px solid rgba(96,165,250,.35); color: #BFDBFE;
        padding: 6px 12px; border-radius: 999px; font-size: .82rem;
    }

    /* ---------- Barra lateral compacta ---------- */
    [data-testid="stSidebar"] > div:first-child {
        height: 100vh; padding-top: 0.5rem; padding-bottom: 0.5rem;
        display: flex; flex-direction: column; justify-content: flex-start;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.2rem !important; }
    [data-testid="stSidebar"] hr { margin-top: 0.4rem !important; margin-bottom: 0.4rem !important; }

    footer { visibility: hidden; }

    /* ---------- Responsivo ---------- */
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem !important; }
        .sub-header  { font-size: 1rem !important; margin-bottom: 10px; }
        .hero { flex-direction: column; align-items: flex-start; }
        .hero-title { font-size: 1.25rem; }
        .kpi-value { font-size: 1.35rem; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.25rem !important; }
        h3 { font-size: 1.1rem !important; }
        [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
        .stSelectbox, .stTextInput, .stFileUploader { width: 100% !important; }
        .stDataFrame { width: 100% !important; overflow-x: auto !important; }
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
        col_esq, col_centro, col_dir = st.columns([1, 1.2, 1])

        with col_centro:
            st.markdown("## 🔐 Autenticação")
            st.markdown("Insira suas credenciais para acessar o painel.")

            with st.form("form_login"):
                usuario_input = st.text_input("Usuário").strip()
                senha_input = st.text_input("Senha", type="password")
                submit = st.form_submit_button("Entrar", use_container_width=True)

                if submit:
                    usuarios_cadastrados = {}
                    try:
                        if hasattr(st, "secrets") and "SENHA_APP" in st.secrets:
                            usuarios_cadastrados = dict(st.secrets["SENHA_APP"])
                    except Exception:
                        pass

                    if usuario_input in usuarios_cadastrados and senha_input == usuarios_cadastrados[usuario_input]:
                        st.session_state.autenticado = True
                        st.session_state.usuario_atual = usuario_input
                        st.rerun()
                    else:
                        st.error("❌ Usuário ou senha incorretos.")
        return False
    return True


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
            "distribuição elétrica", "distribuicao eletrica",
            "subestação", "transformador", "poste", "cabo pp"
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
            "dps", "proteção contra descargas",
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


def gerar_relatorio_ia(termo: str, df_filtrado: pd.DataFrame, api_key: str) -> str:
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral."
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}"

    modelo_ativo = obter_modelo_ativo(client, api_key)

    col_obs = "OBSERVAÇÃO ABERTURA" if "OBSERVAÇÃO ABERTURA" in df_filtrado.columns else df_filtrado.columns[0]

    col_maq = None
    for c in df_filtrado.columns:
        if c in ["MÁQUINAS", "MAQUINAS", "MÁQUINA", "MAQUINA", "EQUIPAMENTO"]:
            col_maq = c
            break

    observacoes_reais = df_filtrado[col_obs].dropna().astype(str).str.strip()
    observacoes_reais = observacoes_reais[observacoes_reais.str.lower() != "n/a"]

    if len(observacoes_reais) == 0:
        observacoes_reais = df_filtrado["TEXTO_BUSCA"].dropna().astype(str).str.strip()

    df_obs = pd.DataFrame({"obs": observacoes_reais.str.lower()})

    if col_maq:
        df_obs["maquina"] = df_filtrado.loc[df_obs.index, col_maq].astype(str).str.strip().str.upper()
    else:
        df_obs["maquina"] = "N/A"

    top_falhas = df_obs["obs"].value_counts().head(15).reset_index()
    top_falhas.columns = ["Descricao_Falha", "Frequencia"]

    ranking_texto = "\n".join([f"- {row['Descricao_Falha']} (Ocorrências: {row['Frequencia']})" for _, row in top_falhas.iterrows()])
    top_maquinas = df_obs["maquina"].value_counts().head(10).to_dict() if col_maq else "Não aplicável"

    prompt = f"""Você é um especialista em manutenção de Ordens de Serviço (OS).
Foram encontrados {len(df_filtrado)} registros no total para a pesquisa/máquina '{termo.upper()}'.
As principais máquinas/equipamentos envolvidos nestes registros são: {top_maquinas}

Abaixo está o ranking real dos relatos e descrições de ocorrências registrados nas observações das OS:
{ranking_texto}

Escreva um relatório executivo detalhado em português contendo:
1. Resumo dos Principais Problemas Relatados (Inclua obrigatoriamente uma tabela em Markdown com colunas 'Descrição da Falha / Problema' e 'Frequência')
2. Análise específica das Máquinas/Equipamentos mais afetados por esses chamados
3. Padrões ou Causas Recorrentes baseados nos relatos reais
4. Recomendações e Ações Preventivas direcionadas a esses equipamentos e problemas"""

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


# ==========================================
# CHAT INTELIGENTE (PLANO DE CONSULTA + PANDAS + BUSCA PRECISA)
# ==========================================
def _norm(txt) -> str:
    """minúsculas e sem acentos (para 'manutenção' casar com 'manutencao')."""
    t = unicodedata.normalize("NFKD", str(txt))
    return "".join(ch for ch in t if not unicodedata.combining(ch)).lower().strip()


def obter_modelo_chat(client) -> str:
    """Prefere um modelo maior (melhor para planejar consultas); cai para o 8b."""
    try:
        ids = [m.id for m in client.models.list().data]
        for preferido in ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"]:
            if preferido in ids:
                return preferido
    except Exception:
        pass
    return obter_modelo_ativo(client, "ok")


def preparar_df_chat(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas auxiliares (datas, custos, status, tempo) e um texto completo pesquisável."""
    d = df.copy()

    for col in d.columns:
        if col in ("ABERTO EM", "FECHADO EM", "FINALIZAÇÃO", "EXEC. PREVISTA", "DATA", "DATA FECHAMENTO"):
            d[col] = pd.to_datetime(d[col], errors="coerce")

    # TEXTO_COMPLETO: junta TODAS as colunas de texto (setor, local, máquina, observações...), sem acentos
    cols_txt = [c for c in d.columns
                if c not in ("TEXTO_BUSCA", "TEXTO_COMPLETO")
                and not pd.api.types.is_numeric_dtype(d[c])
                and not pd.api.types.is_datetime64_any_dtype(d[c])]
    texto = pd.Series("", index=d.index)
    for c in cols_txt:
        texto = texto + " " + d[c].astype(str)
    d["TEXTO_COMPLETO"] = (texto.str.normalize("NFKD")
                           .str.encode("ascii", "ignore").str.decode("ascii").str.lower())

    if "ABERTO EM" in d.columns:
        d["ANO_MES"] = d["ABERTO EM"].dt.to_period("M").astype(str).replace("NaT", np.nan)
        d["ANO"] = d["ABERTO EM"].dt.year
        d["MES"] = d["ABERTO EM"].dt.month

    if "STATUS" in d.columns:
        def _grupo(s):
            s = str(s).lower()
            if any(t in s for t in ["concluíd", "concluid", "fechad", "encerrad", "finalizad"]):
                return "Concluídas"
            if "cancelad" in s:
                return "Canceladas"
            if any(t in s for t in ["program", "planejad"]):
                return "Programadas"
            return "Pendentes / Abertas"
        d["STATUS_GRUPO"] = d["STATUS"].apply(_grupo)

    if "CUSTO COM MATERIAIS" in d.columns:
        d["CUSTO_MATERIAIS_NUM"] = limpar_coluna_monetaria(d["CUSTO COM MATERIAIS"]).values
    if "MÃO DE OBRA EXTERNA" in d.columns:
        d["MAO_DE_OBRA_NUM"] = limpar_coluna_monetaria(d["MÃO DE OBRA EXTERNA"]).values
    if "CUSTO_MATERIAIS_NUM" in d.columns or "MAO_DE_OBRA_NUM" in d.columns:
        d["CUSTO_TOTAL_NUM"] = d.get("CUSTO_MATERIAIS_NUM", 0) + d.get("MAO_DE_OBRA_NUM", 0)

    col_ini = "EXEC. PREVISTA" if "EXEC. PREVISTA" in d.columns else None
    col_fim = "FINALIZAÇÃO" if "FINALIZAÇÃO" in d.columns else ("FECHADO EM" if "FECHADO EM" in d.columns else None)
    if col_ini and col_fim:
        dias = (d[col_fim] - d[col_ini]).dt.total_seconds() / 86400
        d["TEMPO_ATENDIMENTO_DIAS"] = dias.where(dias >= 0)

    return d


def descrever_esquema(d: pd.DataFrame) -> str:
    """Resumo das colunas (nome, tipo e valores possíveis) para a IA planejar a consulta."""
    linhas = []
    for col in d.columns:
        if col in ("TEXTO_BUSCA", "TEXTO_COMPLETO"):
            continue
        serie = d[col]
        if pd.api.types.is_datetime64_any_dtype(serie):
            tipo = "data"
            exemplo = f"de {serie.min():%d/%m/%Y} até {serie.max():%d/%m/%Y}" if serie.notna().any() else "vazia"
        elif pd.api.types.is_numeric_dtype(serie):
            tipo = "número"
            exemplo = f"min {serie.min():.2f} | max {serie.max():.2f}" if serie.notna().any() else "vazia"
        else:
            tipo = "texto"
            nun = serie.nunique(dropna=True)
            if nun <= 25:
                exemplo = "valores: " + ", ".join(map(str, serie.dropna().unique()[:25]))
            else:
                exemplo = f"{nun} valores distintos, ex.: " + ", ".join(map(str, serie.dropna().unique()[:3]))
        linhas.append(f"- {col} ({tipo}): {str(exemplo)[:300]}")
    return "\n".join(linhas)


def montar_estatisticas_globais(d: pd.DataFrame) -> str:
    partes = [f"Total de registros: {len(d)}"]

    if "ABERTO EM" in d.columns and d["ABERTO EM"].notna().any():
        partes.append(f"Período: {d['ABERTO EM'].min():%d/%m/%Y} até {d['ABERTO EM'].max():%d/%m/%Y}")

    def top(col, n=10):
        if col in d.columns:
            return d[col].value_counts().head(n).to_dict()
        return None

    for rotulo, col in [("Status", "STATUS"), ("Grupo de status", "STATUS_GRUPO"),
                        ("Categorias NLP", "CATEGORIA_NLP"), ("Setores", "SETOR"),
                        ("Máquinas", "MÁQUINA"), ("Máquinas", "MÁQUINAS"),
                        ("Tipos de manutenção", "TIPO DE MANUTENÇÃO"), ("Tipos", "TIPO")]:
        r = top(col)
        if r:
            partes.append(f"Top {rotulo}: {r}")

    if "CUSTO_MATERIAIS_NUM" in d.columns:
        partes.append(f"Custo total de materiais: {formatar_moeda_br(d['CUSTO_MATERIAIS_NUM'].sum())}")
    if "MAO_DE_OBRA_NUM" in d.columns:
        partes.append(f"Custo total de mão de obra externa: {formatar_moeda_br(d['MAO_DE_OBRA_NUM'].sum())}")
    if "CUSTO_TOTAL_NUM" in d.columns:
        partes.append(f"Custo total geral: {formatar_moeda_br(d['CUSTO_TOTAL_NUM'].sum())}")
    if "TEMPO_ATENDIMENTO_DIAS" in d.columns and d["TEMPO_ATENDIMENTO_DIAS"].notna().any():
        partes.append(f"Tempo médio de atendimento: {d['TEMPO_ATENDIMENTO_DIAS'].mean():.1f} dias")
    if "ANO_MES" in d.columns:
        partes.append(f"OS por mês: {d['ANO_MES'].value_counts().sort_index().to_dict()}")

    return "\n".join(partes)


def _extrair_json(texto: str) -> dict:
    try:
        texto = re.sub(r"```(?:json)?", "", texto).strip()
        ini, fim = texto.find("{"), texto.rfind("}")
        return json.loads(texto[ini:fim + 1])
    except Exception:
        return {}


def planejar_consulta(client, modelo, pergunta, esquema, historico) -> dict:
    """A IA converte a pergunta em um plano estruturado (não executa nada)."""
    hist_txt = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in (historico or [])[-4:])
    prompt = f"""Hoje é {pd.Timestamp.today():%d/%m/%Y}.
Você converte perguntas sobre uma base de Ordens de Serviço (OS) em um plano de consulta JSON.

COLUNAS DISPONÍVEIS:
{esquema}

CONVERSA RECENTE (para entender perguntas de continuação):
{hist_txt or '(vazia)'}

PERGUNTA ATUAL: {pergunta}

Responda APENAS com JSON neste formato (use somente colunas listadas acima):
{{
  "modo": "listar",
  "busca_todas": [["ar condicionado", "ar-condicionado", "split"], ["sala de manutencao"]],
  "filtros": [{{"coluna": "STATUS_GRUPO", "op": "igual", "valor": "Concluídas"}}],
  "ordenar_por": "FINALIZAÇÃO",
  "ordem": "desc",
  "limite": 3,
  "agrupar_por": [],
  "metrica": "contagem",
  "coluna_metrica": null
}}

Regras:
- modo "listar": a pergunta pede registros específicos (última/primeira/mais recente, quais OS, quando foi, o que foi feito, detalhes de um equipamento/local). Retorna as OS ordenadas.
- modo "agregar": a pergunta pede contagem, soma, média, ranking ou evolução (use agrupar_por, metrica, coluna_metrica).
- busca_todas: lista de CONCEITOS; TODOS precisam aparecer no texto da OS (E). Cada conceito é uma lista de sinônimos (OU). Escreva sem acentos e em minúsculas. Separe equipamento/problema do LOCAL/SETOR em conceitos diferentes (ex.: [["ar condicionado","split"],["sala de manutencao"]]). Use só as palavras essenciais.
- Use termos curtos e simples, como o usuário digitaria na pesquisa textual (ex.: "ar condicionado", "vazamento", "bomba").
- Perguntas "feita/realizada/executada/concluída": adicione filtro STATUS_GRUPO igual "Concluídas". "pendente/aberta": "Pendentes / Abertas".
- "última/mais recente": ordenar_por = coluna de data de conclusão (FINALIZAÇÃO ou FECHADO EM, se existir; senão ABERTO EM), ordem "desc", limite 3. "primeira/mais antiga": ordem "asc".
- op permitidos: igual, diferente, contem, em, maior, menor, entre (valor = [inicio, fim]; datas AAAA-MM-DD).
- metrica: contagem, soma, media, max ou min; se não for contagem, informe coluna_metrica numérica (CUSTO_TOTAL_NUM, CUSTO_MATERIAIS_NUM, MAO_DE_OBRA_NUM, TEMPO_ATENDIMENTO_DIAS).
- agrupar_por: no máximo 2 colunas (ANO_MES para evolução mensal, STATUS_GRUPO, CATEGORIA_NLP, SETOR etc.).
- Pergunta conceitual (sem dados): modo "agregar" com tudo vazio."""

    kwargs = dict(messages=[{"role": "user", "content": prompt}], model=modelo,
                  temperature=0.0, max_tokens=600)
    try:
        resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            return {}
    return _extrair_json(resp.choices[0].message.content)


def _aplicar_filtro(d: pd.DataFrame, f: dict) -> pd.DataFrame:
    col, op, val = f.get("coluna"), f.get("op"), f.get("valor")
    if col not in d.columns or val is None:
        return d
    s = d[col]

    if op in ("igual", "diferente", "contem", "em"):
        txt = s.astype(str).map(_norm)
        if op == "igual":
            m = txt == _norm(val)
        elif op == "diferente":
            m = txt != _norm(val)
        elif op == "contem":
            m = txt.str.contains(re.escape(_norm(val)), na=False)
        else:
            vals = val if isinstance(val, list) else [val]
            m = txt.isin([_norm(v) for v in vals])
        return d[m]

    is_data = pd.api.types.is_datetime64_any_dtype(s)
    conv = (lambda x: pd.to_datetime(x, errors="coerce")) if is_data else (lambda x: pd.to_numeric(x, errors="coerce"))
    serie = s if is_data else pd.to_numeric(s, errors="coerce")

    if op in ("maior", "menor"):
        ref = conv(val)
        if pd.isna(ref):
            return d
        return d[serie > ref] if op == "maior" else d[serie < ref]
    if op == "entre" and isinstance(val, list) and len(val) == 2:
        a, b = conv(val[0]), conv(val[1])
        if pd.isna(a) or pd.isna(b):
            return d
        return d[(serie >= a) & (serie <= b)]
    return d


def _colunas_exibicao(d: pd.DataFrame) -> list:
    base = ["CÓDIGO", "ABERTO EM", "EXEC. PREVISTA", "FINALIZAÇÃO", "FECHADO EM", "STATUS",
            "TIPO DE MANUTENÇÃO", "TIPO", "SETOR", "MÁQUINA", "MÁQUINAS", "CATEGORIA_NLP", "CUSTO_TOTAL_NUM"]
    extras = [c for c in d.columns
              if any(k in c for k in ("OBSERV", "DESCR", "SOLU", "SERVI", "LOCAL"))
              and c not in base and c not in ("TEXTO_BUSCA", "TEXTO_COMPLETO")]
    return [c for c in base + extras if c in d.columns]


def registros_para_texto(d: pd.DataFrame, limite_txt: int = 400) -> str:
    """Formata cada OS como uma linha 'campo: valor | campo: valor' (datas em dd/mm/aaaa)."""
    linhas = []
    for _, row in d.iterrows():
        partes = []
        for c in d.columns:
            v = row[c]
            if pd.isna(v) or str(v).strip().upper() in ("N/A", "NAN", "NAT", ""):
                continue
            if isinstance(v, pd.Timestamp):
                v = v.strftime("%d/%m/%Y")
            elif isinstance(v, float):
                v = f"{v:.2f}"
            partes.append(f"{c}: {str(v)[:limite_txt]}")
        linhas.append("- " + " | ".join(partes))
    return "\n".join(linhas) if linhas else "(nenhum registro)"


STOPWORDS_PT = set("""
a o as os um uma uns umas de do da dos das no na nos nas em para por com sem sobre entre ate ao aos que qual quais
quanto quantos quantas quando onde como foi foram fez feita feito feitas feitos fazer ser esta estao tem temos tinha
ultima ultimo ultimas ultimos primeira primeiro mais menos maior menor ja ainda todas todos toda todo me mostre
mostra diga fale sobre existe existem houve manutencao manutencoes os.
""".split())


def extrair_termos_fallback(pergunta: str) -> list:
    """Sem IA: pega as palavras relevantes da pergunta (como alguém digitaria na pesquisa textual)."""
    palavras = re.findall(r"[a-zA-ZÀ-ÿ0-9\-]+", _norm(pergunta))
    termos = [p for p in palavras if len(p) > 2 and p not in STOPWORDS_PT]
    return termos[:4]


def perfil_conjunto(d: pd.DataFrame) -> str:
    """Resumo do conjunto de OS encontradas (para a IA analisar padrões, recorrências e custos)."""
    if d.empty:
        return "(nenhuma OS)"
    p = [f"Total de OS encontradas: {len(d)}"]

    if "ABERTO EM" in d.columns and d["ABERTO EM"].notna().any():
        p.append(f"Período das OS: {d['ABERTO EM'].min():%d/%m/%Y} até {d['ABERTO EM'].max():%d/%m/%Y}")

    for rotulo, col in [("Status", "STATUS"), ("Setores", "SETOR"), ("Máquinas", "MÁQUINA"),
                        ("Máquinas", "MÁQUINAS"), ("Tipos de manutenção", "TIPO DE MANUTENÇÃO"),
                        ("Tipos", "TIPO"), ("Categorias", "CATEGORIA_NLP")]:
        if col in d.columns:
            p.append(f"{rotulo}: {d[col].value_counts().head(8).to_dict()}")

    if "ANO_MES" in d.columns:
        p.append(f"OS por mês: {d['ANO_MES'].value_counts().sort_index().tail(18).to_dict()}")

    col_obs = "OBSERVAÇÃO ABERTURA" if "OBSERVAÇÃO ABERTURA" in d.columns else None
    if col_obs:
        obs = d[col_obs].astype(str).str.strip().str.lower().str.slice(0, 90)
        obs = obs[~obs.isin(["n/a", "nan", ""])]
        if len(obs):
            p.append(f"Relatos mais recorrentes: {obs.value_counts().head(10).to_dict()}")

    if "CUSTO_TOTAL_NUM" in d.columns:
        p.append(f"Custo total (materiais + mão de obra externa): {formatar_moeda_br(d['CUSTO_TOTAL_NUM'].sum())}")
    if "TEMPO_ATENDIMENTO_DIAS" in d.columns and d["TEMPO_ATENDIMENTO_DIAS"].notna().any():
        p.append(f"Tempo médio de atendimento: {d['TEMPO_ATENDIMENTO_DIAS'].mean():.1f} dias")
    return "\n".join(p)


COLUNAS_DATA_PADRAO = ["FINALIZAÇÃO", "FECHADO EM", "DATA FECHAMENTO", "ABERTO EM", "DATA"]


def executar_plano(d: pd.DataFrame, plano: dict):
    """Executa o plano com pandas. Retorna (df_filtrado, tabela_resultado, descricao, modo, aviso)."""
    desc, aviso = [], ""
    modo = plano.get("modo") if plano.get("modo") in ("listar", "agregar") else "agregar"

    # ---- 1) Busca de texto: conceitos em E, sinônimos em OU (sem acentos) ----
    grupos = plano.get("busca_todas") or []
    if plano.get("busca_texto"):
        grupos = list(grupos) + [plano["busca_texto"]]
    padroes, rotulos = [], []
    for g in grupos:
        if isinstance(g, str):
            g = [g]
        termos = [_norm(x) for x in g if str(x).strip()]
        if termos:
            padroes.append("|".join(re.escape(t) for t in termos))
            rotulos.append(" / ".join(termos))

    col_txt = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in d.columns else "TEXTO_BUSCA"
    d0 = d
    for pad, rot in zip(padroes, rotulos):
        antes = len(d)
        d = d[d[col_txt].str.contains(pad, na=False)]
        desc.append(f"contém [{rot}] ({antes} → {len(d)})")

    # Nenhuma OS tem TODOS os conceitos: informa e mostra apenas correspondência parcial (rotulada)
    if padroes and d.empty:
        contagens = {rot: int(d0[col_txt].str.contains(pad, na=False).sum()) for pad, rot in zip(padroes, rotulos)}
        aviso = (f"NENHUMA OS contém TODOS os conceitos pedidos {rotulos}. "
                 f"Quantidade de OS que contém cada conceito isoladamente: {contagens}.")
        score = sum(d0[col_txt].str.contains(pad, na=False).astype(int) for pad in padroes)
        melhor = int(score.max()) if len(score) else 0
        if melhor > 0:
            d = d0[score == melhor]
            aviso += (f" Os registros abaixo são CORRESPONDÊNCIA PARCIAL (contêm {melhor} de {len(padroes)} conceitos) "
                      f"e NÃO atendem a pergunta por completo.")

    # ---- 2) Filtros estruturados (se um filtro zerar o resultado, é ignorado e avisado) ----
    for f in plano.get("filtros", []) or []:
        novo = _aplicar_filtro(d, f)
        if novo.empty and not d.empty:
            aviso += f" O filtro {f.get('coluna')} {f.get('op')} {f.get('valor')} não encontrou nenhuma OS e foi ignorado."
            desc.append(f"{f.get('coluna')} {f.get('op')} {f.get('valor')} (IGNORADO: zerava o resultado)")
            continue
        desc.append(f"{f.get('coluna')} {f.get('op')} {f.get('valor')} ({len(d)} → {len(novo)})")
        d = novo

    d = d.copy()
    try:
        limite = int(plano.get("limite") or (5 if modo == "listar" else 15))
    except Exception:
        limite = 5 if modo == "listar" else 15
    limite = max(1, min(limite, 30))
    ascendente = (plano.get("ordem") == "asc")

    # ---- 3a) Modo LISTAR: registros ordenados por data ----
    if modo == "listar":
        col_ord = plano.get("ordenar_por")
        candidatas = ([col_ord] if col_ord in d.columns else []) + [c for c in COLUNAS_DATA_PADRAO if c in d.columns]
        col_ord = next((c for c in candidatas
                        if pd.api.types.is_datetime64_any_dtype(d[c]) and d[c].notna().any()), None)
        if col_ord:
            d = d.sort_values(col_ord, ascending=ascendente, na_position="last")
            desc.append(f"ordenado por {col_ord} ({'crescente' if ascendente else 'decrescente'})")
        tabela = d[_colunas_exibicao(d)].head(max(limite, 8))
        return d, tabela, desc, modo, aviso

    # ---- 3b) Modo AGREGAR ----
    metrica = plano.get("metrica") or "contagem"
    colm = plano.get("coluna_metrica")
    grupos_col = [g for g in (plano.get("agrupar_por") or []) if g in d.columns][:2]

    if metrica != "contagem" and colm in d.columns:
        d[colm] = pd.to_numeric(d[colm], errors="coerce")
        func = {"soma": "sum", "media": "mean", "max": "max", "min": "min"}.get(metrica, "sum")
        if grupos_col:
            res = d.groupby(grupos_col)[colm].agg(func).reset_index().sort_values(colm, ascending=ascendente)
        else:
            res = pd.DataFrame({f"{metrica.upper()}_{colm}": [getattr(d[colm], func)()]})
    else:
        if grupos_col:
            res = d.groupby(grupos_col).size().reset_index(name="QUANTIDADE").sort_values("QUANTIDADE", ascending=ascendente)
        else:
            res = pd.DataFrame({"QUANTIDADE": [len(d)]})

    if grupos_col and grupos_col[0] == "ANO_MES":
        res = res.sort_values(grupos_col[0])
    return d, res.head(limite if grupos_col else 1), desc, modo, aviso


def buscar_linhas_relevantes(d: pd.DataFrame, pergunta: str, k: int = 15) -> pd.DataFrame:
    """TF-IDF: retorna as k OS mais parecidas com a pergunta (usado só no modo agregar)."""
    if d.empty:
        return d
    base = d["TEXTO_COMPLETO"] if "TEXTO_COMPLETO" in d.columns else d.iloc[:, 0].astype(str)
    try:
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(base.astype(str))
        q = vec.transform([_norm(pergunta)])
        scores = (X @ q.T).toarray().ravel()
        idx = [i for i in np.argsort(scores)[::-1][:k] if scores[i] > 0]
        return d.iloc[idx] if idx else d.head(k)
    except Exception:
        return d.head(k)


def responder_pergunta_livre_com_todo_arquivo(pergunta_usuario: str, df: pd.DataFrame,
                                              api_key: str, historico=None, termo_busca=None):
    """Pesquisa como a Pesquisa Textual e responde com inteligência.
    Retorna (resposta_texto, tabela_de_OS_encontradas_ou_None, resumo_da_busca)."""
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral.", None, ""
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}", None, ""

    modelo = obter_modelo_chat(client)

    try:
        d = preparar_df_chat(df)
        esquema = descrever_esquema(d)
        stats_globais = montar_estatisticas_globais(d)

        plano = planejar_consulta(client, modelo, pergunta_usuario, esquema, historico)

        if termo_busca and termo_busca.strip():
            # Termo manual: pesquisa exatamente como na aba Pesquisa Textual
            plano["busca_todas"] = [[termo_busca.strip()]]
            plano.pop("busca_texto", None)
            if plano.get("modo") not in ("listar", "agregar"):
                plano["modo"] = "listar"
        elif (not plano.get("busca_todas") and not plano.get("busca_texto") and not plano.get("filtros")
              and not plano.get("agrupar_por") and (plano.get("metrica") or "contagem") == "contagem"):
            # Plano vazio/falhou: usa as palavras da pergunta como termos de pesquisa
            kws = extrair_termos_fallback(pergunta_usuario)
            if kws:
                plano["busca_todas"] = [[k] for k in kws]
                plano["modo"] = "listar"

        d_filtrado, tabela, desc_filtros, modo, aviso = executar_plano(d, plano)
    except Exception as e:
        return f"❌ Erro ao processar os dados: {e}", None, ""

    busca_aplicada = any(x.startswith("contém") for x in desc_filtros)
    filtros_txt = "; ".join(desc_filtros) if desc_filtros else "nenhum"
    hist_txt = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in (historico or [])[-4:])

    # ---- Blocos de dados do prompt ----
    bloco_busca = f"""=== BUSCA REALIZADA (mesma lógica da Pesquisa Textual) ===
Critérios: {filtros_txt}
OS encontradas: {len(d_filtrado)}
{('ATENÇÃO: ' + aviso) if aviso else ''}"""

    bloco_perfil = ""
    if busca_aplicada:
        bloco_perfil = f"\n=== PERFIL DAS OS ENCONTRADAS ===\n{perfil_conjunto(d_filtrado)}\n"

    if modo == "listar":
        bloco_dados = f"""=== REGISTROS (ordenados; o PRIMEIRO é o mais recente quando a ordem é decrescente) ===
{registros_para_texto(tabela)}"""
        regras_extra = """- Para "última/mais recente", baseie-se no PRIMEIRO registro: cite código da OS, datas, status, setor/máquina e o que foi feito (observações). Para "primeira", use a ordem crescente.
- Para perguntas analíticas (problemas recorrentes, causas, custos, tendência, recomendações), use o PERFIL DAS OS ENCONTRADAS e os registros, identifique padrões e dê recomendações práticas.
- Mencione brevemente quantas OS foram encontradas no total."""
    else:
        amostra = buscar_linhas_relevantes(d_filtrado, pergunta_usuario, k=15)
        amostra_txt = (registros_para_texto(amostra[_colunas_exibicao(amostra)], limite_txt=250)
                       if len(amostra) else "(nenhuma OS correspondente)")
        bloco_dados = f"""=== RESULTADO EXATO DA CONSULTA (calculado com pandas; fonte mais confiável para números) ===
{tabela.to_string(index=False)}

=== OS MAIS RELEVANTES PARA A PERGUNTA ({len(amostra)} de {len(d_filtrado)}) ===
{amostra_txt}"""
        regras_extra = """- Use os números do "RESULTADO EXATO" para contagens, somas e médias; não recalcule nem invente valores.
- Depois dos números, acrescente uma breve análise (o que chama atenção, possíveis causas, recomendação) quando fizer sentido."""

    prompt = f"""Você é um analista sênior de gestão de manutenção. Responda SOMENTE com base nos dados abaixo.

=== ESTATÍSTICAS GLOBAIS DA BASE ===
{stats_globais}

{bloco_busca}
{bloco_perfil}
{bloco_dados}

=== CONVERSA RECENTE ===
{hist_txt or '(início da conversa)'}

REGRAS:
{regras_extra}
- Se houver ATENÇÃO dizendo que nenhuma OS atende todos os critérios, diga isso com clareza no início; só depois, se útil, cite os registros parciais como aproximação, deixando claro que NÃO são exatamente o que foi perguntado.
- Valores monetários em formato brasileiro (R$ 1.234,56).
- Se a informação não estiver nos dados, diga claramente que não foi possível encontrar. Nunca invente.
- Seja objetivo e use tabela Markdown quando houver ranking ou comparação.
- Responda em português.

PERGUNTA: {pergunta_usuario}"""

    evidencia = (d_filtrado[_colunas_exibicao(d_filtrado)].head(500)
                 if (busca_aplicada or modo == "listar") else tabela)
    resumo = f"{filtros_txt} → {len(d_filtrado)} OS" if desc_filtros else f"{len(d_filtrado)} OS"

    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo,
            temperature=0.15,
            max_tokens=1000,
        )
        return resp.choices[0].message.content, evidencia, resumo
    except Exception as e:
        return f"❌ Erro ao responder pergunta: {e}", None, resumo


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
# INTERFACE GRÁFICA (BARRA LATERAL)
# ==========================================
st.sidebar.title("⚙️ Configurações")

st.sidebar.markdown(f"👤 Logado como: **{st.session_state.get('usuario_atual', 'Usuário')}**")

if st.sidebar.button("🚪 Sair", use_container_width=True, type="secondary"):
    st.session_state.autenticado = False
    st.session_state.usuario_atual = ""
    st.rerun()

st.sidebar.markdown("---")

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

lista_unidades = [
    "VISÃO GERAL",
    "OS CASA DO COMERCIO",
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
    "UNIDADE MOVEL INFORMÁTICA",
    "UNIDADE MOVEL MODA E BELEZA",
    "UNIDADE MOVEL TURISMO E HOTELARIA",
    "CENTRO DE DISTRIBUIÇÃO SENAC"
]

unidade_escolhida = st.sidebar.selectbox("Escolha a Unidade", lista_unidades)

st.sidebar.markdown("---")
arquivo_carregado = st.sidebar.file_uploader("Enviar manualmente ", type=["csv", "xls", "xlsx"])


# ==========================================
# FUNÇÃO DE CARREGAMENTO COM CACHE (OTIMIZAÇÃO DE PERFORMANCE)
# ==========================================
@st.cache_data(ttl=1800)
def carregar_dados_unidade(url, arquivo_submetido, nome_unidade):
    df_temp = None
    if url:
        try:
            link_download = url
            if "?" in link_download and "download=1" not in link_download:
                link_download += "&download=1"
            elif "?" not in link_download:
                link_download += "?download=1"

            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(link_download, headers=headers)

            if response.status_code == 200:
                if ".csv" in link_download.lower():
                    df_temp = pd.read_csv(io.BytesIO(response.content))
                else:
                    df_temp = pd.read_excel(io.BytesIO(response.content))
                st.success(f"✅ Dados da unidade **{nome_unidade}** carregados com sucesso ({len(df_temp):,} registros)!")
            else:
                st.error(f"❌ Erro HTTP {response.status_code} ao acessar o link do OneDrive para `{nome_unidade}`.")
        except Exception as e:
            st.error(f"❌ Erro ao baixar dados da unidade: {e}")

    elif arquivo_submetido is not None:
        try:
            if arquivo_submetido.name.endswith('.csv'):
                df_temp = pd.read_csv(arquivo_submetido)
            else:
                df_temp = pd.read_excel(arquivo_submetido)
            st.success(f"✅ Arquivo `{arquivo_submetido.name}` carregado com sucesso ({len(df_temp):,} registros)!")
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo enviado: {e}")

    elif os.path.exists(f"{nome_unidade}.csv"):
        try:
            df_temp = pd.read_csv(f"{nome_unidade}.csv")
            st.info(f"ℹ️ Usando arquivo local: `{nome_unidade}.csv` ({len(df_temp):,} registros)")
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo local: {e}")

    elif os.path.exists(f"{nome_unidade}.xls"):
        try:
            df_temp = pd.read_excel(f"{nome_unidade}.xls")
            st.info(f"ℹ️ Usando arquivo local: `{nome_unidade}.xls` ({len(df_temp):,} registros)")
        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo local: {e}")

    return df_temp


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


coluna_comentario = "OBSERVAÇÃO ABERTURA"

url_unidade = ""
try:
    if hasattr(st, "secrets") and "UNIDADES" in st.secrets and unidade_escolhida in st.secrets["UNIDADES"]:
        url_unidade = st.secrets["UNIDADES"][unidade_escolhida]
except Exception:
    pass

df = carregar_dados_unidade(url_unidade, arquivo_carregado, unidade_escolhida)

# --- Guardas: sem base ou sem coluna obrigatória, o app para aqui (evita erros nas etapas seguintes) ---
if df is None:
    st.info("📂 Nenhuma base carregada. Selecione uma unidade com dados disponíveis ou envie um arquivo na barra lateral.")
    st.stop()

df.columns = [str(c).strip().upper() for c in df.columns]

if coluna_comentario not in df.columns:
    st.error(f"❌ A coluna obrigatória **'{coluna_comentario}'** não foi encontrada na planilha.")
    st.write(f"📋 **Colunas disponíveis na planilha:** `{list(df.columns)}`")
    st.stop()

df = df.fillna("N/A")
df = df.dropna(subset=[coluna_comentario]).copy()

if "CATEGORIA_NLP" not in df.columns:
    with st.spinner(f"Processando algoritmo de classificação NLP em {len(df):,} registros..."):
        df = classificar_texto_nlp(df, coluna_comentario)

# Identifica a coluna de máquinas
col_maquinas_ref = None
for c in df.columns:
    if c.strip().upper() in ["MÁQUINAS", "MAQUINAS", "MÁQUINA", "MAQUINA", "EQUIPAMENTO"]:
        col_maquinas_ref = c
        break

# Texto de busca: Observação + Máquina/Equipamento + Categoria NLP
if col_maquinas_ref:
    df["TEXTO_BUSCA"] = (
        df[coluna_comentario].astype(str).str.lower().fillna("") + " " +
        df[col_maquinas_ref].astype(str).str.lower().fillna("") + " " +
        df["CATEGORIA_NLP"].astype(str).str.lower().fillna("")
    )
else:
    df["TEXTO_BUSCA"] = (
        df[coluna_comentario].astype(str).str.lower().fillna("") + " " +
        df["CATEGORIA_NLP"].astype(str).str.lower().fillna("")
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filtros Dashboard")

df_filtrado_dashboard = df.copy()

if "STATUS" in df.columns:
    lista_status = ["Todos"] + list(df["STATUS"].dropna().unique())
    status_escolhido = st.sidebar.selectbox("Filtrar por Status", lista_status)
    if status_escolhido != "Todos":
        df_filtrado_dashboard = df_filtrado_dashboard[df_filtrado_dashboard["STATUS"] == status_escolhido]

# ==========================================
# 🛠️ FILTRO POR TIPO DE MANUTENÇÃO
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
            df_filtrado_dashboard = df_filtrado_dashboard[coluna_str == "checklist"]
        elif "corretiva programada" in termo:
            df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("corretiva", na=False) & coluna_str.str.contains("program", na=False)]
        elif "corretiva emergencial" in termo:
            df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("corretiva", na=False) & coluna_str.str.contains("emerg|urg", na=False)]
        elif "roteiro" in termo:
            df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("roteiro|rot", na=False)]
        elif "melhoria" in termo:
            df_filtrado_dashboard = df_filtrado_dashboard[coluna_str.str.contains("melhoria|melhor", na=False)]
else:
    st.sidebar.info("Coluna de Tipo de Manutenção não encontrada na base.")

# ==========================================
# 📅 FILTRO DE PERÍODO
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Filtrar por Período")

coluna_data = "ABERTO EM"

if coluna_data in df.columns:
    df[coluna_data] = pd.to_datetime(df[coluna_data], errors='coerce')
    df_filtrado_dashboard[coluna_data] = pd.to_datetime(df_filtrado_dashboard[coluna_data], errors='coerce')

    data_minima = df[coluna_data].min()
    data_maxima = df[coluna_data].max()

    if pd.notna(data_minima) and pd.notna(data_maxima):
        periodo_selecionado = st.sidebar.date_input(
            label="Selecione o intervalo (Início / Fim):",
            value=(data_minima.date(), data_maxima.date()),
            min_value=data_minima.date(),
            max_value=data_maxima.date(),
            format="DD/MM/YYYY"
        )

        if isinstance(periodo_selecionado, tuple) and len(periodo_selecionado) == 2:
            data_inicio, data_fim = periodo_selecionado

            mask_periodo = (df_filtrado_dashboard[coluna_data].dt.date >= data_inicio) & (df_filtrado_dashboard[coluna_data].dt.date <= data_fim)
            df_filtrado_dashboard = df_filtrado_dashboard[mask_periodo].copy()

            st.sidebar.success(f"Período: {data_inicio.strftime('%d/%m/%Y')} até {data_fim.strftime('%d/%m/%Y')}")
else:
    st.sidebar.info("Coluna de Data ('ABERTO EM') não encontrada na base.")
PALETA = {
    "azul": "#60A5FA", "azul_claro": "#38BDF8", "verde": "#34D399",
    "laranja": "#FBBF24", "vermelho": "#F87171", "roxo": "#A78BFA",
    "teal": "#2DD4BF", "cinza": "#94A3B8",
}
COR_STATUS = {
    "Concluídas": "#34D399",
    "Programadas": "#60A5FA",
    "Pendentes / Abertas": "#F87171",
    "Canceladas": "#64748B",
}
CFG_PLOT = {"displayModeBar": False, "scrollZoom": False}


def grupo_status(s):
    s = str(s).lower()
    if any(t in s for t in ["concluíd", "concluid", "fechad", "encerrad", "finalizad"]):
        return "Concluídas"
    if "cancelad" in s:
        return "Canceladas"
    if any(t in s for t in ["program", "planejad"]):
        return "Programadas"
    return "Pendentes / Abertas"


def moeda_curta(v):
    v = float(v or 0)
    if abs(v) >= 1e6:
        return f"R$ {v / 1e6:.1f} mi".replace(".", ",")
    if abs(v) >= 1e3:
        return f"R$ {v / 1e3:.1f} mil".replace(".", ",")
    return formatar_moeda_br(v)


def preparar_dash(base: pd.DataFrame) -> pd.DataFrame:
    """Colunas auxiliares (datas, status, custos, tempo) usadas só pelo painel."""
    d = base.copy()
    for c in ["ABERTO EM", "EXEC. PREVISTA", "FINALIZAÇÃO", "FECHADO EM"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    d["STATUS_GRUPO"] = d["STATUS"].apply(grupo_status) if "STATUS" in d.columns else "Pendentes / Abertas"

    d["CUSTO_MAT"] = (limpar_coluna_monetaria(d["CUSTO COM MATERIAIS"]).values
                      if "CUSTO COM MATERIAIS" in d.columns else 0.0)
    d["CUSTO_MOE"] = (limpar_coluna_monetaria(d["MÃO DE OBRA EXTERNA"]).values
                      if "MÃO DE OBRA EXTERNA" in d.columns else 0.0)
    d["CUSTO_TOT"] = d["CUSTO_MAT"] + d["CUSTO_MOE"]

    if "ABERTO EM" in d.columns:
        d["MES_REF"] = d["ABERTO EM"].dt.to_period("M").dt.to_timestamp()

    c_ini = "EXEC. PREVISTA" if "EXEC. PREVISTA" in d.columns else None
    c_fim = "FINALIZAÇÃO" if "FINALIZAÇÃO" in d.columns else ("FECHADO EM" if "FECHADO EM" in d.columns else None)
    if c_ini and c_fim:
        dias = (d[c_fim] - d[c_ini]).dt.total_seconds() / 86400
        d["TEMPO_DIAS"] = dias.where(dias >= 0)
    return d


def estilizar(fig, altura=340, legenda=True):
    """Tema único para todos os gráficos (escuro, executivo, pt-BR)."""
    fig.update_layout(
        template="plotly_dark", height=altura,
        margin=dict(t=20, b=10, l=10, r=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#CBD5E1"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        separators=",.", showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="",
                    font=dict(color="#CBD5E1")),
        hoverlabel=dict(bgcolor="#16213A", bordercolor="#334155",
                        font=dict(size=12, color="#E2E8F0")),
    )
    fig.update_xaxes(showgrid=False, fixedrange=True, linecolor="#334155",
                     tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8"))
    fig.update_yaxes(gridcolor="#1F2A44", fixedrange=True, zeroline=False,
                     tickfont=dict(color="#94A3B8"), title_font=dict(color="#94A3B8"))
    return fig


def mostrar(fig, key=None, selecionavel=False):
    kw = dict(config=CFG_PLOT, theme=None)
    if key:
        kw["key"] = key
    if selecionavel:
        kw["on_select"] = "rerun"
    try:
        return st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, **kw)


def secao(titulo):
    st.markdown(f'<div class="section-title">{titulo}</div>', unsafe_allow_html=True)


def titulo_grafico(titulo, nota=""):
    st.markdown(f'<div class="chart-title">{titulo}</div>', unsafe_allow_html=True)
    if nota:
        st.markdown(f'<div class="chart-note">{nota}</div>', unsafe_allow_html=True)


def card_kpi(titulo, valor, cor, sub="", delta=None, inverso=False):
    """delta em %. inverso=True => subir é ruim (ex.: custo, volume de OS)."""
    chip = ""
    if delta is not None and not pd.isna(delta):
        bom = (delta <= 0) if inverso else (delta >= 0)
        seta = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
        cor_d = "#34D399" if bom else "#F87171"
        txt = f"{abs(delta):.1f}".replace(".", ",")
        chip = f'<span class="kpi-delta" style="color:{cor_d};">{seta} {txt}% vs mês ant.</span>'
    return (f'<div class="kpi-card" style="border-left-color:{cor};">'
            f'<div class="kpi-title">{titulo}</div>'
            f'<div class="kpi-value" style="color:{cor};">{valor}</div>'
            f'<div class="kpi-sub">{sub}{chip}</div></div>')


def gerar_destaques(d: pd.DataFrame, col_setor=None) -> list:
    """Frases automáticas para leitura rápida da diretoria."""
    itens, total = [], len(d)
    hoje = pd.Timestamp.today().normalize()

    if "CATEGORIA_NLP" in d.columns and total:
        top = d["CATEGORIA_NLP"].value_counts()
        itens.append(f"🏷️ Categoria líder: <b>{top.index[0]}</b> ({top.iloc[0] / total * 100:.0f}% das OS)")

    if col_setor and col_setor in d.columns and total:
        top = d[col_setor].value_counts()
        itens.append(f"🏢 Maior demanda: <b>{top.index[0]}</b> ({top.iloc[0]} OS)")

    if "MES_REF" in d.columns and d["MES_REF"].notna().any():
        m = d["MES_REF"].value_counts().idxmax()
        itens.append(f"📅 Mês de pico: <b>{m:%m/%Y}</b>")

    pend = d[d["STATUS_GRUPO"] == "Pendentes / Abertas"]
    if "ABERTO EM" in pend.columns and not pend.empty:
        idade = (hoje - pend["ABERTO EM"]).dt.days
        n30 = int((idade > 30).sum())
        if n30:
            itens.append(f"⏳ <b>{n30}</b> OS pendentes há mais de 30 dias")

    if "CATEGORIA_NLP" in d.columns and d["CUSTO_TOT"].sum() > 0:
        c = d.groupby("CATEGORIA_NLP")["CUSTO_TOT"].sum().sort_values(ascending=False)
        itens.append(f"💰 Maior custo: <b>{c.index[0]}</b> ({moeda_curta(c.iloc[0])})")
    return itens[:5]


def _sel_pareto():
    v = st.session_state.get("select_maquina_pareto_direto", "")
    st.session_state["maquina_pareto_ativa"] = None if str(v).startswith("--") else v


def _fechar_pareto():
    st.session_state["maquina_pareto_ativa"] = None
    st.session_state["select_maquina_pareto_direto"] = "-- Ou selecione a máquina aqui --"
# ==========================================
# NAVEGAÇÃO POR BOTÕES (COM LIMPEZA DE ESTADO)
# ==========================================
if "aba_ativa" not in st.session_state:
    st.session_state.aba_ativa = "dash"

col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    if st.button("📊 Dashboard Geral", use_container_width=True, type="primary" if st.session_state.aba_ativa == "dash" else "secondary"):
        st.session_state.aba_ativa = "dash"
        st.session_state.termo_pesquisa = ""
        if "relatorio_gerado" in st.session_state:
            del st.session_state.relatorio_gerado
        st.rerun()

with col_b2:
    if st.button("🔍 Pesquisa & Relatórios", use_container_width=True, type="primary" if st.session_state.aba_ativa == "busca" else "secondary"):
        st.session_state.aba_ativa = "busca"
        st.rerun()

with col_b3:
    if st.button("🏷️ Classificação (NLP)", use_container_width=True, type="primary" if st.session_state.aba_ativa == "nlp" else "secondary"):
        st.session_state.aba_ativa = "nlp"
        st.session_state.termo_pesquisa = ""
        if "relatorio_gerado" in st.session_state:
            del st.session_state.relatorio_gerado
        st.rerun()

with col_b4:
    if st.button("💬 Chat Inteligente", use_container_width=True, type="primary" if st.session_state.aba_ativa == "chat" else "secondary"):
        st.session_state.aba_ativa = "chat"
        st.session_state.termo_pesquisa = ""
        if "relatorio_gerado" in st.session_state:
            del st.session_state.relatorio_gerado
        st.rerun()

st.markdown("---")

# ==========================================
# CONTEÚDO DE CADA SEÇÃO
# ==========================================
if st.session_state.aba_ativa == "dash":
    d = preparar_dash(df_filtrado_dashboard)
    total_os = len(d)

    if total_os == 0:
        st.warning("Nenhuma OS encontrada para os filtros selecionados.")
        st.stop()

    hoje = pd.Timestamp.today().normalize()
    fmt_int = lambda n: f"{int(n):,}".replace(",", ".")

    # ---------- Faixa executiva ----------
    if "ABERTO EM" in d.columns and d["ABERTO EM"].notna().any():
        periodo_txt = f"{d['ABERTO EM'].min():%d/%m/%Y} a {d['ABERTO EM'].max():%d/%m/%Y}"
    else:
        periodo_txt = "—"
    st.markdown(
        f'<div class="hero"><div>'
        f'<div class="hero-title">Painel Executivo da Manutenção</div>'
        f'<div class="hero-sub">{unidade_escolhida} · Período: {periodo_txt}</div></div>'
        f'<div class="hero-badge">{fmt_int(total_os)} OS no período</div></div>',
        unsafe_allow_html=True)

    # ---------- Indicadores ----------
    cont = d["STATUS_GRUPO"].value_counts()
    n_conc = int(cont.get("Concluídas", 0))
    n_prog = int(cont.get("Programadas", 0))
    n_pend = int(cont.get("Pendentes / Abertas", 0))
    n_canc = int(cont.get("Canceladas", 0))
    pct_conc = n_conc / total_os * 100

    custo_mat, custo_moe = d["CUSTO_MAT"].sum(), d["CUSTO_MOE"].sum()
    custo_tot = custo_mat + custo_moe

    tempo_medio = d["TEMPO_DIAS"].mean() if "TEMPO_DIAS" in d.columns else np.nan
    tempo_txt = f"{tempo_medio:.1f} dias".replace(".", ",") if pd.notna(tempo_medio) else "N/A"

    delta_os = delta_custo = None
    if "MES_REF" in d.columns:
        base_m = d.dropna(subset=["MES_REF"])
        s_os = base_m.groupby("MES_REF").size().sort_index()
        if len(s_os) >= 2 and s_os.iloc[-2] > 0:
            delta_os = (s_os.iloc[-1] / s_os.iloc[-2] - 1) * 100
        s_c = base_m.groupby("MES_REF")["CUSTO_TOT"].sum().sort_index()
        if len(s_c) >= 2 and s_c.iloc[-2] > 0:
            delta_custo = (s_c.iloc[-1] / s_c.iloc[-2] - 1) * 100

    k = st.columns(6)
    cards = [
        card_kpi("Total de OS", fmt_int(total_os), PALETA["azul"], "no período", delta_os, inverso=True),
        card_kpi("Concluídas", fmt_int(n_conc), PALETA["verde"], f"{pct_conc:.1f}% do total".replace(".", ",")),
        card_kpi("Backlog", fmt_int(n_pend), PALETA["roxo"], f"{fmt_int(n_prog)} programadas"),
        card_kpi("Conformidade", f"{pct_conc:.1f}%".replace(".", ","), PALETA["azul_claro"], "concluídas ÷ total"),
        card_kpi("Tempo médio", tempo_txt, PALETA["teal"], "previsto → finalização"),
        card_kpi("Custo total", moeda_curta(custo_tot), PALETA["vermelho"],
                 f"Mat. {moeda_curta(custo_mat)} · MO {moeda_curta(custo_moe)}", delta_custo, inverso=True),
    ]
    for col, html in zip(k, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)

    # ---------- Destaques automáticos ----------
    col_setor = next((c for c in ["SETOR", "MÁQUINA", "MÁQUINAS", "MAQUINA", "MAQUINAS"] if c in d.columns), None)
    destaques = gerar_destaques(d, col_setor)
    if destaques:
        st.markdown('<div class="insights">' + "".join(f'<span class="chip">{t}</span>' for t in destaques) + '</div>',
                    unsafe_allow_html=True)

    # =================================================================
    # 1) SITUAÇÃO ATUAL — gauge + rosca de status + rosca por tipo
    # =================================================================
    secao("🎯 Situação atual")
    g1, g2, g3 = st.columns(3)

    with g1:
        titulo_grafico("Conformidade geral", "Meta sugerida: ≥ 85%")
        cor_g = PALETA["verde"] if pct_conc >= 85 else (PALETA["laranja"] if pct_conc >= 60 else PALETA["vermelho"])
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pct_conc,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 38, "color": cor_g}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B",
                           "tickfont": {"color": "#94A3B8"}},
                   "bar": {"color": cor_g, "thickness": 0.28},
                   "bgcolor": "#16213A", "borderwidth": 0,
                   "steps": [{"range": [0, 60], "color": "rgba(248,113,113,.22)"},
                             {"range": [60, 85], "color": "rgba(251,191,36,.22)"},
                             {"range": [85, 100], "color": "rgba(52,211,153,.22)"}],
                   "threshold": {"line": {"color": "#F1F5F9", "width": 3}, "thickness": 0.8, "value": 85}}))
        estilizar(fig_g, 280, legenda=False)
        fig_g.update_layout(margin=dict(t=30, b=0, l=20, r=20))
        mostrar(fig_g, key="dash_gauge")

    with g2:
        titulo_grafico("Distribuição por status")
        df_st = d["STATUS_GRUPO"].value_counts().reset_index()
        df_st.columns = ["Status", "Quantidade"]
        fig_st = px.pie(df_st, names="Status", values="Quantidade", hole=0.62,
                        color="Status", color_discrete_map=COR_STATUS)
        fig_st.update_traces(textinfo="percent", textposition="outside", sort=False,
                             marker=dict(line=dict(color="#0B1220", width=2)))
        fig_st.add_annotation(text=f"<b>{fmt_int(total_os)}</b><br>OS", showarrow=False, font=dict(size=20, color="#F1F5F9"))
        estilizar(fig_st, 380)
        fig_st.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", yanchor="top"))
        mostrar(fig_st, key="dash_status")

    with g3:
        if col_tipo_servico and col_tipo_servico in d.columns:
            titulo_grafico("Mix por tipo de manutenção")
            df_tp = d[col_tipo_servico].astype(str).str.strip().value_counts().head(6).reset_index()
            df_tp.columns = ["Tipo", "Quantidade"]
            rot_centro = f"<b>{len(df_tp)}</b><br>tipos"
            nomes, vals = "Tipo", "Quantidade"
        else:
            titulo_grafico("Mix por categoria (NLP)")
            df_tp = d["CATEGORIA_NLP"].value_counts().head(6).reset_index()
            df_tp.columns = ["Tipo", "Quantidade"]
            rot_centro = "<b>Top 6</b>"
        fig_tp = px.pie(df_tp, names="Tipo", values="Quantidade", hole=0.62,
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig_tp.update_traces(textinfo="percent", textposition="outside",
                             marker=dict(line=dict(color="#0B1220", width=2)))
        fig_tp.add_annotation(text=rot_centro, showarrow=False, font=dict(size=18, color="#F1F5F9"))
        estilizar(fig_tp, 380)
        fig_tp.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", yanchor="top"))
        mostrar(fig_tp, key="dash_tipo")

    # =================================================================
    # 2) EVOLUÇÃO & BACKLOG
    # =================================================================
    secao("📈 Evolução e backlog")
    e1, e2 = st.columns([2, 1])

    with e1:
        titulo_grafico("OS por mês e % de conclusão", "Barras = volume por status · Linha = % concluído no mês")
        if "MES_REF" in d.columns and d["MES_REF"].notna().any():
            base_m = d.dropna(subset=["MES_REF"])
            agr = base_m.groupby(["MES_REF", "STATUS_GRUPO"]).size().reset_index(name="Qtd")
            fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
            for g in ["Concluídas", "Programadas", "Pendentes / Abertas", "Canceladas"]:
                sub = agr[agr["STATUS_GRUPO"] == g]
                if not sub.empty:
                    fig_ev.add_bar(x=sub["MES_REF"], y=sub["Qtd"], name=g,
                                   marker_color=COR_STATUS[g], secondary_y=False)
            tot_m = base_m.groupby("MES_REF").size()
            conc_m = (base_m[base_m["STATUS_GRUPO"] == "Concluídas"].groupby("MES_REF").size()
                      .reindex(tot_m.index, fill_value=0))
            pct_m = conc_m / tot_m * 100
            fig_ev.add_scatter(x=pct_m.index, y=pct_m.values, name="% concluído",
                               mode="lines+markers", line=dict(color=PALETA["azul"], width=3),
                               marker=dict(size=7), secondary_y=True)
            estilizar(fig_ev, 360)
            fig_ev.update_layout(barmode="stack")
            fig_ev.update_xaxes(tickformat="%m/%Y")
            fig_ev.update_yaxes(title_text="OS", secondary_y=False)
            fig_ev.update_yaxes(range=[0, 105], ticksuffix="%", showgrid=False, secondary_y=True)
            mostrar(fig_ev, key="dash_evolucao")
        else:
            st.info("Coluna de data indisponível para a evolução temporal.")

    with e2:
        titulo_grafico("Idade do backlog", "OS pendentes por tempo em aberto")
        pend = d[d["STATUS_GRUPO"] == "Pendentes / Abertas"]
        if "ABERTO EM" in pend.columns and not pend.empty and pend["ABERTO EM"].notna().any():
            idade = (hoje - pend["ABERTO EM"]).dt.days.dropna()
            faixas = ["0–7 dias", "8–30 dias", "31–90 dias", "> 90 dias"]
            cats = pd.cut(idade, bins=[-1, 7, 30, 90, 10 ** 6], labels=faixas)
            qtds = [int((cats == f).sum()) for f in faixas]
            fig_ag = go.Figure(go.Bar(x=faixas, y=qtds, text=qtds, textposition="outside",
                                      marker_color=["#34D399", "#FBBF24", "#FB923C", "#F87171"]))
            estilizar(fig_ag, 360, legenda=False)
            fig_ag.update_yaxes(showgrid=False, visible=False)
            fig_ag.update_traces(cliponaxis=False)
            mostrar(fig_ag, key="dash_aging")
        else:
            st.success("Nenhuma OS pendente no período. 🎉")

    # =================================================================
    # 3) ONDE ESTÃO OS PROBLEMAS
    # =================================================================
    secao("🏢 Onde estão os problemas")
    p1, p2 = st.columns(2)

    with p1:
        titulo_grafico("Top 10 setores / máquinas", "Clique numa barra para abrir o Diagrama de Pareto")
        if col_setor:
            df_setor = d[col_setor].value_counts().head(10).reset_index()
            df_setor.columns = ["Setor/Máquina", "Quantidade"]
            fig_set = px.bar(df_setor, x="Quantidade", y="Setor/Máquina", orientation="h",
                             color="Quantidade", color_continuous_scale=[[0, "#1E3A8A"], [1, "#60A5FA"]], text="Quantidade")
            fig_set.update_traces(textposition="outside", cliponaxis=False)
            fig_set.update_coloraxes(showscale=False)
            estilizar(fig_set, 380, legenda=False)
            fig_set.update_yaxes(categoryorder="total ascending", title_text="")
            fig_set.update_xaxes(visible=False)

            evento = mostrar(fig_set, key="grafico_barras_setor_pareto", selecionavel=True)
            try:
                pts = evento.get("selection", {}).get("points", [])
                if pts:
                    st.session_state["maquina_pareto_ativa"] = pts[0].get("y")
            except Exception:
                pass

            opcoes = ["-- Ou selecione a máquina aqui --"] + sorted(d[col_setor].dropna().unique().astype(str))
            st.selectbox("📊 Seleção rápida para o Diagrama de Pareto:", options=opcoes,
                         key="select_maquina_pareto_direto", on_change=_sel_pareto)
        else:
            st.info("Coluna de Setor ou Máquina não encontrada.")

    with p2:
        titulo_grafico("Mapa de categorias (NLP)", "Área proporcional ao nº de OS")
        if "CATEGORIA_NLP" in d.columns:
            df_cat = d["CATEGORIA_NLP"].value_counts().reset_index()
            df_cat.columns = ["Categoria", "Quantidade"]
            fig_tm = px.treemap(df_cat, path=["Categoria"], values="Quantidade",
                                color="Quantidade", color_continuous_scale=[[0, "#134E4A"], [1, "#2DD4BF"]])
            fig_tm.update_traces(texttemplate="<b>%{label}</b><br>%{value}",
                                 textfont=dict(size=13, color="#F8FAFC"),
                                 marker=dict(cornerradius=4, line=dict(width=2, color="#0B1220")))
            fig_tm.update_coloraxes(showscale=False)
            estilizar(fig_tm, 380, legenda=False)
            fig_tm.update_layout(margin=dict(t=10, b=10, l=0, r=0))
            mostrar(fig_tm, key="dash_treemap")
        else:
            st.info("Dado de categoria NLP não disponível.")

    # =================================================================
    # 4) CUSTOS
    # =================================================================
    secao("💰 Custos")
    if custo_tot > 0 and "MES_REF" in d.columns:
        c1, c2 = st.columns([2, 1])
        with c1:
            titulo_grafico("Custo mensal (materiais + mão de obra externa)")
            cm = (d.dropna(subset=["MES_REF"]).groupby("MES_REF")[["CUSTO_MAT", "CUSTO_MOE"]]
                  .sum().reset_index())
            fig_cm = go.Figure()
            fig_cm.add_bar(x=cm["MES_REF"], y=cm["CUSTO_MAT"], name="Materiais", marker_color=PALETA["verde"])
            fig_cm.add_bar(x=cm["MES_REF"], y=cm["CUSTO_MOE"], name="Mão de obra externa", marker_color=PALETA["laranja"])
            estilizar(fig_cm,340)
            fig_cm.update_layout(barmode="stack")
            fig_cm.update_xaxes(tickformat="%m/%Y")
            fig_cm.update_yaxes(tickprefix="R$ ", tickformat=",.0f")
            mostrar(fig_cm, key="dash_custo_mensal")
        with c2:
            titulo_grafico("Top 8 categorias por custo")
            cc = (d.groupby("CATEGORIA_NLP")["CUSTO_TOT"].sum()
                  .sort_values(ascending=False).head(8).reset_index())
            cc["Texto"] = cc["CUSTO_TOT"].apply(moeda_curta)
            fig_cc = px.bar(cc, x="CUSTO_TOT", y="CATEGORIA_NLP", orientation="h", text="Texto")
            fig_cc.update_traces(marker_color=PALETA["vermelho"], textposition="outside", cliponaxis=False)
            estilizar(fig_cc, 340, legenda=False)
            fig_cc.update_yaxes(categoryorder="total ascending", title_text="")
            fig_cc.update_xaxes(visible=False)
            mostrar(fig_cc, key="dash_custo_cat")
    else:
        st.info("Sem custos registrados no período selecionado.")

    # =================================================================
    # 5) PADRÕES E DESEMPENHO
    # =================================================================
    secao("🔬 Padrões e desempenho")
    h1, h2 = st.columns(2)

    with h1:
        titulo_grafico("Mapa de calor: categoria × mês", "Últimos 12 meses · top 8 categorias")
        if "MES_REF" in d.columns and "CATEGORIA_NLP" in d.columns and d["MES_REF"].notna().any():
            meses = sorted(d["MES_REF"].dropna().unique())[-12:]
            top_cats = d["CATEGORIA_NLP"].value_counts().head(8).index
            sub = d[d["MES_REF"].isin(meses) & d["CATEGORIA_NLP"].isin(top_cats)]
            ct = pd.crosstab(sub["CATEGORIA_NLP"], sub["MES_REF"]).reindex(columns=meses, fill_value=0)
            ct = ct.loc[ct.sum(axis=1).sort_values().index]
            fig_hm = go.Figure(go.Heatmap(
                z=ct.values, x=[pd.Timestamp(m).strftime("%m/%Y") for m in ct.columns], y=list(ct.index),
                colorscale=[[0, "#16213A"], [1, "#3B82F6"]], texttemplate="%{z}",
                textfont=dict(color="#F1F5F9"), xgap=2, ygap=2, showscale=False,
                hovertemplate="%{y}<br>%{x}: %{z} OS<extra></extra>"))
            estilizar(fig_hm, 360, legenda=False)
            fig_hm.update_yaxes(showgrid=False)
            mostrar(fig_hm, key="dash_heatmap")
        else:
            st.info("Dados insuficientes para o mapa de calor.")

    with h2:
        titulo_grafico("Tempo médio de atendimento por categoria", "Em dias (previsto → finalização)")
        if "TEMPO_DIAS" in d.columns and d["TEMPO_DIAS"].notna().any() and "CATEGORIA_NLP" in d.columns:
            tm = (d.groupby("CATEGORIA_NLP")["TEMPO_DIAS"].mean().dropna()
                  .sort_values(ascending=False).head(8).reset_index())
            tm["Texto"] = tm["TEMPO_DIAS"].apply(lambda x: f"{x:.1f} d".replace(".", ","))
            fig_tm2 = px.bar(tm, x="TEMPO_DIAS", y="CATEGORIA_NLP", orientation="h", text="Texto")
            fig_tm2.update_traces(marker_color=PALETA["teal"], textposition="outside", cliponaxis=False)
            estilizar(fig_tm2, 360, legenda=False)
            fig_tm2.update_yaxes(categoryorder="total ascending", title_text="")
            fig_tm2.update_xaxes(visible=False)
            mostrar(fig_tm2, key="dash_tempo_cat")
        else:
            st.info("Colunas de datas (EXEC. PREVISTA / FINALIZAÇÃO) indisponíveis.")

    # =================================================================
    # 6) DIAGRAMA DE PARETO (abre ao clicar no gráfico de setores)
    # =================================================================
    maq_ativa = st.session_state.get("maquina_pareto_ativa")
    if maq_ativa:
        secao(f"📊 Diagrama de Pareto de falhas — {maq_ativa}")
        st.button("❌ Fechar Pareto", key="btn_fechar_pareto", on_click=_fechar_pareto)

        col_maq_filtro = next((c for c in ["MÁQUINA", "MAQUINA", "MÁQUINAS", "MAQUINAS", "SETOR"] if c in d.columns), None)
        if col_maq_filtro:
            base_p = d[d[col_maq_filtro].astype(str) == str(maq_ativa)]
            col_falha = next((c for c in ["CATEGORIA_NLP", "FALHA", "PROBLEMA", coluna_comentario]
                              if c and c in base_p.columns), None)
            if col_falha and not base_p.empty:
                dp = base_p[col_falha].astype(str).str.slice(0, 60).value_counts().head(15).reset_index()
                dp.columns = ["Falha", "Quantidade"]
                dp["Acum"] = dp["Quantidade"].cumsum() / dp["Quantidade"].sum() * 100

                fig_p = make_subplots(specs=[[{"secondary_y": True}]])
                fig_p.add_bar(x=dp["Falha"], y=dp["Quantidade"], name="Ocorrências",
                              marker_color=PALETA["azul"], text=dp["Quantidade"], textposition="outside",
                              secondary_y=False)
                fig_p.add_scatter(x=dp["Falha"], y=dp["Acum"], name="Acumulado %", mode="lines+markers",
                                  line=dict(color=PALETA["vermelho"], width=3), secondary_y=True)
                fig_p.add_hline(y=80, line_dash="dash", line_color="#94A3B8",
                                annotation_text="80%", secondary_y=True)
                estilizar(fig_p, 420)
                fig_p.update_xaxes(tickangle=-35)
                fig_p.update_yaxes(title_text="Ocorrências", secondary_y=False)
                fig_p.update_yaxes(range=[0, 105], ticksuffix="%", showgrid=False, secondary_y=True)
                mostrar(fig_p, key="dash_pareto")
            else:
                st.warning(f"Não há dados suficientes de falhas para '{maq_ativa}'.")

    # =================================================================
    # 7) DETALHAMENTO (substitui os botões de KPI)
    # =================================================================
    secao("📋 Detalhamento das OS")
    with st.expander("Abrir tabela detalhada", expanded=False):
        visao = st.radio("Visão", ["Todas as OS", "Materiais", "Mão de obra externa", "Custos (comparativo)"],
                         horizontal=True, key="visao_detalhe")
        base_det = df_filtrado_dashboard
        mapa_cols = {
            "Todas as OS": None,
            "Materiais": ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "CUSTO COM MATERIAIS"],
            "Mão de obra externa": ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "MÃO DE OBRA EXTERNA"],
            "Custos (comparativo)": ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "CUSTO COM MATERIAIS", "MÃO DE OBRA EXTERNA"],
        }
        cols_sel = mapa_cols[visao]
        tabela_det = base_det if cols_sel is None else base_det[[c for c in cols_sel if c in base_det.columns]]
        st.dataframe(tabela_det, use_container_width=True, hide_index=True)

elif st.session_state.aba_ativa == "busca":
    st.subheader("🔍 Pesquisa Textual & Geração de Relatórios por IA")

    termo_pesquisa = st.text_input(
        "Digite um termo para pesquisar nas observações ou máquinas (ex: ar condicionado, vazamento, bomba, gerador):",
        key="termo_pesquisa"
    )

    if termo_pesquisa:
        termo_busca = termo_pesquisa.lower().strip()

        df_resultado = df[df["TEXTO_BUSCA"].str.contains(re.escape(termo_busca), na=False)]

        st.markdown(f"**Registros encontrados:** {len(df_resultado):,} de {len(df):,}".replace(",", "."))

        if len(df_resultado) > 0:
            st.markdown("---")
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                st.write(f"Gerar relatório executivo automatizado por Inteligência Artificial para os resultados de **'{termo_pesquisa}'**?")
            with col_r2:
                if st.button("🤖 Gerar Relatório IA", use_container_width=True):
                    with st.spinner("Gerando relatório com IA via Groq..."):
                        relatorio_gerado = gerar_relatorio_ia(termo_pesquisa, df_resultado, groq_api_key_input)
                        st.session_state.relatorio_gerado = relatorio_gerado
                        st.session_state.termo_relatorio = termo_pesquisa

            if "relatorio_gerado" in st.session_state:
                st.markdown("---")
                st.markdown(f"### 📋 Relatório Executivo da IA para: *{st.session_state.get('termo_relatorio', '')}*")
                st.markdown(st.session_state.relatorio_gerado)

                serie_tabela = extrair_dados_tabela_markdown(st.session_state.relatorio_gerado)
                if serie_tabela is not None and not serie_tabela.empty:
                    st.markdown("#### 📊 Gráfico extraído do Relatório")
                    fig_rel = px.bar(x=serie_tabela.index, y=serie_tabela.values, labels={'x': 'Área/Problema', 'y': 'Frequência'})
                    fig_rel.update_traces(marker_color=PALETA["azul_claro"])
                    estilizar(fig_rel, 340, legenda=False)
                    mostrar(fig_rel, key="rel_ia_grafico")

            st.markdown("---")
            st.markdown("#### 📋 Listagem das OS Encontradas")
            cols_exibir = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "SETOR", "MÁQUINA", "MÁQUINAS", "CATEGORIA_NLP", coluna_comentario] if c in df_resultado.columns]
            st.dataframe(df_resultado[cols_exibir], use_container_width=True, hide_index=True)
        else:
            st.warning("Nenhuma Ordem de Serviço encontrada com o termo informado.")

elif st.session_state.aba_ativa == "nlp":
    st.subheader("🔍 Detalhamento por Categoria e Equipamentos")

    if "CATEGORIA_NLP" in df_filtrado_dashboard.columns:
        categorias_disponiveis = ["-- Selecione uma categoria --", "Todas"] + sorted(df_filtrado_dashboard["CATEGORIA_NLP"].dropna().unique())

        cat_selecionada = st.selectbox(
            "Selecione uma Categoria para abrir e ver os itens:",
            options=categorias_disponiveis,
            index=0
        )

        if cat_selecionada and cat_selecionada != "-- Selecione uma categoria --":
            if cat_selecionada == "Todas":
                df_cat = df_filtrado_dashboard.copy()
                st.write(f"Exibindo registros de **Todas** as categorias (filtradas): **{len(df_cat):,}**")
            else:
                df_cat = df_filtrado_dashboard[df_filtrado_dashboard["CATEGORIA_NLP"] == cat_selecionada]
                st.write(f"Total de registros na categoria **{cat_selecionada}**: **{len(df_cat):,}**")

            col_maq = None
            for c in ["MÁQUINAS", "MAQUINAS", "MÁQUINA", "MAQUINA"]:
                if c in df_filtrado_dashboard.columns:
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
                        if c in df_filtrado_dashboard.columns:
                            col_exec_prevista = c
                            break

                    candidatos_cols = ["CÓDIGO", "ABERTO EM", col_exec_prevista, "STATUS", "SETOR", col_maq, coluna_comentario, "CUSTO COM MATERIAIS"]
                    cols_mostrar = [c for c in candidatos_cols if c and c in df_os_maquina.columns]

                    st.dataframe(df_os_maquina[cols_mostrar], use_container_width=True, hide_index=True)
                else:
                    st.info("👈 Clique em uma linha na tabela acima ou selecione um equipamento no menu para carregar as Ordens de Serviço específicas.")
    else:
        st.info("Dado de categoria NLP não disponível na base filtrada.")

elif st.session_state.aba_ativa == "chat":
    st.subheader("💬 Chat Inteligente sobre a Base de Dados")
    st.write("Pergunte em linguagem natural: contagens, custos, rankings, evolução mensal, "
             "problemas recorrentes, OS específicas, etc.")

    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        usar_filtros = st.checkbox("Usar os filtros do dashboard (status, tipo, período)", value=False)
    with col_c2:
        if st.button("🧹 Limpar conversa", use_container_width=True):
            st.session_state.mensagens_chat = []
            st.rerun()

    base_chat = df_filtrado_dashboard if usar_filtros else df
    st.caption(f"Base consultada: **{len(base_chat):,}** registros".replace(",", "."))

   # with st.expander("💡 Exemplos de perguntas"):
   #     st.markdown(
   #         "- Quantas OS pendentes temos por setor?\n"
   #         "- Qual o custo total de materiais por categoria em 2025?\n"
   #         "- Quais os 10 equipamentos com mais chamados de ar condicionado?\n"
   #         "- Qual o tempo médio de atendimento por tipo de manutenção?\n"
   #         "- Mostre a evolução mensal de OS de vazamento.\n"
   #         "- Resuma os problemas mais frequentes dos últimos 3 meses."
   #     )

  # ---- Campo de pergunta (sempre visível) ----
    with st.form("form_chat", clear_on_submit=True):
        pergunta = st.text_input(
            "✍️ Digite sua pergunta",
            placeholder="Ex: Digite sua pergunta aqui"
        )
        enviado = st.form_submit_button("Enviar pergunta", use_container_width=True, type="primary")

    if enviado and pergunta.strip():
        historico_anterior = list(st.session_state.mensagens_chat)
        st.session_state.mensagens_chat.append({"role": "user", "content": pergunta.strip()})
        with st.spinner("Pesquisando na base e analisando..."):
            resposta, evidencia, resumo_busca = responder_pergunta_livre_com_todo_arquivo(
                pergunta.strip(), base_chat, groq_api_key_input,
                historico=historico_anterior, termo_busca=None
            )
        st.session_state.mensagens_chat.append(
            {"role": "assistant", "content": resposta, "evidencia": evidencia, "busca": resumo_busca}
        )

    # ---- Histórico da conversa ----
    st.markdown("---")
    if not st.session_state.mensagens_chat:
        st.info("Faça sua primeira pergunta acima 👆")
    for msg in st.session_state.mensagens_chat:
        with st.chat_message(msg["role"]):
            if msg.get("busca"):
                st.caption(f"🔎 Pesquisa aplicada: {msg['busca']}")
            st.markdown(msg["content"])
            ev = msg.get("evidencia")
            if ev is not None and len(ev) > 0:
                with st.expander(f"📋 Ver as OS usadas nesta resposta ({len(ev)} exibidas)"):
                    st.dataframe(ev, use_container_width=True, hide_index=True)
