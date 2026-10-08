import os
import io
import re
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

# Estilização CSS personalizada para um dashboard executivo moderno
st.markdown("""
    <style>
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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# FUNÇÕES DE AUXÍLIO PARA TRATAMENTO DE DADOS
# ==========================================
def limpar_coluna_monetaria(serie):
    """Converte com segurança colunas de formato monetário (ex: 'R$ 1.234,56') para float."""
    if serie is None:
        return pd.Series([0.0])
    
    if serie.dtype in ['float64', 'int64']:
        return serie.fillna(0.0)
    
    def converter_valor(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).strip()
        # Remove tudo exceto dígitos, vírgula e ponto
        val_limpo = re.sub(r'[^\d,.-]', '', val_str)
        if not val_limpo or val_limpo in ['-', '.', ',']:
            return 0.0
        
        try:
            # Se houver ponto e vírgula (ex: 1.234,56)
            if '.' in val_limpo and ',' in val_limpo:
                val_limpo = val_limpo.replace('.', '').replace(',', '.')
            elif ',' in val_limpo:
                # Se houver apenas vírgula, assume que é separador decimal (ex: 1234,56)
                val_limpo = val_limpo.replace(',', '.')
            return float(val_limpo)
        except Exception:
            return 0.0

    return serie.apply(converter_valor).astype(float).fillna(0.0)

# ==========================================
# MÓDULO DE NLP & CLASSIFICAÇÃO DE TEXTO
# ==========================================
def classificar_texto_nlp(df: pd.DataFrame, coluna_texto: str) -> pd.DataFrame:
    """Aplica algoritmo NLP (TF-IDF + Regras/Clusters) para classificar as observações."""
    df_cls = df.copy()
    if coluna_texto not in df_cls.columns:
        return df_cls
    
    textos = df_cls[coluna_texto].fillna("").astype(str).str.lower()
    
    def categorizar_por_regras(texto):
        texto = str(texto).lower().strip()

        # 1. HIDRÁULICA / SANEAMENTO
        if any(w in texto for w in [
            "vazamento", "vazando", "vaza", "água", "agua",
            "cano", "tubulação", "tubulacao", "tubo",
            "esgoto", "ralo", "torneira", "registro",
            "válvula", "valvula", "hydra", "descarga",
            "bacia sanitária", "vaso sanitário", "vaso sanitario",
            "pia", "chuveiro", "bebedouro", "caixa d'água",
            "caixa dagua", "reservatório", "reservatorio",
            "bomba d'água", "bomba dagua", "boia", "boia de nível",
            "pressão da água", "pressao da agua", "entupimento",
            "entupido", "infiltração hidráulica", "infiltracao hidraulica"
        ]):
            return "Hidráulica / Saneamento"

        # 2. ELÉTRICA
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

        # 3. CLIMATIZAÇÃO / REFRIGERAÇÃO
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

        # 4. CIVIL / EDIFICAÇÕES
        elif any(w in texto for w in [
            "parede", "reboco", "alvenaria", "argamassa",
            "cimento", "concreto", "piso", "revestimento",
            "cerâmica", "ceramica", "azulejo",
            "forro", "teto", "telhado", "cobertura",
            "calha", "rufo", "fachada", "pintura",
            "infiltração", "infiltracao", "umidade",
            "mofo", "trinca", "rachadura", "fissura",
            "porta", "janela", "batente", "fechadura",
            "dobradiça", "dobradica", "vidro",
            "espelho", "divisória", "divisoria",
            "drywall", "gesso", "granito", "mármore",
            "marcenaria", "carpintaria", "serralheria"
        ]):
            return "Civil / Edificações"

        # 5. MECÂNICA
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

        # 6. EQUIPAMENTOS
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

        # 7. ELEVADORES / TRANSPORTE VERTICAL
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

        # 8. INCÊNDIO / SEGURANÇA
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

        # 9. SPDA / ATERRAMENTO
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

        # 10. TELECOM / DADOS
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

        # 11. GÁS
        elif any(w in texto for w in [
            "gás", "gas", "glp", "gn",
            "botijão", "botijao",
            "central de gás", "central de gas",
            "tubulação de gás", "tubulacao de gas",
            "vazamento de gás", "vazamento de gas",
            "regulador de gás", "regulador de gas"
        ]):
            return "Gás / GLP"

        # 12. COZINHA / EQUIPAMENTOS GASTRONÔMICOS
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

        # 13. PISCINA
        elif any(w in texto for w in [
            "piscina", "casa de máquinas da piscina",
            "casa de maquinas da piscina",
            "filtro da piscina", "bomba da piscina",
            "clorador", "dosador de cloro",
            "tratamento da piscina",
            "aspiração da piscina", "aspiracao da piscina"
        ]):
            return "Piscina / Tratamento de Água"

        # 14. PAISAGISMO / ÁREAS EXTERNAS
        elif any(w in texto for w in [
            "jardim", "jardinagem", "grama",
            "gramado", "árvore", "arvore",
            "poda", "paisagismo", "canteiro",
            "irrigação", "irrigacao",
            "plantio", "mato", "vegetação", "vegetacao"
        ]):
            return "Paisagismo / Áreas Externas"

        # 15. LIMPEZA / CONSERVAÇÃO
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

        # 16. HOTELARIA / HOSPEDAGEM
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

        # 17. ACESSIBILIDADE
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

        # 18. MOBILIÁRIO
        elif any(w in texto for w in [
            "cadeira", "mesa", "armário", "armario",
            "gaveta", "estante", "balcão", "balcao",
            "banco", "mobiliário", "mobiliario",
            "móvel", "movel", "escrivaninha",
            "arquivo", "prateleira"
        ]):
            return "Mobiliário"

        # 19. SEGURANÇA PATRIMONIAL
        elif any(w in texto for w in [
            "segurança patrimonial", "seguranca patrimonial",
            "alarme", "sensor de presença",
            "sensor de presenca", "cerca elétrica",
            "cerca eletrica", "portão eletrônico",
            "portao eletronico", "fechamento eletrônico",
            "fechamento eletronico"
        ]):
            return "Segurança Patrimonial"

        # 20. OUTROS
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
    amosta = [str(obs)[:120] for obs in lista_observacoes[:20]]
    texto_observacoes = "\n".join([f"- {obs}" for obs in amosta])

    prompt = f"""Você é um especialista em manutenção de Ordens de Serviço (OS).
Foram encontrados {len(lista_observacoes)} registros no total para o termo '{termo.upper()}'. Abaixo estão exemplos representativos:
{texto_observacoes}

Escreva um relatório executivo detalhado em português contendo:
1. Resumo dos Principais Problemas Relatados (Inclua obrigatoriamente uma tabela em Markdown com colunas 'Área/Problema' e 'Frequência')
2. Padrões ou Causas Recorrentes
3. Recomendações e Ações Preventivas"""

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
# INTERFACE GRÁFICA (BARRA LATERAL)
# ==========================================
st.sidebar.title("⚙️ Configurações & Filtros")

chave_default = os.getenv("GROQ_API_KEY", "")
groq_api_key_input = st.sidebar.text_input("🔑 Chave API Groq", value=chave_default, type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Carregar Dados")
arquivo_carregado = st.sidebar.file_uploader("Envie sua planilha Excel (.xls, .xlsx)", type=["xls", "xlsx"])

# ==========================================
# CORPO PRINCIPAL DO APLICATIVO
# ==========================================
st.markdown('<p class="main-header">🛠️ Gestão da Manutenção - Visão Geral & Painel Analítico</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Classificação NLP em lote, indicadores executivos, gráficos dinâmicos e chat com IA.</p>', unsafe_allow_html=True)

if 'termo_pesquisa' not in st.session_state:
    st.session_state.termo_pesquisa = ""

if 'mensagens_chat' not in st.session_state:
    st.session_state.mensagens_chat = []

def limpar_pesquisa():
    if "relatorio_gerado" in st.session_state:
        del st.session_state.relatorio_gerado
    if "termo_relatorio" in st.session_state:
        del st.session_state.termo_relatorio

# Carregamento do arquivo
df = None
coluna_comentario = "OBSERVAÇÃO ABERTURA"

if arquivo_carregado is not None:
    try:
        df = pd.read_excel(arquivo_carregado)
        st.success(f"✅ Arquivo `{arquivo_carregado.name}` carregado com sucesso ({len(df):,} registros)!")
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo enviado: {e}")
elif os.path.exists("OS Geral.xls"):
    try:
        df = pd.read_excel("OS Geral.xls")
        st.info(f"ℹ️ Usando o arquivo padrão local: `OS Geral.xls` ({len(df):,} registros)")
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo local `OS Geral.xls`: {e}")
else:
    st.warning("⚠️ Por favor, envie uma planilha na barra lateral para começar.")

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

        df["TEXTO_BUSCA"] = df[coluna_comentario].astype(str).str.lower()

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Filtros Globais do Dashboard")
        
        df_filtrado_dashboard = df.copy()
        
        if "STATUS" in df.columns:
            lista_status = ["Todos"] + list(df["STATUS"].dropna().unique())
            status_escolhido = st.sidebar.selectbox("Filtrar por Status", lista_status)
            if status_escolhido != "Todos":
                df_filtrado_dashboard = df_filtrado_dashboard[df_filtrado_dashboard["STATUS"] == status_escolhido]

        if "CATEGORIA_NLP" in df.columns:
            lista_cat = ["Todas"] + list(df["CATEGORIA_NLP"].dropna().unique())
            cat_escolhida = st.sidebar.selectbox("Filtrar por Categoria NLP", lista_cat)
            if cat_escolhida != "Todas":
                df_filtrado_dashboard = df_filtrado_dashboard[df_filtrado_dashboard["CATEGORIA_NLP"] == cat_escolhida]

        st.markdown("---")
        
        aba_dash, aba_busca, aba_nlp, aba_chat = st.tabs([
            "📊 Dashboard Geral (KPIs & Gráficos)",
            "🔍 Pesquisa & Relatórios por Termo", 
            "🏷️ Classificação Automática (NLP)", 
            "💬 Chat Inteligente com a Base Completa"
        ])

        # ==========================================
        # ABA 1: DASHBOARD GERAL
        # ==========================================
        with aba_dash:
            st.subheader("📈 Visão Executiva Completa da Manutenção")
            
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            
            total_os = len(df_filtrado_dashboard)
            
            if "CUSTO COM MATERIAIS" in df_filtrado_dashboard.columns:
                val_mat = limpar_coluna_monetaria(df_filtrado_dashboard["CUSTO COM MATERIAIS"]).sum()
                custo_materiais = f"R$ {val_mat:,.2f}"
            else:
                custo_materiais = "R$ 0,00"

            if "MÃO DE OBRA EXTERNA" in df_filtrado_dashboard.columns:
                val_moe = limpar_coluna_monetaria(df_filtrado_dashboard["MÃO DE OBRA EXTERNA"]).sum()
                mao_de_obra = f"R$ {val_moe:,.2f}"
            else:
                mao_de_obra = "R$ 0,00"
            
            with col_kpi1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">TOTAL DE OS</h4>
                        <h2 style="color: #1E3A8A; font-size: 26px; margin: 0;">{total_os:,}</h2>
                    </div>
                """, unsafe_allow_html=True)
            with col_kpi2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">CUSTO MATERIAIS</h4>
                        <h2 style="color: #059669; font-size: 26px; margin: 0;">{custo_materiais}</h2>
                    </div>
                """, unsafe_allow_html=True)
            with col_kpi3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">MÃO DE OBRA EXTERNA</h4>
                        <h2 style="color: #D97706; font-size: 26px; margin: 0;">{mao_de_obra}</h2>
                    </div>
                """, unsafe_allow_html=True)
            with col_kpi4:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #6B7280; font-size: 14px; margin-bottom: 5px;">VOLUME FILTRADO</h4>
                        <h2 style="color: #DC2626; font-size: 26px; margin: 0;">{total_os} OS</h2>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown("### 🥧 Proporção de OS por Categoria NLP")
                if "CATEGORIA_NLP" in df_filtrado_dashboard.columns:
                    df_pie = df_filtrado_dashboard["CATEGORIA_NLP"].value_counts().reset_index()
                    df_pie.columns = ["Categoria", "Quantidade"]
                    fig_pie = px.pie(df_pie, names="Categoria", values="Quantidade", hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Dado de categoria não disponível.")

            with col_g2:
                st.markdown("### 📊 Ocorrências por Status")
                if "STATUS" in df_filtrado_dashboard.columns:
                    df_bar = df_filtrado_dashboard["STATUS"].value_counts().reset_index()
                    df_bar.columns = ["Status", "Quantidade"]
                    fig_bar = px.bar(df_bar, x="Status", y="Quantidade", text="Quantidade", color="Status")
                    fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Dado de status não disponível.")

        # ==========================================
        # ABA 2: PESQUISA & RELATÓRIOS POR TERMO
        # ==========================================
        with aba_busca:
            st.subheader("🔍 Filtrar Registros por Palavra-Chave")
            termo_busca = st.text_input("Digite o termo a pesquisar (ex: 'vazamento', 'ar condicionado', 'elevador'):", key="input_termo")
            
            if termo_busca:
                df_busca = df[df["TEXTO_BUSCA"].str.contains(termo_busca.lower(), na=False)]
                st.write(f"Encontrados **{len(df_busca)}** registros com o termo '{termo_busca}'.")
                st.dataframe(df_busca[[c for c in ["CÓDIGO", "STATUS", "CATEGORIA_NLP", coluna_comentario] if c in df_busca.columns]], use_container_width=True)
                
                if st.button("🤖 Gerar Relatório IA sobre estas OS"):
                    with st.spinner("Gerando relatório com IA..."):
                        observacoes = df_busca[coluna_comentario].tolist()
                        relatorio = gerar_relatorio_ia(termo_busca, observacoes, groq_api_key_input)
                        st.markdown(relatorio)

        # ==========================================
        # ABA 3: CLASSIFICAÇÃO AUTOMÁTICA (NLP)
        # ==========================================
        with aba_nlp:
            st.subheader("🏷️ Detalhes da Classificação Automática")
            st.dataframe(df[["CATEGORIA_NLP", coluna_comentario]], use_container_width=True)

        # ==========================================
        # ABA 4: CHAT INTELIGENTE COM A BASE COMPLETA
        # ==========================================
        with aba_chat:
            st.subheader("💬 Chat com IA sobre toda a base de Ordens de Serviço")
            
            for msg in st.session_state.mensagens_chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            if prompt_usuario := st.chat_input("Faça uma pergunta sobre as OS (ex: 'Qual a categoria com mais problemas?'):"):
                st.session_state.mensagens_chat.append({"role": "user", "content": prompt_usuario})
                with st.chat_message("user"):
                    st.markdown(prompt_usuario)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analisando dados..."):
                        resposta = responder_pergunta_livre_com_todo_arquivo(prompt_usuario, df, groq_api_key_input)
                        st.markdown(resposta)
                        st.session_state.mensagens_chat.append({"role": "assistant", "content": resposta})
