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
        if any(w in texto for w in ["vazamento", "agua", "cano", "infiltracao", "esgoto", "torneira", "registro", "valvula", "hydra", "ralo", "bebedouro"]):
            return "Hidráulica / Saneamento"
        elif any(w in texto for w in ["luz", "lampada", "disjuntor", "tomada", "energia", "curto", "quadro eletrico", "fio", "cabo", "iluminacao"]):
            return "Elétrica"
        elif any(w in texto for w in ["ar condicionado", "split", "climatizacao", "temperatura", "geladeira", "ventilador"]):
            return "Climatização / Refrigeração"
        elif any(w in texto for w in ["porta", "janela", "fechadura", "piso", "parede", "teto", "telhado", "vidro", "pintura", "civil"]):
            return "Estrutural / Civil"
        elif any(w in texto for w in ["motor", "bomba", "correia", "peca", "maquina", "equipamento", "mecanica"]):
            return "Mecânica / Equipamentos"
        elif any(w in texto for w in ["limpeza", "lixo", "entulho", "higienizacao"]):
            return "Limpeza / Conservação"
        elif any(w in texto for w in ["hospede", "hospedes", "quarto", "suite", "hospedagem", "cliente"]):
            return "Atendimento ao Hóspede / Quartos"
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
    
    resumo_status = df["STATUS"].value_counts().to_dict() if "STATUS" in df.columns else "Não disponível"
    resumo_setor = df["SETOR"].value_counts().to_dict() if "SETOR" in df.columns else (df["MÁQUINA"].value_counts().to_dict() if "MÁQUINA" in df.columns else "Não disponível")
    resumo_nlp = df["CATEGORIA_NLP"].value_counts().to_dict() if "CATEGORIA_NLP" in df.columns else "Não disponível"
    
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

    df_amostra = df_relevante.head(12).copy()
    if col_obs in df_amostra.columns:
        df_amostra[col_obs] = df_amostra[col_obs].astype(str).str.slice(0, 90)

    colunas_visiveis = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "MÁQUINA", "CATEGORIA_NLP", col_obs] if c in df_amostra.columns]
    amostra_relevante = df_amostra[colunas_visiveis].to_string(index=False)

    prompt = f"""Você é um analista especialista em gestão de manutenção. A planilha completa foi 100% processada com NLP e possui os seguintes dados consolidados:

--- ESTATÍSTICAS GLOBAIS DA PLANILHA INTEIRA ({total_linhas} registros analisados) ---
- Total de registros: {total_linhas}
- Período coberto: {periodo_dados}
- Distribuição Completa por Categoria NLP: {resumo_nlp}
- Distribuição Completa por Status: {resumo_status}
- Distribuição Completa por Setor/Máquina: {resumo_setor}
-------------------------------------------------------------------------------------

--- AMOSTRA DOS REGISTROS MAIS DIRETAMENTE RELACIONADOS À PERGUNTA ({len(df_relevante)} encontrados no total) ---
{amostra_relevante}
---------------------------------------------------------------------------------------------------------------

Com base em **todos os dados consolidados da planilha completa** e nas estatísticas analíticas acima, responda de forma precisa, técnica, com números exatos e em português à pergunta abaixo:
Pergunta: {pergunta_usuario}"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo_ativo,
            temperature=0.2,
            max_tokens=1000,
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
    # Normalizar nomes das colunas para maiúsculas para evitar erros de case-sensitive
    df.columns = [str(c).strip().upper() for c in df.columns]
    
    if coluna_comentario not in df.columns:
        st.error(f"❌ A coluna obrigatória **'{coluna_comentario}'** não foi encontrada na planilha.")
        st.write(f"📋 **Colunas disponíveis na planilha:** `{list(df.columns)}`")
    else:
        df = df.dropna(subset=[coluna_comentario]).copy()
        
        # Processamento automático NLP
        if "CATEGORIA_NLP" not in df.columns:
            with st.spinner(f"Processando algoritmo de classificação NLP em {len(df):,} registros..."):
                df = classificar_texto_nlp(df, coluna_comentario)

        df["TEXTO_BUSCA"] = df[coluna_comentario].astype(str).str.lower()

        # Filtros Globais na Barra Lateral
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
        
        # Abas Principais
        aba_dash, aba_busca, aba_nlp, aba_chat = st.tabs([
            "📊 Dashboard Geral (KPIs & Gráficos)",
            "🔍 Pesquisa & Relatórios por Termo", 
            "🏷️ Classificação Automática (NLP)", 
            "💬 Chat Inteligente com a Base Completa"
        ])

        with aba_dash:
            st.subheader("📈 Visão Executiva Completa da Manutenção")
            
            # 1. Métricas Principais (KPI Cards Calculadas Dinamicamente)
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
            
            total_os = len(df_filtrado_dashboard)
            
            # Cálculo de Custo com Materiais real com segurança
            if "CUSTO COM MATERIAIS" in df_filtrado_dashboard.columns:
                val_mat = limpar_coluna_monetaria(df_filtrado_dashboard["CUSTO COM MATERIAIS"]).sum()
                custo_materiais = f"R$ {val_mat:,.2f}"
            else:
                custo_materiais = "R$ 0,00"

            # Cálculo de Mão de Obra Externa real com segurança
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

            # 2. Gráfico de Pizza e Barras
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown("### 🥧 Proporção de OS por Categoria NLP (Pizza)")
                if "CATEGORIA_NLP" in df_filtrado_dashboard.columns:
                    df_pie = df_filtrado_dashboard["CATEGORIA_NLP"].value_counts().reset_index()
                    df_pie.columns = ["Categoria", "Quantidade"]
                    fig_pie = px.pie(df_pie, names="Categoria", values="Quantidade", hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Dado de categoria não disponível.")

            with col_g2:
                st.markdown("### 📊 Ocorrências por Status (Barras)")
                if "STATUS" in df_filtrado_dashboard.columns:
                    df_bar = df_filtrado_dashboard["STATUS"].value_counts().reset_index()
                    df_bar.columns = ["Status", "Quantidade"]
                    fig_bar = px.bar(df_bar, x="Status", y="Quantidade", text="Quantidade", color="Status", color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Dado de status não disponível.")

            st.markdown("---")

            # 3. Gráficos de Linha Temporal e Barras Horizontais (Máquina / Setor)
            col_g3, col_g4 = st.columns(2)
            
            with col_g3:
                st.markdown("### 📈 Evolução de Ocorrências ao Longo do Tempo (Linhas)")
                col_data = "ABERTO EM" if "ABERTO EM" in df_filtrado_dashboard.columns else ("DATA" if "DATA" in df_filtrado_dashboard.columns else None)
                if col_data:
                    try:
                        df_temp = df_filtrado_dashboard.copy()
                        df_temp["DATA_FORMATADA"] = pd.to_datetime(df_temp[col_data], errors='coerce').dt.to_period("M").astype(str)
                        df_line = df_temp["DATA_FORMATADA"].value_counts().sort_index().reset_index()
                        df_line.columns = ["Mês", "Chamados"]
                        fig_line = px.line(df_line, x="Mês", y="Chamados", markers=True, line_shape="spline", color_discrete_sequence=["#1E3A8A"])
                        fig_line.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
                        st.plotly_chart(fig_line, use_container_width=True)
                    except Exception:
                        st.info("Formato de data inválido para gráfico temporal.")
                else:
                    st.info("Coluna de data ('ABERTO EM') não encontrada.")

            with col_g4:
                st.markdown("### ⚙️ Solicitações por Máquina / Local (Barras Horizontais)")
                coluna_agrupamento = "MÁQUINA" if "MÁQUINA" in df_filtrado_dashboard.columns else ("SETOR" if "SETOR" in df_filtrado_dashboard.columns else "CATEGORIA_NLP")
                df_hbar = df_filtrado_dashboard[coluna_agrupamento].value_counts().reset_index().head(8)
                df_hbar.columns = ["Máquina", "Quantidade"]
                fig_hbar = px.bar(df_hbar, x="Quantidade", y="Máquina", orientation="h", text="Quantidade", color_discrete_sequence=["#059669"])
                fig_hbar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_hbar, use_container_width=True)

        with aba_busca:
            col_input, col_btn = st.columns([3, 1])
            with col_input:
                termo_usuario = st.text_input(
                    "🔍 Digite o termo de busca (ex: vazamento, ralo, civil, elétrica):", 
                    key="termo_pesquisa",
                    on_change=limpar_pesquisa
                ).strip().lower()
                
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True) 
                st.button("🔄 Reiniciar Pesquisa", on_click=limpar_pesquisa, use_container_width=True)

            if termo_usuario:
                df_filtrado = df[df["TEXTO_BUSCA"].str.contains(termo_usuario, na=False)]
                total_encontrados = len(df_filtrado)

                st.markdown(f"### 📊 Resultados para: `{termo_usuario.upper()}` (Base Completa: {total_encontrados:,} ocorrências)")
                st.metric(label="Total de Ocorrências Encontradas", value=total_encontrados)

                if total_encontrados > 0:
                    colunas_exibicao = [c for c in ["CÓDIGO", "ABERTO EM", "STATUS", "MÁQUINA", "CATEGORIA_NLP", coluna_comentario] if c in df.columns]

                    st.markdown("---")
                    st.subheader("📈 Análise Gráfica dos Chamados Filtrados")
                    col_sub1, col_sub2 = st.columns(2)
                    
                    with col_sub1:
                        if "STATUS" in df_filtrado.columns:
                            st.markdown("**Ocorrências por Status (Pizza)**")
                            df_p_sub = df_filtrado["STATUS"].value_counts().reset_index()
                            df_p_sub.columns = ["Status", "Qtd"]
                            fig_p_sub = px.pie(df_p_sub, names="Status", values="Qtd", hole=0.3)
                            fig_p_sub.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                            st.plotly_chart(fig_p_sub, use_container_width=True)

                    with col_sub2:
                        st.markdown("**Ocorrências por Categoria NLP (Barras)**")
                        df_b_sub = df_filtrado["CATEGORIA_NLP"].value_counts().reset_index()
                        df_b_sub.columns = ["Categoria", "Qtd"]
                        fig_b_sub = px.bar(df_b_sub, x="Categoria", y="Qtd", text="Qtd", color_discrete_sequence=["#3B82F6"])
                        fig_b_sub.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                        st.plotly_chart(fig_b_sub, use_container_width=True)

                    with st.expander(f"📋 Ver registros detalhados ({total_encontrados:,} encontrados)", expanded=False):
                        st.dataframe(df_filtrado[colunas_exibicao], use_container_width=True)

                    st.markdown("---")
                    if st.button("🤖 Gerar Relatório Executivo com IA", type="primary"):
                        with st.spinner("Analisando todos os registros e gerando relatório executivo..."):
                            lista_obs = df_filtrado[coluna_comentario].astype(str).tolist()
                            relatorio = gerar_relatorio_ia(termo_usuario, lista_obs, groq_api_key_input)
                            st.session_state.relatorio_gerado = relatorio
                            st.session_state.termo_relatorio = termo_usuario

                    if "relatorio_gerado" in st.session_state and st.session_state.get("termo_relatorio") == termo_usuario:
                        st.markdown("---")
                        st.subheader("📋 Resumo dos Principais Problemas Relatados & Relatório IA")
                        st.markdown(st.session_state.relatorio_gerado)

                        dados_grafico = extrair_dados_tabela_markdown(st.session_state.relatorio_gerado)
                        if dados_grafico is not None and not dados_grafico.empty:
                            st.markdown("---")
                            st.subheader("📊 Gráficos a partir do Resumo dos Principais Problemas Relatados")
                            fig_rel = px.bar(dados_grafico.reset_index(), x="Categoria", y="Frequência", text="Frequência", color_discrete_sequence=["#10B981"])
                            fig_rel.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
                            st.plotly_chart(fig_rel, use_container_width=True)
                else:
                    st.warning("⚠️ Nenhum registro encontrado com esse termo na planilha.")

        with aba_nlp:
            st.subheader("🏷️ Visão Geral da Classificação NLP (100% dos Registros)")
            st.markdown(f"O algoritmo de NLP categorizou automaticamente **{len(df):,} registros** com base no conteúdo textual de `{coluna_comentario}`.")
            
            contagem_nlp = df["CATEGORIA_NLP"].value_counts()
            col_met1, col_met2 = st.columns([2, 1])
            
            with col_met1:
                st.markdown("**📊 Distribuição por Categoria (Barras)**")
                df_nlp_bar = contagem_nlp.reset_index()
                df_nlp_bar.columns = ["Categoria", "Quantidade"]
                fig_nlp_bar = px.bar(df_nlp_bar, x="Quantidade", y="Categoria", orientation="h", text="Quantidade", color_discrete_sequence=["#6366F1"])
                fig_nlp_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_nlp_bar, use_container_width=True)
                
            with col_met2:
                st.markdown("**📋 Resumo Numérico**")
                st.dataframe(contagem_nlp.reset_index(name="Quantidade"), use_container_width=True)

            st.markdown("---")
            st.subheader("📥 Exportar Dados Classificados")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='OS Classificadas NLP')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Baixar Planilha Completa com Coluna 'CATEGORIA_NLP' (.xlsx)",
                data=buffer,
                file_name="OS_Classificadas_NLP.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        with aba_chat:
            st.subheader("💬 Chat Inteligente com a Base Completa de Manutenção")
            st.markdown(f"Faça perguntas abertas sobre os **{len(df):,} registros** da planilha.")

            for mensagem in st.session_state.mensagens_chat:
                with st.chat_message(mensagem["role"]):
                    st.markdown(mensagem["content"])

            if prompt_usuario := st.chat_input("Digite sua pergunta sobre a planilha inteira..."):
                st.session_state.mensagens_chat.append({"role": "user", "content": prompt_usuario})
                with st.chat_message("user"):
                    st.markdown(prompt_usuario)

                with st.chat_message("assistant"):
                    with st.spinner(f"Varrendo os {len(df):,} registros da planilha para responder..."):
                        resposta_ia = responder_pergunta_livre_com_todo_arquivo(prompt_usuario, df, groq_api_key_input)
                        st.markdown(resposta_ia)
                
                st.session_state.mensagens_chat.append({"role": "assistant", "content": resposta_ia})
                        st.markdown(resposta_ia)
                
                st.session_state.mensagens_chat.append({"role": "assistant", "content": resposta_ia})
