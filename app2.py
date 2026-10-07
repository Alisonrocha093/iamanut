import os
import io
import pandas as pd
import streamlit as st
from groq import Groq

# ==========================================
# CONFIGURAÇÃO DA PÁGINA (STREAMLIT)
# ==========================================
st.set_page_config(
    page_title="Gerenciador de OS & Relatórios IA",
    page_icon="🛠️",
    layout="wide"
)

# ==========================================
# CONFIGURAÇÃO DA IA (GROQ)
# ==========================================
def obter_modelo_ativo(client, api_key: str) -> str:
    """Busca dinamicamente na API da Groq um modelo leve e adequado."""
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
    """Gera o relatório executivo baseado nas ocorrências encontradas."""
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral."

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}"

    modelo_ativo = obter_modelo_ativo(client, api_key)

    amostra = [str(obs)[:150] for obs in lista_observacoes[:30]]
    texto_observacoes = "\n".join([f"- {obs}" for obs in amostra])

    prompt = f"""Você é um especialista em manutenção de Ordens de Serviço (OS).
Foram encontrados {len(lista_observacoes)} registros no total para o termo '{termo.upper()}'. Abaixo estão alguns exemplos representativos:
{texto_observacoes}

Escreva um relatório executivo detalhado em português contendo:
1. Resumo dos Principais Problemas Relatados (Inclua obrigatoriamente uma tabela em Markdown com colunas 'Área/Problema' e 'Frequência' ex: | Área | Frequência | / | --- | --- | / | Hidráulica | 5 ocorrências |)
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
    """Analisa todo o arquivo processando estatísticas completas e buscando linhas relevantes."""
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral."

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}"

    modelo_ativo = obter_modelo_ativo(client, api_key)

    total_linhas = len(df)
    
    # 1. Agregações estatísticas globais de todo o arquivo
    resumo_status = df["STATUS"].value_counts().to_dict() if "STATUS" in df.columns else "Não disponível"
    resumo_setor = df["SETOR"].value_counts().head(5).to_dict() if "SETOR" in df.columns else (df["TIPO"].value_counts().head(5).to_dict() if "TIPO" in df.columns else "Não disponível")

    # 2. Busca inteligente por termos mencionados na pergunta do usuário em todo o DataFrame
    pergunta_lower = pergunta_usuario.lower()
    palavras_chave = [palavra for palavra in pergunta_lower.split() if len(palavra) > 3]
    
    df_relevante = df
    if palavras_chave:
        # Filtra linhas que contenham alguma palavra-chave da pergunta do usuário nas observações
        filtro = df["OBSERVAÇÃO ABERTURA"].astype(str).str.lower().apply(lambda x: any(p in x for p in palavras_chave))
        df_filtrado_ia = df[filtro]
        if len(df_filtrado_ia) > 0:
            df_relevante = df_filtrado_ia

    # Pega uma amostra focada dos registros mais relevantes encontrados na base completa (até 25 linhas)
    amostra_relevante = df_relevante[["OS", "DATA", "STATUS", "SETOR", "OBSERVAÇÃO ABERTURA"]].head(25).to_string(index=False) if "SETOR" in df.columns else df_relevante.head(25).to_string(index=False)

    prompt = f"""Você é um analista especialista em gestão de manutenção. A planilha completa possui {total_linhas} registros analisados.

--- ESTATÍSTICAS GLOBAIS DA PLANILHA COMPLETA ---
- Total de registros: {total_linhas}
- Distribuição por Status: {resumo_status}
- Principais Setores/Tipos: {resumo_setor}
--------------------------------------------------

--- REGISTROS MAIS RELEVANTES ENCONTRADOS PARA A SUA PERGUNTA ---
{amostra_relevante}
-----------------------------------------------------------------

Com base em **todos os dados da planilha completa** e nas estatísticas acima, responda de forma clara, técnica e objetiva em português à seguinte pergunta do usuário:
Pergunta: {pergunta_usuario}"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo_ativo,
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro ao responder pergunta: {e}"

def extrair_dados_tabela_markdown(relatorio_texto: str) -> pd.DataFrame:
    """Extrai dados da tabela markdown do relatório gerado pela IA para montar os gráficos."""
    try:
        tabelas = pd.read_html(io.StringIO(relatorio_texto))
        if tabelas:
            df_tabela = tabelas[0]
            col_freq = None
            col_cat = None
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

chave_default = os.getenv("GROQ_API_KEY", "")
groq_api_key_input = st.sidebar.text_input("🔑 Chave API Groq", value=chave_default, type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Carregar Dados")
arquivo_carregado = st.sidebar.file_uploader("Envie sua planilha Excel (.xls, .xlsx)", type=["xls", "xlsx"])

# ==========================================
# CORPO PRINCIPAL DO APLICATIVO
# ==========================================
st.title("🛠️ Sistema Inteligente de Ordens de Serviço (OS)")
st.markdown("Consulte registros, analise todo o arquivo e tire dúvidas via Chat com Inteligência Artificial.")

if 'termo_pesquisa' not in st.session_state:
    st.session_state.termo_pesquisa = ""

if 'mensagens_chat' not in st.session_state:
    st.session_state.mensagens_chat = []

def limpar_pesquisa():
    st.session_state.termo_pesquisa = ""
    if "relatorio_gerado" in st.session_state:
        del st.session_state.relatorio_gerado

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
    if coluna_comentario not in df.columns:
        st.error(f"❌ A coluna obrigatória **'{coluna_comentario}'** não foi encontrada na planilha.")
        st.write(f"📋 **Colunas disponíveis na planilha:** `{list(df.columns)}`")
    else:
        df = df.dropna(subset=[coluna_comentario]).copy()
        df["texto_busca"] = df[coluna_comentario].astype(str).str.lower()

        st.markdown("---")
        
        aba_busca, aba_chat = st.tabs(["🔍 Pesquisa & Relatórios por Termo", "💬 Chat Inteligente com a Base Completa"])

        with aba_busca:
            col_input, col_btn = st.columns([3, 1])
            with col_input:
                termo_usuario = st.text_input(
                    "🔍 Digite o termo de busca (ex: vazamento, luz, ar):", 
                    key="termo_pesquisa"
                ).strip().lower()
                
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True) 
                st.button("🔄 Reiniciar Pesquisa", on_click=limpar_pesquisa, use_container_width=True)

            if termo_usuario:
                df_filtrado = df[df["texto_busca"].str.contains(termo_usuario, na=False)]
                total_encontrados = len(df_filtrado)

                st.markdown(f"### 📊 Resultados para: `{termo_usuario.upper()}` (Base Completa: {total_encontrados:,} ocorrências)")
                st.metric(label="Total de Ocorrências Encontradas", value=total_encontrados)

                if total_encontrados > 0:
                    colunas_exibicao = [c for c in ["ID", "OS", "DATA", "STATUS", coluna_comentario] if c in df.columns]
                    if not colunas_exibicao:
                        colunas_exibicao = [coluna_comentario]

                    st.markdown("---")
                    st.subheader("📈 Análise Gráfica dos Chamados")
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        if "STATUS" in df_filtrado.columns:
                            st.markdown("**Ocorrências por Status**")
                            st.bar_chart(df_filtrado["STATUS"].value_counts())
                        else:
                            st.info("ℹ️ Coluna 'STATUS' não encontrada.")

                    with col_g2:
                        coluna_cat_alternativa = None
                        for col in ["SETOR", "LOCAL", "TIPO", "EQUIPAMENTO"]:
                            if col in df_filtrado.columns:
                                coluna_cat_alternativa = col
                                break
                        
                        if coluna_cat_alternativa:
                            st.markdown(f"**Ocorrências por {coluna_cat_alternativa.title()}**")
                            st.bar_chart(df_filtrado[coluna_cat_alternativa].value_counts().head(10))
                        else:
                            if "DATA" in df_filtrado.columns:
                                st.markdown("**Ocorrências por Data**")
                                try:
                                    data_counts = pd.to_datetime(df_filtrado["DATA"]).dt.date.value_counts().sort_index()
                                    st.line_chart(data_counts)
                                except Exception:
                                    pass

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
                            st.bar_chart(dados_grafico)
                else:
                    st.warning("⚠️ Nenhum registro encontrado com esse termo na planilha.")

        with aba_chat:
            st.subheader("💬 Chat Inteligente com a Base Completa de Manutenção")
            st.markdown(f"Faça perguntas abertas sobre os **{len(df):,} registros** da planilha (ex: *'Quantas OS estão pendentes?'*, *'Quais os problemas mais comuns no setor hidráulico?'*).")

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
