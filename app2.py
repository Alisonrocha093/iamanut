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
    """Busca dinamicamente na API da Groq um modelo de texto ativo e disponível."""
    if not api_key:
        return "llama-3.1-8b-instant"
    try:
        modelos_disponiveis = client.models.list()
        for m in modelos_disponiveis.data:
            m_id = m.id.lower()
            if any(nome in m_id for nome in ["llama-3.1", "llama-3.3", "gpt-oss", "qwen"]) and not any(ign in m_id for ign in ["guard", "safeguard", "whisper", "vision", "embed"]):
                return m.id
    except Exception:
        pass
    return "llama-3.1-8b-instant"

def gerar_relatorio_ia(termo: str, lista_observacoes: list, api_key: str) -> str:
    """Gera o relatório executivo baseado em um termo específico."""
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral."

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}"

    modelo_ativo = obter_modelo_ativo(client, api_key)

    amostra = [str(obs)[:200] for obs in lista_observacoes[:20]]
    texto_observacoes = "\n".join([f"- {obs}" for obs in amostra])

    prompt = f"""Você é um especialista em manutenção e análise de Ordens de Serviço (OS).
Abaixo estão {len(amostra)} chamados encontrados para o termo '{termo.upper()}':

--- REGISTROS ---
{texto_observacoes}
--- FIM DOS REGISTROS ---

Escreva um relatório executivo em português seguindo rigorosamente a estrutura abaixo:
1. Resumo dos Principais Problemas Relatados
(IMPORTANTE: Nesta seção, inclua obrigatoriamente uma tabela em Markdown contendo colunas como 'Área' ou 'Problema' e uma coluna 'Frequência' contendo valores numéricos seguidos de 'ocorrências')
2. Padrões ou Causas Recorrentes
3. Recomendações e Ações Preventivas para a Equipe"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelo_ativo,
            temperature=0.2,
            max_tokens=1600,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro ao gerar relatório com a IA: {e}"

def responder_pergunta_livre(pergunta_usuario: str, df: pd.DataFrame, api_key: str) -> str:
    """Permite que a IA responda qualquer pergunta sobre o conteúdo geral da planilha."""
    if not api_key:
        return "⚠️ Chave da Groq não configurada! Insira sua chave na barra lateral."

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"❌ Erro ao inicializar o cliente Groq: {e}"

    modelo_ativo = obter_modelo_ativo(client, api_key)

    # Prepara um resumo amostral dos dados da planilha para dar contexto à IA
    # Convertemos uma amostra das linhas principais para texto estruturado
    total_linhas = len(df)
    amostra_df = df.head(50).to_string(index=False)

    prompt = f"""Você é um assistente analítico especialista em gestão de manutenção.
Você tem acesso a uma base de dados de Ordens de Serviço (OS) com um total de {total_linhas} registros.
Abaixo está uma amostra dos dados disponíveis na planilha:

{amostra_df}

Responda à seguinte pergunta do usuário com base nos dados fornecidos ou no seu conhecimento técnico de manutenção correlato. Seja direto, claro e objetivo em português.

Pergunta do usuário: {pergunta_usuario}"""

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
st.markdown("Consulte registros, gere relatórios e tire dúvidas livres sobre a sua planilha com Inteligência Artificial.")

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
        st.success(f"✅ Arquivo `{arquivo_carregado.name}` carregado com sucesso via upload!")
    except Exception as e:
        st.error(f"❌ Erro ao ler o arquivo enviado: {e}")
elif os.path.exists("OS Geral.xls"):
    try:
        df = pd.read_excel("OS Geral.xls")
        st.info("ℹ️ Usando o arquivo padrão local: `OS Geral.xls`")
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
        
        # Abas para Organizar: Pesquisa/Relatórios vs Chat Inteligente Geral
        aba_busca, aba_chat = st.tabs(["🔍 Pesquisa & Relatórios por Termo", "💬 Chat Livre com a Planilha (IA)"])

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

                st.markdown(f"### 📊 Resultados para: `{termo_usuario.upper()}`")
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

                    with st.expander("📋 Ver registros detalhados encontrados", expanded=False):
                        st.dataframe(df_filtrado[colunas_exibicao], use_container_width=True)

                    st.markdown("---")
                    if st.button("🤖 Gerar Relatório Executivo com IA", type="primary"):
                        with st.spinner("Analisando dados e gerando relatório executivo..."):
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
            st.subheader("💬 Tire qualquer dúvida sobre a planilha de manutenção")
            st.markdown("Faça perguntas abertas sobre os dados carregados (ex: *'Quais os principais equipamentos com falha?'*, *'Resuma o status geral dos chamados'*).")

            # Exibe o histórico de mensagens do chat
            for mensagem in st.session_state.mensagens_chat:
                with st.chat_message(mensagem["role"]):
                    st.markdown(mensagem["content"])

            # Entrada de texto do chat do Streamlit
            if prompt_usuario := st.chat_input("Digite sua pergunta sobre a planilha..."):
                # Adiciona mensagem do usuário ao histórico
                st.session_state.mensagens_chat.append({"role": "user", "content": prompt_usuario})
                with st.chat_message("user"):
                    st.markdown(prompt_usuario)

                # Gera a resposta da IA
                with st.chat_message("assistant"):
                    with st.spinner("Analisando a planilha para responder..."):
                        resposta_ia = responder_pergunta_livre(prompt_usuario, df, groq_api_key_input)
                        st.markdown(resposta_ia)
                
                # Adiciona resposta da IA ao histórico
                st.session_state.mensagens_chat.append({"role": "assistant", "content": resposta_ia})
