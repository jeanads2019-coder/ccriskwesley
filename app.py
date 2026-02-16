import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

from explain import compute_shap_single, shap_summary
from prompts import build_credit_prompt
from llm import call_llm

@st.cache_data(show_spinner=False)
def generate_explanation(prompt: str) -> str:
    return call_llm(prompt)

# ----------------------
# CONFIGURAÇÃO INICIAL
# ----------------------
st.set_page_config(page_title="Credit Card Risk", layout="wide")

# ----------------------
# CARREGAMENTO DO MODELO
# ----------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("modelo_credito.pkl")

pipeline = load_pipeline()

# ----------------------
# PÁGINA INICIAL
# ----------------------
st.title("💳 Sistema de Decisão de Crédito")

st.markdown(""" Carregue seu arquivo CSV, ajuste o limiar de decisão,  
e receba **explicações claras** e **orientação de soluções** de crédito por um LLM local.""")

st.sidebar.header("1. Upload de Arquivos / Conexão")

data_source = st.sidebar.radio("Fonte de Dados", ["CSV Upload", "Supabase Database"], index=0)

df = None

if data_source == "CSV Upload":
    file = st.sidebar.file_uploader("Arraste seu CSV de Teste aqui", type="csv")
    if df is not None:
        df = pd.read_csv(file)
        st.subheader("Dados carregados (CSV)")
        st.dataframe(df.head())

else:
    # Supabase Connection
    from sqlalchemy import create_engine, text
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    db_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
    
    if st.sidebar.button("Carregar do Supabase"):
        if not db_url:
            st.error("DATABASE_URL não configurada no .env")
        else:
            try:
                with st.spinner("Baixando dados do Supabase..."):
                    engine = create_engine(db_url)
                    query = "SELECT * FROM credit_clients LIMIT 500" # Limiting for demo perf
                    df = pd.read_sql(query, engine)
                    
                    # Rename back to CSV format expected by model
                    if "default_payment_next_month" in df.columns:
                        df.rename(columns={"default_payment_next_month": "default.payment.next.month"}, inplace=True)
                        
                    st.subheader("Dados carregados (Supabase - Top 500)")
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"Erro ao conectar: {e}")

if df is not None:
    # Remove processamento duplicado abaixo e garante fluxo único
    pass

    st.sidebar.markdown("---")
    st.sidebar.header("2. Parâmetros de Negócio")
    
    threshold = st.sidebar.slider("Risco Máximo Aceitável (Corte)", 0.0, 1.0, 0.08, 0.01)
    ticket_medio = st.sidebar.number_input("Lucro por Cliente (R$)", value=100)
    prejuizo_medio = st.sidebar.number_input("Prejuízo por Calote (R$)", value=1000)

    # ----------------------
    # PREDICTION
    # ----------------------
    X_input = df.drop(
        columns=["default.payment.next.month", "ID"], errors="ignore")
    y_true = df['default.payment.next.month']
    
    probs = pipeline.predict_proba(X_input)[:, 1]

    df["default_probability"] = probs
    df["decision"] = np.where(probs >= threshold, "Reprovado", "Aprovado")

    st.subheader("Decisões do Modelo")
    st.dataframe(df)




    # --- 5. APLICANDO O CORTE ---
    decisao_modelo = (probs >= threshold).astype(int)
    
    # Cálculos Reais
    tn, fp, fn, tp = confusion_matrix(y_true, decisao_modelo).ravel()
    
    total = len(df)
    total_aprovados = tn + fn
    taxa_aprovacao = (total_aprovados / total) * 100
    taxa_inadimplencia = (fn / total_aprovados * 100) if total_aprovados > 0 else 0
    resultado_financeiro = (tn * ticket_medio) - (fn * prejuizo_medio)






    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes Processados", f"{total}")
    c2.metric("Taxa de Aprovação", f"{taxa_aprovacao:.1f}%")
    c3.metric("Inadimplência Real da Carteira", f"{taxa_inadimplencia:.2f}%", delta_color="inverse")
    c4.metric("Resultado Financeiro", f"R$ {resultado_financeiro:,.2f}")

    st.markdown("---")
    
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("Distribuição de Risco dos Clientes")
        fig_hist = px.histogram(x=probs, color=y_true.astype(str), nbins=50,
                                labels={'x': 'Probabilidade de Risco', 'color': 'Realmente Inadimplente?'},
                                color_discrete_map={'0': 'green', '1': 'red'},
                                opacity=0.6, title="Separação de Risco)")
        fig_hist.add_vline(x=threshold, line_dash="dash", annotation_text="CORTE")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        st.subheader("Simulação de Carteira")
        fig_pie = go.Figure(data=[go.Pie(labels=['Bons Aprovados', '[ERRO] Maus Aprovados (Erro)', 'Rejeitados'], 
                                         values=[tn, fn, tp+fp], 
                                         hole=.4, marker_colors=['green', 'red', 'gray'])])
        st.plotly_chart(fig_pie, use_container_width=True)

    # Tabela Final Exportável
    with st.expander("Ver Dados Detalhados"):
        df_final = df.copy()
        df_final['Score_Risco'] = probs
        df_final['Decisao_Simulada'] = ['REPROVADO' if x == 1 else 'APROVADO' for x in decisao_modelo]
        st.dataframe(df_final.head(100))




    # ----------------------
    # CLIENT SELECTION
    # ----------------------
    idx = st.selectbox(
        "Selecione o ID do Cliente",
        df.index
    )

    client = df.loc[idx]
    prob = client["default_probability"]
    decision = client["decision"]

    st.markdown(f"""
### 📌 Decisão para o cliente selecionado:
- **Decissão:** `{decision}`
- **Probabilidade de dar default:** `{prob:.2f}`
- **Threshold:** `{threshold}`
""")

   

    # ----------------------
    # LLM EXPLANATION
    # ----------------------
    if st.button("Gerar Explicação (LLM)"):
        with st.spinner("Carregando SHAP + LLM..."):

            # --- SHAP ---
            X_client = X_input.loc[[idx]]

            shap_values, X_transformed = compute_shap_single(
                pipeline,
                X_client
            )


            if isinstance(shap_values, list):
                shap_vals_client = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                shap_vals_client = shap_values[0, :, 1]
            else:
                shap_vals_client = shap_values[0]


            feature_names = pipeline.named_steps[
                "preprocessor"
            ].get_feature_names_out()

            shap_df = pd.DataFrame({
                "feature": feature_names,
                "shap_value": shap_vals_client
            })

            client_summary = (
                shap_df
                .assign(abs_val=lambda x: x.shap_value.abs())
                .sort_values("abs_val", ascending=False)
                .head(5)
                .drop(columns="abs_val")
                .to_dict(orient="records")
            )

            # --- Mostra fatores ---
            st.subheader("Principais Fatores (Baseados no Modelo)")
            st.table(pd.DataFrame(client_summary))

            # --- LLM ---
            prompt = build_credit_prompt(
                decision=decision,
                prob=prob,
                threshold=threshold,
                factors=client_summary
            )

            explanation = generate_explanation(prompt)

        st.subheader("📄 Explicação")
        st.write(explanation)