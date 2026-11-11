import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import joblib
from pathlib import Path
from fpdf import FPDF

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Dashboard Manutenção Preditiva | Adécio Alberto",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS PERSONALIZADO (Tema Escuro)
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #070d2b; color: #f4f6ff; }
    [data-testid="stSidebar"] { background-color: #0a1335; border-right: 1px solid #26345f; }
    .custom-card {
        background-color: #0c1438; border: 1px solid #26345f;
        border-radius: 10px; padding: 15px; margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { color: #f4f6ff !important; font-family: 'Segoe UI', sans-serif; }
    .header-title { font-size: 24px; font-weight: bold; margin: 0; }
    .header-sub { font-size: 14px; color: #93a0c5; margin: 0; }
    .text-green { color: #42dd8b; font-weight: bold; }
    .text-yellow { color: #ffc95e; font-weight: bold; }
    .text-red { color: #ff626d; font-weight: bold; }
    .status-container { display: flex; align-items: center; background: #0c1438; border: 1px solid #26345f; border-radius: 10px; padding: 20px; }
    .status-icon { font-size: 40px; margin-right: 20px; }
    .alert-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .alert-table th { text-align: left; color: #93a0c5; padding: 8px; border-bottom: 1px solid #26345f; }
    .alert-table td { padding: 8px; border-bottom: 1px solid #111b46; }
    .badge { padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }
    .badge-normal { background-color: #1a422f; color: #42dd8b; }
    .badge-alerta { background-color: #4a3b1a; color: #ffc95e; }
    .badge-critico { background-color: #4a1a23; color: #ff626d; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNÇÕES AUXILIARES, DADOS E ML
# ==========================================
@st.cache_resource
def carregar_recursos_ml():
    """Carrega o Scaler, Selector e Modelo."""
    base_path = Path(__file__).parent
    try:
        scaler = joblib.load(base_path / 'scaler.joblib')
        selector = joblib.load(base_path / 'selector.joblib')
        modelo = joblib.load(base_path / 'random_forest_model.joblib')
        return scaler, selector, modelo
    except Exception:
        return None, None, None

@st.cache_data
def dados_fallback():
    """Dados simulados caso não haja upload de ficheiro."""
    horas = pd.date_range(end=datetime.now(), periods=10, freq='h')
    return pd.DataFrame({
        'Hora': horas,
        'Temperatura': np.random.uniform(70, 85, 10),
        'Pressao': np.random.uniform(3.0, 4.0, 10),
        'Vibracao': np.random.uniform(5.0, 6.5, 10)
    })

def prever_falha(df, scaler, selector, modelo):
    """Pipeline Real: Dados -> Tradução -> Scaler -> Selector -> Random Forest"""
    # 1. Seleciona os dados e traduz os nomes para o que o modelo espera (minúsculas)
    features_df = df[['Temperatura', 'Pressao', 'Vibracao']].tail(1).copy()
    features_df.columns = ['temperature', 'pressure', 'vibration']
    
    if scaler is not None and selector is not None and modelo is not None:
        try:
            # 2. Escalonamento
            X_scaled = scaler.transform(features_df)
            # 3. Seleção de Features
            X_selected = selector.transform(X_scaled)
            # 4. Previsão
            prob = modelo.predict_proba(X_selected)[0][1] * 100
        except Exception as e:
            st.sidebar.error(f"Erro na inferência ML: {e}")
            prob = 15.0
    else:
        # Fallback Simulado (se os ficheiros .joblib não forem encontrados)
        prob = 18.0 if (features_df['temperature'].values[0] < 85) else 75.0
        
    if prob < 25: risco, cor = "Baixo", "#42dd8b"
    elif prob < 60: risco, cor = "Médio", "#ffc95e"
    else: risco, cor = "Alto", "#ff626d"
    
    return prob, risco, cor

def gerar_pdf(df, prob_falha, risco):
    """Gera um relatório PDF com os dados e o resultado do modelo."""
    pdf = FPDF()
    pdf.add_page()
    
    # Cabeçalho (sem acentos para não quebrar o PDF)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Relatorio de Manutencao Preditiva", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt="Responsavel: Adecio Alberto", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Data de Emissao: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    pdf.ln(5)
    
    # Resumo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Resumo do Diagnostico", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Probabilidade de Falha: {prob_falha:.2f}%", ln=True)
    pdf.cell(200, 8, txt=f"Nivel de Risco: {risco}", ln=True)
    pdf.ln(10)
    
    # Tabela de Histórico
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(40, 10, "Hora", border=1)
    pdf.cell(40, 10, "Temp (C)", border=1)
    pdf.cell(40, 10, "Pressao (bar)", border=1)
    pdf.cell(40, 10, "Vibracao (mm/s)", border=1)
    pdf.ln()
    
    pdf.set_font("Arial", size=9)
    for _, row in df.tail(20).iterrows():
        hora_str = row['Hora'].strftime('%d/%m %H:%M') if pd.notnull(row['Hora']) else '-'
        pdf.cell(40, 10, hora_str, border=1)
        pdf.cell(40, 10, f"{row['Temperatura']:.1f}", border=1)
        pdf.cell(40, 10, f"{row['Pressao']:.2f}", border=1)
        pdf.cell(40, 10, f"{row['Vibracao']:.2f}", border=1)
        pdf.ln()

    return bytes(pdf.output())

# ==========================================
# 4. SIDEBAR E GESTÃO DE DADOS
# ==========================================
with st.sidebar:
    try:
        st.image("logo_kenmare.png", use_container_width=True)
        st.image("logo_uem.png", use_container_width=True)
    except:
        st.warning("Adicione 'logo_kenmare.png' e 'logo_uem.png' à pasta.")
    
    st.divider()
    
    st.markdown("### 📁 Fonte de Dados")
    uploaded_file = st.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_real = pd.read_csv(uploaded_file)
            else:
                df_real = pd.read_excel(uploaded_file)
            
            # Normalização de nomes de colunas
            # (Assume que se o ficheiro vier com nomes em inglês, nós traduzimos para português na UI)
            df_real.columns = [c.capitalize() for c in df_real.columns]
            
            # Caso o ficheiro já venha em inglês, renomeia para a UI em português
            if 'Temperature' in df_real.columns: df_real.rename(columns={'Temperature': 'Temperatura'}, inplace=True)
            if 'Pressure' in df_real.columns: df_real.rename(columns={'Pressure': 'Pressao'}, inplace=True)
            if 'Vibration' in df_real.columns: df_real.rename(columns={'Vibration': 'Vibracao'}, inplace=True)

            if 'Temperatura' in df_real.columns and 'Pressao' in df_real.columns and 'Vibracao' in df_real.columns:
                if 'Hora' in df_real.columns:
                    df_real['Hora'] = pd.to_datetime(df_real['Hora'])
                else:
                    df_real['Hora'] = pd.date_range(end=datetime.now(), periods=len(df_real), freq='h')
                
                st.session_state.df = df_real
                st.success("Dados carregados com sucesso!")
            else:
                st.error("O ficheiro deve conter as colunas: Temperatura, Pressao, Vibracao.")
        except Exception as e:
            st.error(f"Erro ao ler ficheiro: {e}")

    if 'df' not in st.session_state:
        st.session_state.df = dados_fallback()

    st.divider()
    st.markdown("""
    <div style="margin-top: 50px; padding: 10px; background: #0c1438; border-radius: 8px; border: 1px solid #42dd8b;">
        <span style="color: #42dd8b; font-size: 20px;">🛡️</span> <b>Sistema Online</b><br>
        <span style="font-size: 12px; color: #93a0c5;">Aguardando dados</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. HEADER
# ==========================================
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #26345f; padding-bottom: 15px; margin-bottom: 20px;">
    <div>
        <h1 class="header-title">Dashboard de Manutenção Preditiva</h1>
        <p class="header-sub">Monitorização em Tempo Real das Bombas Centrífugas</p>
    </div>
    <div style="text-align: right; font-size: 13px; color: #93a0c5;">
        <span style="font-size: 20px; margin-right: 10px;">👤</span> <b>Adécio Alberto</b><br>
        UEM - Engenharia Mecânica
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. PROCESSAMENTO REAL (Pipeline ML)
# ==========================================
df = st.session_state.df
scaler, selector, modelo = carregar_recursos_ml()
prob_falha, risco, cor_risco = prever_falha(df, scaler, selector, modelo)

status_bomba = "Operacional" if prob_falha < 60 else "Crítico"
status_modelo = "Conectado (Real)" if modelo is not None else "Simulado (Fallback)"

# ==========================================
# 7. KPIs (Linha Superior)
# ==========================================
col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])

with col1:
    cor_status = "text-green" if status_bomba == "Operacional" else "text-red"
    icone = "✅" if status_bomba == "Operacional" else "🚨"
    st.markdown(f"""
    <div class="custom-card status-container">
        <div class="status-icon {cor_status}">{icone}</div>
        <div>
            <div style="color: #93a0c5; font-size: 13px;">Estado da Bomba</div>
            <div class="{cor_status}" style="font-size: 24px; font-weight: bold;">{status_bomba}</div>
            <div style="font-size: 12px; color: #f4f6ff;">Probabilidade: {prob_falha:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="custom-card" style="text-align: center;">
        <div style="font-size: 24px; color: #42dd8b;">✅</div>
        <h2 style="margin: 0; color: #f4f6ff;">{len(df[df['Temperatura'] < 85])}</h2>
        <div style="font-size: 12px; color: #93a0c5;">Leituras Normais</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="custom-card" style="text-align: center;">
        <div style="font-size: 24px; color: #ffc95e;">⚠️</div>
        <h2 style="margin: 0; color: #f4f6ff;">{len(df[df['Temperatura'] >= 85])}</h2>
        <div style="font-size: 12px; color: #93a0c5;">Leituras em Alerta</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    fig_risco = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_falha,
        title={'text': "Nível de Risco (ML)", 'font': {'size': 14, 'color': '#93a0c5'}},
        number={'suffix': "%", 'font': {'color': '#f4f6ff'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#26345f"},
            'bar': {'color': cor_risco}, 'bgcolor': "#0c1438", 'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': "rgba(66, 221, 139, 0.2)"},
                {'range': [25, 60], 'color': "rgba(255, 201, 94, 0.2)"},
                {'range': [60, 100], 'color': "rgba(255, 98, 109, 0.2)"}],
        }
    ))
    fig_risco.update_layout(height=160, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f4f6ff"})
    st.plotly_chart(fig_risco, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 8. GRÁFICOS (Linha do Meio)
# ==========================================
c_temp, c_press, c_vib = st.columns(3)

def plot_time_series(df, col, title, unit, limit, color):
    fig = go.Figure()
    df_plot = df.tail(20)
    
    fig.add_trace(go.Scatter(x=df_plot['Hora'], y=df_plot[col], mode='lines+markers',
                             name=col, line=dict(color=color, width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=df_plot['Hora'], y=[limit]*len(df_plot), mode='lines',
                             name='Limite', line=dict(color='#ff626d', width=2, dash='dash')))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#f4f6ff')),
        paper_bgcolor='#0c1438', plot_bgcolor='#0c1438',
        margin=dict(l=30, r=15, t=40, b=20), height=220,
        xaxis=dict(showgrid=True, gridcolor='#202b52', tickformat='%H:%M'),
        yaxis=dict(title=unit, showgrid=True, gridcolor='#202b52', range=[min(df_plot[col])-5, max(limit, max(df_plot[col]))+5]),
        showlegend=False
    )
    return fig

with c_temp:
    st.plotly_chart(plot_time_series(df, 'Temperatura', 'Temperatura', '°C', 95, '#41d9ff'), use_container_width=True, config={'displayModeBar': False})
with c_press:
    st.plotly_chart(plot_time_series(df, 'Pressao', 'Pressão', 'bar', 5.5, '#42dd8b'), use_container_width=True, config={'displayModeBar': False})
with c_vib:
    st.plotly_chart(plot_time_series(df, 'Vibracao', 'Vibração', 'mm/s', 9.0, '#ffc95e'), use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 9. TABELA DE ALERTAS E RECOMENDAÇÃO (Linha Inferior)
# ==========================================
c_alerts, c_recomend = st.columns([2, 1])

with c_alerts:
    rows_html = ""
    for _, row in df.tail(3).iloc[::-1].iterrows():
        status = "Normal" if row['Temperatura'] < 85 else "Crítico"
        badge = "badge-normal" if status == "Normal" else "badge-critico"
        msg = "Condição estável" if status == "Normal" else "Temperatura elevada"
        hora_str = row['Hora'].strftime('%d/%m/%Y %H:%M') if pd.notnull(row['Hora']) else '-'
        
        rows_html += f"""
        <tr>
            <td>{hora_str}</td>
            <td>PUMP 33302</td>
            <td><span class="badge {badge}">{status}</span></td>
            <td>{msg}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="custom-card" style="padding: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h3 style="margin: 0; font-size: 16px;">🔔 Histórico de Alertas (Últimos 3)</h3>
        </div>
        <table class="alert-table">
            <tr>
                <th>Data e Hora</th>
                <th>Equipamento</th>
                <th>Estado</th>
                <th>Mensagem</th>
            </tr>
            {rows_html}
        </table>
    </div>
    """, unsafe_allow_html=True)

with c_recomend:
    st.markdown(f"""
    <div class="custom-card">
        <h3 style="margin: 0 0 10px 0; font-size: 16px;">💡 Probabilidade de Falha</h3>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 32px; font-weight: bold; color: {cor_risco};">{prob_falha:.1f}%</div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: #93a0c5;">Recomendação</div>
                <div style="color: {cor_risco}; font-weight: bold; font-size: 14px;">
                    {'✅ Manter monitorização' if prob_falha < 60 else '🚨 Intervenção imediata'}
                </div>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 11px; color: #93a0c5; border-top: 1px solid #26345f; padding-top: 10px;">
            Modelo: Random Forest<br>
            Status ML: {status_modelo}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    pdf_bytes = gerar_pdf(df, prob_falha, risco)
    st.download_button(
        label="📄 Descarregar Relatório em PDF",
        data=pdf_bytes,
        file_name=f"relatorio_adecio_alberto_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )