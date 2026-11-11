with c_alerts:
    # Pega sempre os 3 registos mais recentes do histórico geral
    ultimos_registos = df.tail(3).iloc[::-1]

    rows_html = ""
    for _, row in ultimos_registos.iterrows():
        # Lógica de classificação dinâmica para cada linha
        if row['Temperatura'] >= 95:
            status = "Crítico"
            badge = "badge-critico"
            msg = "Temperatura crítica detectada"
        elif row['Vibracao'] >= 9.0:
            status = "Crítico"
            badge = "badge-critico"
            msg = "Vibração excessiva detectada"
        elif row['Pressao'] >= 5.5:
            status = "Alerta"
            badge = "badge-alerta"
            msg = "Pressão acima do normal"
        else:
            status = "Normal"
            badge = "badge-normal"
            msg = "Condição estável"
            
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
            <h3 style="margin: 0; font-size: 16px;">🔔 Histórico de Alertas (Mais Recentes)</h3>
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