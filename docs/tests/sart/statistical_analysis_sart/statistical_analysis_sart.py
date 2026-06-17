import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import glob
import shutil
from datetime import datetime
from scipy.stats import exponnorm

PROGRAM_VERSION = "SART-OSWeb v2.3.1"

# ==========================================
# 1. CONFIGURAÇÃO E LOOP AUTOMÁTICO
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
arquivos_json = glob.glob(os.path.join(base_dir, "*.json"))

if not arquivos_json:
    print(f"ERRO: Nenhum arquivo .json encontrado em: {base_dir}")
    sys.exit()

for input_file in arquivos_json:
    plt.close('all') 
    print(f"\n>>> Lendo arquivo bruto: {os.path.basename(input_file)}")
    
    df = pd.read_json(input_file)
    
    # --- PROCURA O NOME DO PARTICIPANTE ---
    nome_bruto = None
    if 'subject_name' in df.columns and not df['subject_name'].dropna().empty:
        nome_bruto = str(df['subject_name'].dropna().iloc[0])
    elif 'vars' in df.columns and not df['vars'].dropna().empty:
        primeira_linha_vars = df['vars'].dropna().iloc[0]
        if isinstance(primeira_linha_vars, dict) and 'subject_name' in primeira_linha_vars:
            nome_bruto = str(primeira_linha_vars['subject_name'])
            
    if not nome_bruto:
        for col in df.columns:
            if df[col].astype(str).str.contains('subject_name').any():
                linha = df[df[col].astype(str).str.contains('subject_name')].iloc[0]
                if 'subject_name' in str(linha):
                    nome_bruto = "Identificado_No_Log"
                    
    if not nome_bruto:
        nome_bruto = "Participante_Anonimo"
        
    nome_sujeito = "".join([c for c in nome_bruto if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
    if not nome_sujeito: nome_sujeito = "Participante_Anonimo"

    data_atual = datetime.now().strftime("%d-%m-%Y")
    nome_base_pasta = f"{nome_sujeito}_{data_atual}"
    
    output_dir = os.path.join(base_dir, nome_base_pasta)
    nome_final_paciente = nome_base_pasta
    
    if os.path.exists(output_dir):
        contador = 1
        while os.path.exists(os.path.join(base_dir, f"{nome_base_pasta}_{contador}")):
            contador += 1
        nome_final_paciente = f"{nome_base_pasta}_{contador}"
        output_dir = os.path.join(base_dir, nome_final_paciente)

    os.makedirs(output_dir, exist_ok=True)
    print(f"    Paciente identificado: {nome_sujeito}")

    # ==========================================
    # 2. LIMPEZA E SEPARAÇÃO DE FASES
    # ==========================================
    if 'count_Trial_Loop_Task_Phase' in df.columns:
        df_task = df.dropna(subset=['count_Trial_Loop_Task_Phase']).copy()
    else:
        df_task = df.copy()

    for col in ['response_time', 'correct', 'is_artifact']:
        if col in df_task.columns:
            df_task[col] = pd.to_numeric(df_task[col], errors='coerce')

    # ==========================================
    # 3. CLASSIFICAÇÃO CLÍNICA (NO DADO BRUTO)
    # ==========================================
    def categorizar_trial(row):
        if row.get('is_artifact') == 1: return 'artefato'
        if row['condition'] == 'go': return 'go_correto' if row['correct'] == 1 else 'omissao'
        elif row['condition'] == 'nogo': return 'nogo_correto' if row['correct'] == 1 else 'comissao'
        return 'outro'

    df_task['trial_type'] = df_task.apply(categorizar_trial, axis=1)
    
    # CRÍTICO: Mapeia o passado histórico VERDADEIRO antes de remover qualquer artefato
    df_task['prev_trial_type'] = df_task['trial_type'].shift(1)

    # ==========================================
    # 4. FILTRAGEM METODOLÓGICA E MÉTRICAS
    # ==========================================
    # Isola apenas os dados válidos para estatística
    validos = df_task[df_task['trial_type'] != 'artefato'].copy()
    
    n_total_executado = len(df_task)
    artefatos = len(df_task[df_task['trial_type'] == 'artefato'])
    trials_validos_total = len(validos)
    
    # Denominadores estritamente válidos
    n_nogo_validos = len(validos[validos['condition'] == 'nogo'])
    n_go_validos = len(validos[validos['condition'] == 'go'])

    c = len(validos[validos['trial_type'] == 'comissao'])
    o = len(validos[validos['trial_type'] == 'omissao'])
    
    # RT Geral (Apenas acertos Go limpos)
    rt_corretos = validos[validos['trial_type'] == 'go_correto']['response_time'].dropna()
    mean_rt = rt_corretos.mean() if not rt_corretos.empty else 0
    min_rt = rt_corretos.min() if not rt_corretos.empty else 0
    max_rt = rt_corretos.max() if not rt_corretos.empty else 0
    sd_rt = rt_corretos.std() if not rt_corretos.empty else 0
    cv_rt = (sd_rt / mean_rt * 100) if mean_rt > 0 else 0

    # --- MODELAGEM EX-GAUSSIANA (TAU) ---
    tau = 0
    mu = 0
    sigma_ex = 0
    if len(rt_corretos) > 20:
        K, loc, scale = exponnorm.fit(rt_corretos)
        mu = loc
        sigma_ex = scale
        tau = K * scale

    # RT Contextual (Efeitos Sequenciais Puros)
    corretos_go_df = validos[validos['trial_type'] == 'go_correto'].copy()
    
    # Filtra RTs cujo passado verdadeiro (sem recortes) foi limpo
    rt_cruzeiro = corretos_go_df[corretos_go_df['prev_trial_type'] == 'go_correto']['response_time']
    rt_recuperacao = corretos_go_df[corretos_go_df['prev_trial_type'] == 'nogo_correto']['response_time']
    
    n_cruzeiro = len(rt_cruzeiro)
    n_recuperacao = len(rt_recuperacao)
    mean_cruzeiro = rt_cruzeiro.mean() if n_cruzeiro > 0 else np.nan
    mean_recuperacao = rt_recuperacao.mean() if n_recuperacao > 0 else np.nan
    pis_estimavel = n_cruzeiro > 0 and n_recuperacao > 0
    custo_inibicao = mean_recuperacao - mean_cruzeiro if pis_estimavel else np.nan

    def fmt_ms(value, signed=False, digits=2):
        if pd.isna(value):
            return "N/A"
        sign = "+" if signed else ""
        return f"{value:{sign}.{digits}f} ms"

    # Dinâmica de Fadiga (Gaussiana sobre válidos)
    validos['is_error'] = validos['trial_type'].apply(lambda x: 1 if x in ['comissao', 'omissao'] else 0)
    window_size = 30 if len(validos) > 60 else max(5, len(validos)//5)
    rolling_error = validos['is_error'].rolling(window=window_size, win_type='gaussian', center=True).mean(std=window_size/4).fillna(0)
    gradient = np.gradient(rolling_error)
    
    inflection_idx = np.argmax(gradient) if np.max(gradient) > 0.001 else len(validos) // 2
    pre_fadiga = validos.iloc[:inflection_idx]
    pos_fadiga = validos.iloc[inflection_idx:]
    
    taxa_pre = pre_fadiga['is_error'].mean() * 100 if len(pre_fadiga) > 0 else 0
    taxa_pos = pos_fadiga['is_error'].mean() * 100 if len(pos_fadiga) > 0 else 0
    delta_fadiga = taxa_pos - taxa_pre

    # ==========================================
    # 5. GRÁFICOS
    # ==========================================
    sns.set_theme(style="ticks", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'

    # Fig 1 - Comportamento
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    taxas = {'Comissão\n(Falha Inibitória)': {'pct': (c/n_nogo_validos*100 if n_nogo_validos else 0), 'count': f"{c}/{n_nogo_validos}", 'color': '#d62728'},
             'Omissão\n(Lapso Atencional)': {'pct': (o/n_go_validos*100 if n_go_validos else 0), 'count': f"{o}/{n_go_validos}", 'color': '#ff7f0e'},
             'Antecipatória\n(RT <100 ms)': {'pct': (artefatos/n_total_executado*100 if n_total_executado else 0), 'count': f"{artefatos}/{n_total_executado}", 'color': '#7f7f7f'}}
    bars = ax1.bar(list(taxas.keys()), [d['pct'] for d in taxas.values()], color=[d['color'] for d in taxas.values()], edgecolor='black', linewidth=1.5, width=0.6)
    for bar, key in zip(bars, taxas.keys()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%\n(n={taxas[key]['count']})", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    pct_desc = (artefatos / n_total_executado * 100) if n_total_executado > 0 else 0
    pct_val = (trials_validos_total / n_total_executado * 100) if n_total_executado > 0 else 0
    filtro_box = f"Total Executado: {n_total_executado}\nDescartados (Artefatos): {artefatos} ({pct_desc:.1f}%)\nVálidos Pós-Filtro: {trials_validos_total} ({pct_val:.1f}%)"
    ax1.text(0.95, 0.95, filtro_box, transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))
    
    ax1.set_ylim(0, max([d['pct'] for d in taxas.values()]) + 20 if max([d['pct'] for d in taxas.values()]) > 0 else 100)
    ax1.set_ylabel('Taxa de Ocorrência (%)', fontweight='bold')
    ax1.set_title(f'SART - {nome_sujeito}', fontweight='bold', pad=20)
    sns.despine(); plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'SART_Fig1_Comportamento.png'), dpi=300); plt.close(fig1)

    # Fig 2 - RT Ex-Gaussiano
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if not rt_corretos.empty:
        sns.histplot(rt_corretos, bins=15, kde=True, color='#1f77b4', edgecolor='black', alpha=0.6, ax=ax2)
        ax2.axvline(mean_rt, color='red', linestyle='--', linewidth=2.5, label='Média')
        ax2.axvline(mean_rt + sd_rt, color='gray', linestyle=':', linewidth=2, label='+1 SD')
        ax2.axvline(mean_rt - sd_rt, color='gray', linestyle=':', linewidth=2, label='-1 SD')
        
        stats_box = f"Mínimo: {min_rt:.1f} ms\nMáximo: {max_rt:.1f} ms\nMédia (μ total): {mean_rt:.1f} ms\nVariabilidade (σ): {sd_rt:.1f} ms\nCV: {cv_rt:.1f}%\n---\nTau (τ): {tau:.1f} ms\nMu (μ proc.): {mu:.1f} ms\nSigma (σ proc.): {sigma_ex:.1f} ms\nN (Acertos): {len(rt_corretos)}"
        ax2.text(0.95, 0.95, stats_box, transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
        ax2.set_xlabel('Tempo de Reação (ms)', fontweight='bold')
        ax2.set_title(f'Distribuição RT e Ex-Gaussiana - {nome_sujeito}', fontweight='bold', pad=20)
        ax2.legend(loc='upper left')
        sns.despine(); plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, 'SART_Fig2_TempoReacao.png'), dpi=300); plt.close(fig2)

    # Fig 3 - Fadiga
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    eixo_x_validos = np.arange(1, len(validos) + 1)
    sns.lineplot(x=eixo_x_validos, y=rolling_error * 100, ax=ax3, color='indigo', linewidth=2.5, label='Erro (Gaussiana)')
    ax3.axvline(x=inflection_idx + 1, color='crimson', linestyle='--', linewidth=2, label=f'Inflexão (Trial {inflection_idx + 1})')
    ax3.set_title(f'Dinâmica de Fadiga Cognitiva - {nome_sujeito}', fontweight='bold', pad=20)
    ax3.set_xlabel('Progresso da Tarefa (Trials Válidos)', fontweight='bold')
    ax3.set_ylabel('Taxa de Erro Estimada (%)', fontweight='bold')
    ax3.legend(loc='upper left')
    
    delta_box = f"Delta Pós-Fadiga: {delta_fadiga:+.1f}%\n(Pré: {taxa_pre:.1f}% | Pós: {taxa_pos:.1f}%)"
    ax3.text(0.95, 0.95, delta_box, transform=ax3.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))
    sns.despine(); plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'SART_Fig3_FadigaDinamica.png'), dpi=300); plt.close(fig3)

    # Fig 4 - Custo de Inibição (Post-Inhibitory Slowing)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    plot_df = corretos_go_df.dropna(subset=['prev_trial_type']).copy()
    plot_df = plot_df[plot_df['prev_trial_type'].isin(['go_correto', 'nogo_correto'])]
    plot_df['Contexto'] = plot_df['prev_trial_type'].map({'go_correto': 'Go pós-Go correto\n(Baseline sequencial)', 'nogo_correto': 'Go pós-No-Go correto\n(Pós-inibição)'})
    context_order = ['Go pós-Go correto\n(Baseline sequencial)', 'Go pós-No-Go correto\n(Pós-inibição)']
    context_palette = {context_order[0]: '#4C72B0', context_order[1]: '#C44E52'}
    
    if not plot_df.empty:
        sns.boxplot(x='Contexto', y='response_time', data=plot_df, hue='Contexto', order=context_order, hue_order=context_order, width=0.4, palette=context_palette, ax=ax4, showfliers=False, boxprops=dict(alpha=0.7), legend=False)
        sns.stripplot(x='Contexto', y='response_time', data=plot_df, order=context_order, color='black', alpha=0.6, jitter=True, size=6, ax=ax4)
        
        ax4.set_title(f'Lentificação Pós-Inibitória (PIS) - {nome_sujeito}', fontweight='bold', pad=20, fontsize=16, wrap=True)
        ax4.set_ylabel('Tempo de Reação (ms)', fontweight='bold')
        ax4.set_xlabel('')
        ax4.set_xlim(-0.5, 1.5)
        
        y_min, y_max = ax4.get_ylim()
        y_mid = y_min + (y_max - y_min) * 0.5
        if n_cruzeiro == 0:
            ax4.text(0, y_mid, 'Sem dados\n(n=0)', ha='center', va='center', fontsize=11, color='gray')
        if n_recuperacao == 0:
            ax4.text(1, y_mid, 'Sem dados\n(n=0)', ha='center', va='center', fontsize=11, color='gray')

        if pis_estimavel:
            custo_box = f"Delta PIS: {fmt_ms(custo_inibicao, signed=True, digits=1)}\n(Go pós-Go n={n_cruzeiro}: {fmt_ms(mean_cruzeiro, digits=1)} | Go pós-No-Go n={n_recuperacao}: {fmt_ms(mean_recuperacao, digits=1)})"
        else:
            custo_box = f"Delta PIS: não estimável\n(Go pós-Go n={n_cruzeiro}: {fmt_ms(mean_cruzeiro, digits=1)} | Go pós-No-Go n={n_recuperacao}: {fmt_ms(mean_recuperacao, digits=1)})"
        ax4.text(0.5, 0.95, custo_box, transform=ax4.transAxes, fontsize=11, ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))
        
        sns.despine(); plt.tight_layout()
        fig4.savefig(os.path.join(output_dir, 'SART_Fig4_CustoInibicao.png'), dpi=300)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'PIS não estimável\nSem trials Go corretos com contexto sequencial válido', transform=ax4.transAxes, ha='center', va='center', fontsize=13, bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))
        fig4.savefig(os.path.join(output_dir, 'SART_Fig4_CustoInibicao.png'), dpi=300)
    plt.close(fig4)

    # ==========================================
    # 6. EXPORTAÇÃO TXT
    # ==========================================
    log_path = os.path.join(output_dir, 'Relatorio_SART.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== RELATÓRIO CLÍNICO SART: {nome_sujeito} ===\n")
        f.write(f"Versão do Programa: {PROGRAM_VERSION}\n")
        f.write(f"Data de Processamento: {data_atual}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total de Trials Executados: {n_total_executado}\n")
        f.write(f"  (-) Trials Descartados (Artefatos): {artefatos} ({(artefatos/n_total_executado*100 if n_total_executado > 0 else 0):.2f}%)\n")
        f.write(f"  (=) Trials Válidos Pós-Filtro:       {trials_validos_total} ({(trials_validos_total/n_total_executado*100 if n_total_executado > 0 else 0):.2f}%)\n")
        f.write(f"  * Nota Metodológica: Respostas <100 ms foram classificadas como respostas antecipatórias/artefatos temporais e isoladas das métricas principais de RT e acurácia.\n\n")
        
        f.write("MÉTRICAS COMPORTAMENTAIS SART (SOBRE VÁLIDOS):\n")
        f.write(f"  Erros de Comissão (falha de inibição de resposta):      {c} / {n_nogo_validos} ({(c/n_nogo_validos*100 if n_nogo_validos > 0 else 0):.2f}%)\n")
        f.write(f"  Erros de Omissão (lapso de atenção sustentada):         {o} / {n_go_validos} ({(o/n_go_validos*100 if n_go_validos > 0 else 0):.2f}%)\n")
        f.write(f"  Respostas Antecipatórias (<100 ms; artefato temporal):  {artefatos} / {n_total_executado} ({(artefatos/n_total_executado*100 if n_total_executado > 0 else 0):.2f}%)\n\n")
        
        f.write("MÉTRICAS DE TEMPO DE REAÇÃO (GO CORRETOS PÓS-FILTRO):\n")
        f.write(f"  Tempo de Reação Mínimo (Min):     {min_rt:.2f} ms\n" if not rt_corretos.empty else "  Tempo de Reação Mínimo (Min): N/A\n")
        f.write(f"  Tempo de Reação Máximo (Max):     {max_rt:.2f} ms\n" if not rt_corretos.empty else "  Tempo de Reação Máximo (Max): N/A\n")
        f.write(f"  Tempo de Reação Médio (μ total):  {mean_rt:.2f} ms\n")
        f.write(f"  Variabilidade do RT (σ):          {sd_rt:.2f} ms\n")
        f.write(f"  Coeficiente de Variação (CV):     {cv_rt:.2f} %\n\n")

        f.write("MODELAGEM EX-GAUSSIANA DA DISTRIBUIÇÃO DE RT:\n")
        f.write(f"  Mu (μ - componente gaussiano/location):       {mu:.2f} ms\n")
        f.write(f"  Sigma (σ - variabilidade gaussiana/scale):    {sigma_ex:.2f} ms\n")
        f.write(f"  Tau (τ - cauda exponencial/IIV lenta):        {tau:.2f} ms\n")
        f.write(f"  * Nota Metodológica: Na parametrização exponnorm do SciPy, μ=loc, σ=scale e τ=K*scale. Tau indexa a cauda lenta da distribuição de RT, frequentemente associada à variabilidade intraindividual e a lapsos atencionais.\n\n")

        f.write("LENTIFICAÇÃO PÓS-INIBITÓRIA (PIS - EFEITOS SEQUENCIAIS):\n")
        f.write(f"  RT Go pós-Go correto (baseline sequencial):   {fmt_ms(mean_cruzeiro)} (n={n_cruzeiro})\n")
        f.write(f"  RT Go pós-No-Go correto (pós-inibição):       {fmt_ms(mean_recuperacao)} (n={n_recuperacao})\n")
        f.write(f"  Delta PIS (pós-inibição - baseline):          {fmt_ms(custo_inibicao, signed=True)}\n")
        if pis_estimavel:
            f.write(f"  * Nota Metodológica: O Delta PIS estima a lentificação do RT após uma inibição bem-sucedida, comparada ao baseline sequencial pós-Go.\n\n")
        else:
            f.write(f"  * Nota Metodológica: PIS não estimável porque falta pelo menos um trial Go correto após No-Go correto; isso ocorre quando não há inibições No-Go bem-sucedidas suficientes para formar o contexto pós-inibição.\n\n")

        f.write("DINÂMICA DE FADIGA (TIME-ON-TASK):\n")
        f.write(f"  Ponto de Inflexão Detectado:      Trial Válido {inflection_idx + 1}\n")
        f.write(f"  Taxa de Erro Pré-Fadiga:          {taxa_pre:.2f}%\n")
        f.write(f"  Taxa de Erro Pós-Fadiga:          {taxa_pos:.2f}%\n")
        f.write(f"  Delta de Piora Clínica:           {delta_fadiga:+.2f}%\n")
        f.write(f"  * Nota Metodológica: O ponto de inflexão de fadiga é calculado via suavização Gaussiana móvel, evitando o viés de divisão artificial da amostra.\n")
        f.write("-" * 60 + "\n")

    # ==========================================
    # 7. ORGANIZAÇÃO FINAL
    # ==========================================
    novo_nome_json = f"SART_DadosBrutos_{nome_final_paciente}.json"
    caminho_json_final = os.path.join(output_dir, novo_nome_json)
    
    shutil.move(input_file, caminho_json_final)
    print(f"    [OK] Processamento estatístico finalizado.")

print("\n[SUCESSO] Processamento completo executado com sucesso.")
