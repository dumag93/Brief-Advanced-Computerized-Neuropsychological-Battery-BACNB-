import glob
import os
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROGRAM_NAME = "Flanker-OSWeb v1"
ANTICIPATION_CUTOFF_MS = 100


def safe_num(series):
    return pd.to_numeric(series, errors="coerce")


def identify_subject(df):
    nome = None
    if "subject_name" in df.columns and not df["subject_name"].dropna().empty:
        nome = str(df["subject_name"].dropna().iloc[0])
    elif "vars" in df.columns and not df["vars"].dropna().empty:
        first_vars = df["vars"].dropna().iloc[0]
        if isinstance(first_vars, dict) and "subject_name" in first_vars:
            nome = str(first_vars["subject_name"])
    if not nome:
        nome = "Participante_Anonimo"
    nome = "".join(c for c in nome if c.isalnum() or c in (" ", "_", "-"))
    return nome.strip().replace(" ", "_") or "Participante_Anonimo"


def format_ms(value, signed=False, digits=2):
    if pd.isna(value):
        return "N/A"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f} ms"


def format_pct(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def prepare_task_dataframe(df):
    if "task_name" in df.columns:
        task = df[df["task_name"].astype(str).str.lower() == "flanker"].copy()
    else:
        task = df.copy()
    if "phase" in task.columns and task["phase"].astype(str).str.contains("official", case=False, na=False).any():
        task = task[task["phase"].astype(str).str.lower() == "official"].copy()
    elif "count_official_loop" in task.columns and task["count_official_loop"].notna().any():
        task = task[task["count_official_loop"].notna()].copy()
    elif "count_trial_sequence" in task.columns and task["count_trial_sequence"].notna().any():
        task = task[task["count_trial_sequence"].notna()].copy()
    for col in ["response_time", "correct", "error", "omission", "anticipation", "trial_index"]:
        if col in task.columns:
            task[col] = safe_num(task[col])
    for col in ["condition", "target_direction", "stimulus_text", "correct_response"]:
        if col not in task.columns:
            task[col] = np.nan
    return task


def process_file(input_file, base_dir):
    print(f"\n>>> Lendo arquivo bruto: {os.path.basename(input_file)}")
    df = pd.read_json(input_file)
    nome_sujeito = identify_subject(df)
    print(f"    Participante identificado: {nome_sujeito}")
    data_atual = datetime.now().strftime("%d-%m-%Y")
    output_dir = os.path.join(base_dir, f"{nome_sujeito}_{data_atual}")
    if os.path.exists(output_dir):
        counter = 1
        while os.path.exists(f"{output_dir}_{counter}"):
            counter += 1
        output_dir = f"{output_dir}_{counter}"
    os.makedirs(output_dir, exist_ok=True)

    task = prepare_task_dataframe(df)
    valid_rt = task[(task["correct"] == 1) & (task["response_time"] >= ANTICIPATION_CUTOFF_MS)].copy()
    conditions = ["congruente", "neutra", "incongruente"]

    summary = []
    for cond in conditions:
        sub = task[task["condition"] == cond]
        rt = valid_rt[valid_rt["condition"] == cond]["response_time"].dropna()
        summary.append({
            "Condicao": cond,
            "N": len(sub),
            "Acertos": int((sub["correct"] == 1).sum()),
            "Erros": int((sub["error"] == 1).sum()),
            "Omissoes": int((sub["omission"] == 1).sum()),
            "Antecipacoes": int((sub["anticipation"] == 1).sum()),
            "Acuracia": (sub["correct"] == 1).mean() if len(sub) else np.nan,
            "Erro": (sub["error"] == 1).mean() if len(sub) else np.nan,
            "RT_medio": rt.mean() if not rt.empty else np.nan,
            "RT_sd": rt.std() if len(rt) > 1 else np.nan,
        })
    s = pd.DataFrame(summary)
    rt_cong = s.loc[s["Condicao"] == "congruente", "RT_medio"].iloc[0]
    rt_neu = s.loc[s["Condicao"] == "neutra", "RT_medio"].iloc[0]
    rt_inc = s.loc[s["Condicao"] == "incongruente", "RT_medio"].iloc[0]
    flanker_effect = rt_inc - rt_cong if pd.notna(rt_inc) and pd.notna(rt_cong) else np.nan
    interference = rt_inc - rt_neu if pd.notna(rt_inc) and pd.notna(rt_neu) else np.nan
    facilitation = rt_neu - rt_cong if pd.notna(rt_neu) and pd.notna(rt_cong) else np.nan

    sns.set_theme(style="ticks", context="talk")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = [
        ("Erro\nCongruente", s.loc[s["Condicao"] == "congruente", "Erro"].iloc[0] * 100, "#4C72B0"),
        ("Erro\nNeutro", s.loc[s["Condicao"] == "neutra", "Erro"].iloc[0] * 100, "#8E63B8"),
        ("Erro\nIncongruente", s.loc[s["Condicao"] == "incongruente", "Erro"].iloc[0] * 100, "#D62728"),
        ("Omissao\nTotal", (task["omission"] == 1).mean() * 100 if len(task) else 0, "#7F7F7F"),
        ("Antecipacao\nTotal", (task["anticipation"] == 1).mean() * 100 if len(task) else 0, "#F28E2B"),
    ]
    x = np.arange(len(bars))
    ax1.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="black", linewidth=1.5, width=0.58)
    for i, b in enumerate(bars):
        ax1.text(i, b[1] + 1.5, f"{b[1]:.1f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax1.set_ylim(0, max(30, max([b[1] for b in bars]) + 10))
    ax1.set_ylabel("Taxa de Ocorrencia (%)")
    ax1.set_title(f"Flanker - {nome_sujeito}", pad=20)
    box = f"Trials oficiais: {len(task)}\nCong: {(task['condition']=='congruente').sum()} | Neu: {(task['condition']=='neutra').sum()} | Inc: {(task['condition']=='incongruente').sum()}\nEfeito Flanker: {format_ms(flanker_effect, signed=True, digits=1)}"
    ax1.text(0.98, 0.95, box, transform=ax1.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="gray", alpha=0.9))
    sns.despine()
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "Flanker_Fig1_Comportamento.png"), dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_df = valid_rt.copy()
    if not plot_df.empty:
        sns.boxplot(x="condition", y="response_time", data=plot_df, order=conditions, color="#9ECAE1",
                    width=0.45, showfliers=False, ax=ax2)
        sns.stripplot(x="condition", y="response_time", data=plot_df, order=conditions, color="black",
                      alpha=0.45, jitter=0.18, size=3.5, ax=ax2)
    ax2.set_title(f"Tempo de Reacao por Condicao - {nome_sujeito}", pad=20)
    ax2.set_xlabel("")
    ax2.set_ylabel("Tempo de Reacao (ms)")
    sns.despine()
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "Flanker_Fig2_TempoReacao.png"), dpi=300)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    indices = [("Efeito Flanker\nInc-Cong", flanker_effect), ("Interferencia\nInc-Neutro", interference), ("Facilitacao\nNeutro-Cong", facilitation)]
    ax3.bar(np.arange(3), [v for _, v in indices], color=["#D62728", "#C44E52", "#4C72B0"], edgecolor="black", linewidth=1.5, width=0.58)
    ax3.axhline(0, color="black", linewidth=1)
    for i, (_, v) in enumerate(indices):
        ax3.text(i, v + (5 if pd.notna(v) and v >= 0 else -8), format_ms(v, signed=True, digits=1),
                 ha="center", va="bottom" if pd.notna(v) and v >= 0 else "top", fontsize=10.5, fontweight="bold")
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels([k for k, _ in indices], fontsize=10)
    ax3.set_ylabel("Diferenca de RT (ms)")
    ax3.set_title(f"Indices de Interferencia Flanker - {nome_sujeito}", pad=20)
    sns.despine()
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "Flanker_Fig3_Interferencia.png"), dpi=300)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    dyn = task.sort_values("trial_index").copy()
    dyn["erro_rolling"] = dyn["error"].fillna(0).rolling(15, min_periods=5).mean() * 100
    ax4.plot(dyn["trial_index"], dyn["erro_rolling"], color="#4B0082", linewidth=2.4)
    ax4.set_title(f"Dinamica de Erro - {nome_sujeito}", pad=20)
    ax4.set_xlabel("Progresso da Tarefa (Trials Oficiais)")
    ax4.set_ylabel("Erro Estimado (%)")
    sns.despine()
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, "Flanker_Fig4_Dinamica.png"), dpi=300)
    plt.close(fig4)

    with open(os.path.join(output_dir, "Relatorio_Flanker.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== RELATORIO FLANKER: {nome_sujeito} ===\n")
        f.write(f"Programa: {PROGRAM_NAME}\n")
        f.write(f"Data de Processamento: {data_atual}\n")
        f.write("-" * 68 + "\n")
        f.write(f"Trials oficiais analisados: {len(task)}\n")
        f.write(f"Corte de antecipacao: {ANTICIPATION_CUTOFF_MS} ms\n\n")
        f.write("DESEMPENHO POR CONDICAO\n")
        for _, row in s.iterrows():
            f.write(f"  {row['Condicao']}: n={int(row['N'])}, acuracia={row['Acuracia']*100:.2f}%, erro={row['Erro']*100:.2f}%, RT={format_ms(row['RT_medio'])}, SD={format_ms(row['RT_sd'])}\n")
        f.write("\nINDICES\n")
        f.write(f"  Efeito Flanker (Incongruente - Congruente): {format_ms(flanker_effect, signed=True)}\n")
        f.write(f"  Interferencia (Incongruente - Neutro):    {format_ms(interference, signed=True)}\n")
        f.write(f"  Facilitacao (Neutro - Congruente):        {format_ms(facilitation, signed=True)}\n")
        f.write("\nNOTAS METODOLOGICAS\n")
        f.write("  - RTs principais usam apenas respostas corretas nao antecipatorias.\n")
        f.write("  - O teste mede conflito de resposta, atencao seletiva e controle executivo sob interferencia, nao inibicao pura.\n")
    shutil.move(input_file, os.path.join(output_dir, f"Flanker_DadosBrutos_{os.path.basename(output_dir)}.json"))
    print("    [OK] Processamento Flanker finalizado.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    arquivos_json = glob.glob(os.path.join(base_dir, "*.json"))
    if not arquivos_json:
        print(f"ERRO: Nenhum arquivo .json encontrado em: {base_dir}")
        sys.exit()
    for input_file in arquivos_json:
        plt.close("all")
        process_file(input_file, base_dir)
    print("\n[SUCESSO] Processamento completo executado com sucesso.")


if __name__ == "__main__":
    main()
