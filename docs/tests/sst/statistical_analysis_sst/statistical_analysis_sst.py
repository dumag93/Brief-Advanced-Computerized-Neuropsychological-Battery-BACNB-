import glob
import os
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROGRAM_NAME = "SST-OSWeb v1"
MAX_RT_DEFAULT = 1000
P_RESPOND_ACCEPTABLE_RANGE = (0.25, 0.75)
GO_OMISSION_WARNING = 0.10
GO_OMISSION_INVALID = 0.25
GO_CHOICE_ERROR_WARNING = 0.10


def safe_num(series):
    return pd.to_numeric(series, errors="coerce")


def first_nonempty(df, columns, default=None):
    for col in columns:
        if col in df.columns:
            values = df[col].dropna()
            if not values.empty:
                return values.iloc[0]
    return default


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
    nome = nome.strip().replace(" ", "_")
    return nome or "Participante_Anonimo"


def format_ms(value, signed=False, digits=2):
    if pd.isna(value):
        return "N/A"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f} ms"


def format_pct(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def collect_quality_flags(p_respond_signal, go_omission_rate, go_choice_error_rate, race_ok, ssrt):
    flags = []

    lo, hi = P_RESPOND_ACCEPTABLE_RANGE
    if pd.isna(p_respond_signal) or p_respond_signal <= 0 or p_respond_signal >= 1:
        flags.append("SSRT nao estimavel: p(respond|signal) extremo ou ausente.")
    elif p_respond_signal < lo or p_respond_signal > hi:
        flags.append("SSRT instavel: p(respond|signal) fora da faixa operacional 0.25-0.75.")

    if go_omission_rate >= GO_OMISSION_INVALID:
        flags.append("SSRT nao recomendado: taxa de omissao Go alta (>=25%).")
    elif go_omission_rate >= GO_OMISSION_WARNING:
        flags.append("Aviso: taxa de omissao Go acima de 10%.")

    if go_choice_error_rate >= GO_CHOICE_ERROR_WARNING:
        flags.append("Aviso: taxa de erro de escolha Go acima de 10%.")

    if not race_ok:
        flags.append("Premissa do horse-race possivelmente violada: RT Stop falho >= RT Go.")

    if pd.isna(ssrt) or ssrt <= 0:
        flags.append("SSRT nao estimavel: valor ausente ou fisiologicamente invalido.")

    return flags


def integration_ssrt(task_df, max_rt):
    go_df = task_df[task_df["trial_type"] == "go"].copy()
    stop_df = task_df[task_df["trial_type"] == "stop"].copy()

    if go_df.empty or stop_df.empty:
        return np.nan, np.nan, np.nan, np.nan, np.array([])

    p_respond_signal = stop_df["stop_responded_num"].mean()
    mean_ssd = stop_df["ssd_used"].mean()

    go_rts = []
    for _, row in go_df.iterrows():
        rt = row.get("rt_total")
        if pd.notna(rt):
            go_rts.append(float(rt))
        else:
            go_rts.append(float(max_rt))

    go_dist = np.array(sorted(go_rts), dtype=float)
    if len(go_dist) == 0 or pd.isna(p_respond_signal) or p_respond_signal <= 0 or p_respond_signal >= 1:
        return np.nan, mean_ssd, p_respond_signal, np.nan, go_dist

    nth = int(np.ceil(p_respond_signal * len(go_dist)))
    nth = min(max(nth, 1), len(go_dist))
    nth_rt = go_dist[nth - 1]
    ssrt = nth_rt - mean_ssd
    return ssrt, mean_ssd, p_respond_signal, nth_rt, go_dist


def prepare_task_dataframe(df):
    if "trial_type" not in df.columns:
        raise ValueError("Arquivo sem coluna trial_type. Verifique se o OSEXP atual esta sendo usado.")

    task = df[df["trial_type"].isin(["go", "stop"])].copy()

    # A fase oficial e marcada pelo logger/trial_sequence oficial.
    # Nao use response_source para excluir treino: Stop trials inibidos corretamente
    # podem ficar com response_source="None" e contaminar a analise.
    official_markers = ["count_trial_sequence", "count_logger"]
    for marker in official_markers:
        if marker in task.columns and task[marker].notna().any():
            main = task[task[marker].notna()].copy()
            break
    else:
        main = task.copy()
        if "response_source" in main.columns:
            main = main[~main["response_source"].astype(str).str.contains("training", case=False, na=False)].copy()

        if "count_block_trial_loop" in main.columns and main["count_block_trial_loop"].notna().any():
            main = main.dropna(subset=["count_block_trial_loop"]).copy()
        elif "count_block_loop" in main.columns and main["count_block_loop"].notna().any():
            main = main.dropna(subset=["count_block_loop"]).copy()

    numeric_cols = [
        "rt_total", "ssd_used", "go_omission", "go_choice_error", "go_correct",
        "stop_responded", "is_stop_trial", "inhibition_success", "ssd_next",
        "block_nr", "count_block_loop", "count_block_trial_loop",
        "response_time_kb1_response", "response_time_kb2_response",
        "stop_failure_pre_signal", "stop_failure_after_signal", "signal_respond_rt",
    ]
    for col in numeric_cols:
        if col in main.columns:
            main[col] = safe_num(main[col])

    required = ["rt_total", "ssd_used", "go_omission", "go_choice_error", "go_correct", "stop_responded"]
    for col in required:
        if col not in main.columns:
            main[col] = np.nan

    main["stop_responded_num"] = main["stop_responded"].replace("N/A", np.nan)
    main["stop_responded_num"] = safe_num(main["stop_responded_num"])
    main["go_correct_num"] = main["go_correct"].replace("N/A", np.nan)
    main["go_correct_num"] = safe_num(main["go_correct_num"])
    return main


def process_file(input_file, base_dir):
    print(f"\n>>> Lendo arquivo bruto: {os.path.basename(input_file)}")
    df = pd.read_json(input_file)
    nome_sujeito = identify_subject(df)
    print(f"    Participante identificado: {nome_sujeito}")

    data_atual = datetime.now().strftime("%d-%m-%Y")
    nome_base_pasta = f"{nome_sujeito}_{data_atual}"
    output_dir = os.path.join(base_dir, nome_base_pasta)
    nome_final = nome_base_pasta

    if os.path.exists(output_dir):
        counter = 1
        while os.path.exists(os.path.join(base_dir, f"{nome_base_pasta}_{counter}")):
            counter += 1
        nome_final = f"{nome_base_pasta}_{counter}"
        output_dir = os.path.join(base_dir, nome_final)

    os.makedirs(output_dir, exist_ok=True)
    task = prepare_task_dataframe(df)

    max_rt = first_nonempty(task, ["max_rt"], MAX_RT_DEFAULT)
    try:
        max_rt = float(max_rt)
    except Exception:
        max_rt = float(MAX_RT_DEFAULT)

    n_total = len(task)
    go_df = task[task["trial_type"] == "go"].copy()
    stop_df = task[task["trial_type"] == "stop"].copy()
    n_go = len(go_df)
    n_stop = len(stop_df)

    go_responded = go_df["rt_total"].notna().sum()
    go_omissions = int(go_df["go_omission"].fillna(0).sum())
    go_choice_errors = int(go_df["go_choice_error"].fillna(0).sum())
    go_correct = int(go_df["go_correct_num"].fillna(0).sum())

    go_omission_rate = go_omissions / n_go if n_go else np.nan
    go_choice_error_rate = go_choice_errors / go_responded if go_responded else np.nan
    go_accuracy_rate = go_correct / n_go if n_go else np.nan

    go_response_rts = go_df["rt_total"].dropna()
    go_correct_rts = go_df[go_df["go_correct_num"] == 1]["rt_total"].dropna()
    failed_stop_rts = stop_df[stop_df["stop_responded_num"] == 1]["rt_total"].dropna()

    ssrt, mean_ssd, p_respond_signal, nth_rt, go_dist_replaced = integration_ssrt(task, max_rt)
    p_inhibit = 1 - p_respond_signal if pd.notna(p_respond_signal) else np.nan

    mean_go_rt = go_response_rts.mean() if not go_response_rts.empty else np.nan
    mean_go_correct_rt = go_correct_rts.mean() if not go_correct_rts.empty else np.nan
    sd_go_correct_rt = go_correct_rts.std() if len(go_correct_rts) > 1 else np.nan
    mean_failed_stop_rt = failed_stop_rts.mean() if not failed_stop_rts.empty else np.nan
    race_ok = bool(pd.isna(mean_failed_stop_rt) or pd.isna(mean_go_rt) or mean_failed_stop_rt < mean_go_rt)

    quality_flags = collect_quality_flags(
        p_respond_signal, go_omission_rate, go_choice_error_rate, race_ok, ssrt
    )

    sns.set_theme(style="ticks", context="talk")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Figura 1 - Qualidade comportamental
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    stop_responded = int(stop_df["stop_responded_num"].fillna(0).sum()) if n_stop else 0
    stop_inhibited = n_stop - stop_responded if n_stop else 0
    taxas = {
        "Omissão Go\n(lapso)": {
            "pct": go_omission_rate * 100 if n_go else 0,
            "count": f"{go_omissions}/{n_go}",
            "color": "#F28E2B",
        },
        "Erro escolha Go\n(resposta incorreta)": {
            "pct": go_choice_error_rate * 100 if go_responded else 0,
            "count": f"{go_choice_errors}/{go_responded}",
            "color": "#8E63B8",
        },
        "Resposta no Stop\np(respond|signal)": {
            "pct": p_respond_signal * 100 if pd.notna(p_respond_signal) else 0,
            "count": f"{stop_responded}/{n_stop}",
            "color": "#D62728",
        },
        "Inibição\np(inhibit|signal)": {
            "pct": p_inhibit * 100 if pd.notna(p_inhibit) else 0,
            "count": f"{stop_inhibited}/{n_stop}",
            "color": "#2CA02C",
        },
    }
    labels = list(taxas.keys())
    values = [d["pct"] for d in taxas.values()]
    colors = [d["color"] for d in taxas.values()]
    x = np.arange(len(labels))
    ax1.bar(x, values, color=colors, edgecolor="black", linewidth=1.5, width=0.58)
    for i, (label, value) in enumerate(zip(labels, values)):
        label_y = value + 1.8
        if abs(value - 50) < 8:
            label_y = 57
        ax1.text(
            i,
            min(label_y, 103),
            f"{value:.1f}%\n(n={taxas[label]['count']})",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.6),
        )
    ax1.axhline(50, color="gray", linestyle=":", linewidth=1.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylim(0, max(100, max(values) + 15 if values else 100))
    ax1.set_ylabel("Taxa de Ocorrência (%)")
    ax1.set_title(f"SST - {nome_sujeito}", pad=20)
    box = (
        f"Trials oficiais: {n_total}\n"
        f"Go: {n_go} | Stop: {n_stop}\n"
        f"p(respond|signal): {format_pct(p_respond_signal, digits=1)}\n"
        f"SSRT: {format_ms(ssrt, digits=1)}"
    )
    ax1.text(0.98, 0.95, box, transform=ax1.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="gray", alpha=0.9))
    sns.despine()
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "SST_Fig1_QualidadeComportamental.png"), dpi=300)
    plt.close(fig1)

    # Figura 2 - Distribuicao Go RT e SSRT
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if not go_correct_rts.empty:
        sns.histplot(go_correct_rts, bins=14, kde=True, color="#6BAED6", edgecolor="black", alpha=0.75, ax=ax2)
        if pd.notna(mean_go_correct_rt):
            ax2.axvline(mean_go_correct_rt, color="red", linestyle="--", linewidth=2.2, label="Média Go correto")
        if pd.notna(mean_go_correct_rt) and pd.notna(sd_go_correct_rt):
            ax2.axvline(mean_go_correct_rt + sd_go_correct_rt, color="gray", linestyle=":", linewidth=2.0, label="+1 SD")
            ax2.axvline(mean_go_correct_rt - sd_go_correct_rt, color="gray", linestyle=":", linewidth=2.0, label="-1 SD")
        if pd.notna(nth_rt):
            ax2.axvline(nth_rt, color="black", linestyle="-", linewidth=2.2, label="nth RT integração")
        finite_rts = go_correct_rts.dropna()
        if not finite_rts.empty:
            lo = max(0, finite_rts.quantile(0.01) - 55)
            hi_candidates = [finite_rts.quantile(0.99) + 55]
            if pd.notna(nth_rt):
                hi_candidates.append(nth_rt + 55)
            if pd.notna(mean_go_correct_rt):
                hi_candidates.append(mean_go_correct_rt + 55)
                if pd.notna(sd_go_correct_rt):
                    hi_candidates.append(mean_go_correct_rt + sd_go_correct_rt + 55)
                    lo = max(0, min(lo, mean_go_correct_rt - sd_go_correct_rt - 55))
            ax2.set_xlim(lo, max(hi_candidates))
        ax2.legend(loc="upper left", fontsize=10)
    ax2.set_title(f"Distribuição Go RT e SSRT - {nome_sujeito}", pad=20)
    ax2.set_xlabel("Tempo de Reação Go (ms)")
    ax2.set_ylabel("Frequência")
    stats = (
        f"Média Go correto: {format_ms(mean_go_correct_rt, digits=1)}\n"
        f"SD Go correto: {format_ms(sd_go_correct_rt, digits=1)}\n"
        f"SSD médio: {format_ms(mean_ssd, digits=1)}\n"
        f"nth RT: {format_ms(nth_rt, digits=1)}\n"
        f"SSRT integração: {format_ms(ssrt, digits=1)}"
    )
    ax2.text(0.98, 0.95, stats, transform=ax2.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.95))
    sns.despine()
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "SST_Fig2_SSRTIntegracao.png"), dpi=300)
    plt.close(fig2)

    # Figura 3 - Tracking SSD
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    stop_plot = stop_df.reset_index(drop=True).copy()
    if not stop_plot.empty:
        stop_plot["stop_index"] = np.arange(1, len(stop_plot) + 1)
        ax3.plot(stop_plot["stop_index"], stop_plot["ssd_used"], color="#4C72B0", linewidth=2.0, marker="o", markersize=4, alpha=0.95)
        responded = stop_plot[stop_plot["stop_responded_num"] == 1]
        inhibited = stop_plot[stop_plot["stop_responded_num"] == 0]
        ax3.scatter(responded["stop_index"], responded["ssd_used"], color="#D62728", s=42, label="Resposta no Stop", zorder=3)
        ax3.scatter(inhibited["stop_index"], inhibited["ssd_used"], color="#2CA02C", s=42, label="Inibição bem-sucedida", zorder=3)
        ax3.axhline(mean_ssd, color="gray", linestyle=":", linewidth=2, label="SSD médio")
        ax3.legend(loc="upper right", fontsize=10)
    ax3.set_title(f"Tracking do Stop-Signal Delay - {nome_sujeito}", pad=20)
    ax3.set_xlabel("Trials Stop")
    ax3.set_ylabel("SSD usado (ms)")
    sns.despine()
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "SST_Fig3_TrackingSSD.png"), dpi=300)
    plt.close(fig3)

    # Figura 4 - Funcao de inibicao
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    if not stop_df.empty:
        inhib = stop_df.groupby("ssd_used", dropna=True).agg(
            p_respond=("stop_responded_num", "mean"),
            n=("stop_responded_num", "count")
        ).reset_index().sort_values("ssd_used")
        ax4.plot(inhib["ssd_used"], inhib["p_respond"] * 100, marker="o", linewidth=2.4, color="#D62728", label="p(respond|signal)")
        for _, row in inhib.iterrows():
            ax4.text(row["ssd_used"], min(row["p_respond"] * 100 + 3, 103), f"n={int(row['n'])}",
                     ha="center", va="bottom", fontsize=9)
    ax4.axhline(50, color="gray", linestyle=":", linewidth=2, label="alvo tracking ~50%")
    ax4.set_ylim(0, 105)
    ax4.set_title(f"Função de Inibição - {nome_sujeito}", pad=20)
    ax4.set_xlabel("SSD (ms)")
    ax4.set_ylabel("p(respond|signal) (%)")
    ax4.legend(loc="upper left", fontsize=10)
    sns.despine()
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, "SST_Fig4_FuncaoInibicao.png"), dpi=300)
    plt.close(fig4)

    report_path = os.path.join(output_dir, "Relatorio_SST.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== RELATORIO CLINICO SST: {nome_sujeito} ===\n")
        f.write(f"Programa: {PROGRAM_NAME}\n")
        f.write(f"Data de Processamento: {data_atual}\n")
        f.write("-" * 68 + "\n")
        f.write("AMOSTRA E DESENHO\n")
        f.write(f"  Trials oficiais analisados: {n_total}\n")
        f.write(f"  Go trials:                 {n_go}\n")
        f.write(f"  Stop trials:               {n_stop}\n")
        f.write(f"  Proporcao Stop:            {format_pct(n_stop / n_total if n_total else np.nan)}\n")
        f.write(f"  Max RT usado no calculo:   {max_rt:.0f} ms\n\n")

        f.write("DESEMPENHO GO\n")
        f.write(f"  Respostas Go registradas:  {go_responded} / {n_go}\n")
        f.write(f"  Omissao Go:                {go_omissions} / {n_go} ({format_pct(go_omission_rate)})\n")
        f.write(f"  Erro de escolha Go:        {go_choice_errors} / {go_responded} ({format_pct(go_choice_error_rate)})\n")
        f.write(f"  Acuracia Go total:         {go_correct} / {n_go} ({format_pct(go_accuracy_rate)})\n")
        f.write(f"  RT medio Go correto:       {format_ms(mean_go_correct_rt)}\n")
        f.write(f"  SD RT Go correto:          {format_ms(sd_go_correct_rt)}\n\n")

        f.write("DESEMPENHO STOP\n")
        f.write(f"  p(respond|signal):         {format_pct(p_respond_signal)}\n")
        f.write(f"  p(inhibit|signal):         {format_pct(p_inhibit)}\n")
        f.write(f"  SSD medio:                 {format_ms(mean_ssd)}\n")
        f.write(f"  RT medio Stop falho:       {format_ms(mean_failed_stop_rt)}\n")
        f.write(f"  Checagem horse-race:       {'OK' if race_ok else 'ALERTA'}\n\n")

        f.write("SSRT - METODO DE INTEGRACAO COM SUBSTITUICAO DE OMISSOES GO\n")
        f.write(f"  nth RT integrado:          {format_ms(nth_rt)}\n")
        f.write(f"  SSRT:                      {format_ms(ssrt)}\n")
        if quality_flags:
            f.write("  Alertas metodologicos:\n")
            for flag in quality_flags:
                f.write(f"  - {flag}\n")
        f.write("\n")

        f.write("NOTAS METODOLOGICAS\n")
        f.write("  - SSRT foi estimado pelo metodo de integracao recomendado por Verbruggen et al. (2019).\n")
        f.write("  - Todos os Go trials com resposta entram na distribuicao de RT; omissoes Go sao substituidas pelo Max RT.\n")
        f.write("  - Respostas em Stop antes do sinal contam como p(respond|signal) e entram no tracking, conforme recomendacao do consenso.\n")
        f.write("  - O resultado deve ser usado como dado clinico complementar, nao como diagnostico isolado.\n")
        f.write("-" * 68 + "\n")

    novo_nome_json = f"SST_DadosBrutos_{nome_final}.json"
    shutil.move(input_file, os.path.join(output_dir, novo_nome_json))
    print("    [OK] Processamento SST finalizado.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    arquivos_json = glob.glob(os.path.join(base_dir, "*.json"))

    if not arquivos_json:
        print(f"ERRO: Nenhum arquivo .json encontrado em: {base_dir}")
        sys.exit()

    for input_file in arquivos_json:
        process_file(input_file, base_dir)

    print("\n[SUCESSO] Processamento completo executado com sucesso.")


if __name__ == "__main__":
    main()
