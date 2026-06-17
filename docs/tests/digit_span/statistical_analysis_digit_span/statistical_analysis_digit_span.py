import glob
import os
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROGRAM_NAME = "Digit Span (DS)-OSWeb v1"
TASK_LABEL = "Digit Span (DS)"
TASK_FILTER = "Digit Span"
PREFIX = "DS"


def safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def format_ms(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f} ms"


def format_pct(value, digits=2):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


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


def make_output_dir(base_dir, nome):
    data_atual = datetime.now().strftime("%d-%m-%Y")
    out = os.path.join(base_dir, f"{nome}_{data_atual}")
    if os.path.exists(out):
        counter = 1
        while os.path.exists(f"{out}_{counter}"):
            counter += 1
        out = f"{out}_{counter}"
    os.makedirs(out, exist_ok=True)
    return out


def last_valid(df, col):
    if col not in df.columns:
        return np.nan
    values = df[col].dropna()
    return values.iloc[-1] if not values.empty else np.nan


def prepare_task_dataframe(df):
    task = df.copy()

    if "task_name" in task.columns:
        task = task[task["task_name"].astype(str).str.contains(TASK_FILTER, case=False, na=False)].copy()

    if "phase" in task.columns and task["phase"].astype(str).str.contains("official", case=False, na=False).any():
        task = task[task["phase"].astype(str).str.lower() == "official"].copy()

    if task.empty:
        raise RuntimeError(f"Nenhum trial oficial de {TASK_LABEL} encontrado no JSON.")

    defaults = {
        "condition": "unknown",
        "sequence_length": np.nan,
        "correct": np.nan,
        "error": np.nan,
        "response_time": np.nan,
        "trial_index": np.nan,
        "trial_total_time_ms": np.nan,
        "official_elapsed_ms": np.nan,
        "official_total_time_ms": np.nan,
        "sequence": np.nan,
        "expected_response": np.nan,
        "response": np.nan,
        "response_string": np.nan,
    }
    for col, default in defaults.items():
        if col not in task.columns:
            task[col] = default

    numeric_cols = [
        "sequence_length",
        "correct",
        "error",
        "response_time",
        "trial_index",
        "trial_total_time_ms",
        "official_elapsed_ms",
        "official_total_time_ms",
    ]
    for col in numeric_cols:
        task[col] = safe_num(task[col])

    if task["trial_index"].isna().all():
        task["trial_index"] = np.arange(1, len(task) + 1)

    task["condition"] = task["condition"].astype(str).str.lower().str.strip()
    condition_map = {
        "forward": "forward",
        "direct": "forward",
        "direta": "forward",
        "ordem direta": "forward",
        "backward": "backward",
        "reverse": "backward",
        "inversa": "backward",
        "ordem inversa": "backward",
    }
    task["condition"] = task["condition"].map(lambda x: condition_map.get(x, x))

    task["correct"] = task["correct"].fillna(0)
    if task["error"].isna().all():
        task["error"] = 1 - task["correct"]

    return task


def compute_metrics(task):
    correct_trials = task[task["correct"] == 1]
    spans = correct_trials.groupby("condition")["sequence_length"].max().to_dict()

    forward_span = int(spans.get("forward", 0) or 0)
    backward_span = int(spans.get("backward", 0) or 0)
    total_trials = len(task)
    total_correct = int(task["correct"].sum())

    forward = task[task["condition"] == "forward"]
    backward = task[task["condition"] == "backward"]
    rt = task[task["response_time"].notna()]
    trial_time = task[task["trial_total_time_ms"].notna()]

    official_total_time = last_valid(task, "official_total_time_ms")
    if pd.isna(official_total_time):
        official_total_time = task["official_elapsed_ms"].max()

    return {
        "total_trials": total_trials,
        "total_correct": total_correct,
        "total_errors": int((task["correct"] == 0).sum()),
        "accuracy": total_correct / total_trials if total_trials else np.nan,
        "forward_span": forward_span,
        "backward_span": backward_span,
        "span_sum": forward_span + backward_span,
        "forward_n": len(forward),
        "backward_n": len(backward),
        "forward_accuracy": forward["correct"].mean() if len(forward) else np.nan,
        "backward_accuracy": backward["correct"].mean() if len(backward) else np.nan,
        "mean_response_time": rt["response_time"].mean() if len(rt) else np.nan,
        "sd_response_time": rt["response_time"].std() if len(rt) > 1 else np.nan,
        "mean_trial_total_time": trial_time["trial_total_time_ms"].mean() if len(trial_time) else np.nan,
        "sd_trial_total_time": trial_time["trial_total_time_ms"].std() if len(trial_time) > 1 else np.nan,
        "official_total_time": official_total_time,
    }


def configure_plots():
    sns.set_theme(style="white", font_scale=1.25)
    plt.rcParams.update({
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "font.family": "sans-serif",
    })


def fig_spans(metrics, nome, out):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    labels = ["Direto\n(span imediato)", "Inverso\n(memória operacional)"]
    vals = [metrics["forward_span"], metrics["backward_span"]]
    colors = ["#4C72B0", "#C44E52"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=1.4, width=0.55)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.15,
            f"span {val}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylim(0, max(10, max(vals) + 2))
    ax.set_ylabel("Maior comprimento correto")
    ax.set_title(f"{TASK_LABEL} - {nome}", pad=20)

    box = (
        f"Trials oficiais: {metrics['total_trials']}\n"
        f"Acertos: {metrics['total_correct']}/{metrics['total_trials']} ({format_pct(metrics['accuracy'], 1)})\n"
        f"Soma dos spans: {metrics['span_sum']}\n"
        f"Tempo oficial: {format_ms(metrics['official_total_time'], 1)}"
    )
    ax.text(
        0.98,
        0.92,
        box,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray", lw=1.6),
    )

    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(out, f"{PREFIX}_Fig1_Spans.png"), dpi=300)
    plt.close(fig)


def fig_accuracy_by_length(task, nome, out):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    acc = task.groupby(["condition", "sequence_length"], as_index=False)["correct"].mean()
    acc["accuracy_pct"] = acc["correct"] * 100

    palette = {"forward": "#4C72B0", "backward": "#C44E52"}
    order = [c for c in ["forward", "backward"] if c in acc["condition"].unique()]

    if not acc.empty:
        sns.lineplot(
            data=acc,
            x="sequence_length",
            y="accuracy_pct",
            hue="condition",
            hue_order=order if order else None,
            palette=palette,
            marker="o",
            linewidth=2.6,
            markersize=8,
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                ["Direta" if l == "forward" else "Inversa" for l in labels],
                title="Ordem",
                loc="best",
            )

    ax.set_ylim(0, 105)
    ax.set_xlabel("Comprimento da sequência")
    ax.set_ylabel("Acurácia (%)")
    ax.set_title(f"Acurácia por Comprimento - {nome}", pad=20)

    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(out, f"{PREFIX}_Fig2_AcuraciaComprimento.png"), dpi=300)
    plt.close(fig)


def fig_box_by_condition(task, nome, out, y_col, y_label, title, filename):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    palette = {"forward": "#4C72B0", "backward": "#C44E52"}
    order = [c for c in ["forward", "backward"] if c in task["condition"].unique()]
    plot_df = task[task[y_col].notna()].copy()

    if not plot_df.empty and order:
        sns.boxplot(
            data=plot_df,
            x="condition",
            y=y_col,
            hue="condition",
            order=order,
            hue_order=order,
            palette=palette,
            ax=ax,
            width=0.5,
            showfliers=False,
            legend=False,
            boxprops=dict(alpha=0.75),
        )
        sns.stripplot(
            data=plot_df,
            x="condition",
            y=y_col,
            order=order,
            color="black",
            alpha=0.45,
            jitter=0.18,
            ax=ax,
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["Direto" if c == "forward" else "Inverso" for c in order])
    else:
        ax.text(
            0.5,
            0.5,
            f"Sem dados válidos em {y_col}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            bbox=dict(boxstyle="round", fc="white", ec="gray", lw=1.4),
        )

    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} - {nome}", pad=20)

    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(out, filename), dpi=300)
    plt.close(fig)


def fig_dynamics(task, nome, out):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    dyn = task.sort_values("trial_index").copy()
    dyn["acerto_rolling"] = dyn["correct"].rolling(5, min_periods=2).mean() * 100

    ax.plot(dyn["trial_index"], dyn["acerto_rolling"], color="#4B0082", linewidth=2.4)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Progresso da Tarefa (Trials Oficiais)")
    ax.set_ylabel("Acerto estimado (%)")
    ax.set_title(f"Dinâmica de Desempenho - {nome}", pad=20)

    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(out, f"{PREFIX}_Fig4_Dinamica.png"), dpi=300)
    plt.close(fig)


def save_tables(task, metrics, out):
    resumo = pd.DataFrame([
        {"Metrica": "trials_oficiais", "Valor": metrics["total_trials"]},
        {"Metrica": "acertos_totais", "Valor": metrics["total_correct"]},
        {"Metrica": "erros_totais", "Valor": metrics["total_errors"]},
        {"Metrica": "acuracia_total", "Valor": metrics["accuracy"]},
        {"Metrica": "span_direto", "Valor": metrics["forward_span"]},
        {"Metrica": "span_inverso", "Valor": metrics["backward_span"]},
        {"Metrica": "soma_dos_spans", "Valor": metrics["span_sum"]},
        {"Metrica": "acuracia_direta", "Valor": metrics["forward_accuracy"]},
        {"Metrica": "acuracia_inversa", "Valor": metrics["backward_accuracy"]},
        {"Metrica": "tempo_resposta_medio_ms", "Valor": metrics["mean_response_time"]},
        {"Metrica": "tempo_resposta_sd_ms", "Valor": metrics["sd_response_time"]},
        {"Metrica": "tempo_total_trial_medio_ms", "Valor": metrics["mean_trial_total_time"]},
        {"Metrica": "tempo_total_trial_sd_ms", "Valor": metrics["sd_trial_total_time"]},
        {"Metrica": "tempo_total_fase_oficial_ms", "Valor": metrics["official_total_time"]},
    ])
    resumo.to_csv(
        os.path.join(out, f"{PREFIX}_ResumoMetricas.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    cols = [
        "trial_index",
        "condition",
        "sequence_length",
        "sequence",
        "expected_response",
        "response",
        "response_string",
        "correct",
        "error",
        "response_time",
        "trial_total_time_ms",
        "official_elapsed_ms",
        "official_total_time_ms",
    ]
    cols = [c for c in cols if c in task.columns]
    task[cols].to_csv(
        os.path.join(out, f"{PREFIX}_TrialsProcessados.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_report(metrics, nome, out):
    with open(os.path.join(out, f"Relatorio_{PREFIX}.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== RELATÓRIO {TASK_LABEL.upper()}: {nome} ===\n")
        f.write(f"Programa: {PROGRAM_NAME}\n")
        f.write(f"Data de processamento: {datetime.now().strftime('%d-%m-%Y %H:%M')}\n")
        f.write("-" * 68 + "\n")

        f.write("AMOSTRA E DESEMPENHO GLOBAL\n")
        f.write(f"  Trials oficiais analisados:      {metrics['total_trials']}\n")
        f.write(
            f"  Acertos totais:                  {metrics['total_correct']} / "
            f"{metrics['total_trials']} ({format_pct(metrics['accuracy'])})\n"
        )
        f.write(f"  Erros totais:                    {metrics['total_errors']} / {metrics['total_trials']}\n")
        f.write(f"  Trials em ordem direta:          {metrics['forward_n']}\n")
        f.write(f"  Trials em ordem inversa:         {metrics['backward_n']}\n\n")

        f.write("MÉTRICAS DE SPAN\n")
        f.write(f"  Span direto:                     {metrics['forward_span']}\n")
        f.write(f"  Span inverso:                    {metrics['backward_span']}\n")
        f.write(f"  Soma dos spans:                  {metrics['span_sum']}\n")
        f.write(f"  Acurácia na ordem direta:        {format_pct(metrics['forward_accuracy'])}\n")
        f.write(f"  Acurácia na ordem inversa:       {format_pct(metrics['backward_accuracy'])}\n\n")

        f.write("MÉTRICAS TEMPORAIS DA FASE OFICIAL\n")
        f.write(f"  Tempo total da fase oficial:     {format_ms(metrics['official_total_time'])}\n")
        f.write(f"  Tempo médio de resposta:         {format_ms(metrics['mean_response_time'])}\n")
        f.write(f"  SD do tempo de resposta:         {format_ms(metrics['sd_response_time'])}\n")
        f.write(f"  Tempo total médio por trial:     {format_ms(metrics['mean_trial_total_time'])}\n")
        f.write(f"  SD do tempo total por trial:     {format_ms(metrics['sd_trial_total_time'])}\n\n")

        f.write("NOTAS METODOLÓGICAS\n")
        f.write("  - A condição direta estima span imediato/evocação sequencial direta.\n")
        f.write("  - A condição inversa adiciona manipulação mental e maior carga executiva.\n")
        f.write("  - O tempo de resposta é a latência da tela de evocação até ENTER/Continuar.\n")
        f.write("  - O tempo total da fase oficial é variável contextual/auditoria, não escore cognitivo primário.\n")
        f.write("  - O treino não entra nas métricas clínicas nem temporais.\n")
        f.write("-" * 68 + "\n")


def process_file(input_file, base_dir):
    print(f"\n>>> Lendo arquivo bruto: {os.path.basename(input_file)}")

    df = pd.read_json(input_file)
    nome = identify_subject(df)
    print(f"    Participante identificado: {nome}")

    out = make_output_dir(base_dir, nome)
    task = prepare_task_dataframe(df)
    metrics = compute_metrics(task)

    configure_plots()
    fig_spans(metrics, nome, out)
    fig_accuracy_by_length(task, nome, out)
    fig_box_by_condition(
        task,
        nome,
        out,
        "response_time",
        "Tempo de resposta na evocação (ms)",
        "Tempo de Resposta na Evocação",
        f"{PREFIX}_Fig3_TempoResposta.png",
    )
    fig_dynamics(task, nome, out)
    fig_box_by_condition(
        task,
        nome,
        out,
        "trial_total_time_ms",
        "Tempo total do trial oficial (ms)",
        "Tempo Total por Trial Oficial",
        f"{PREFIX}_Fig5_TempoTotalTrial.png",
    )
    save_tables(task, metrics, out)
    save_report(metrics, nome, out)

    raw_name = f"{PREFIX}_DadosBrutos_{os.path.basename(out)}.json"
    shutil.move(input_file, os.path.join(out, raw_name))

    print(f"    [OK] Processamento {TASK_LABEL} finalizado.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    arquivos = glob.glob(os.path.join(base_dir, "*.json"))

    if not arquivos:
        print(f"ERRO: Nenhum arquivo .json encontrado em: {base_dir}")
        sys.exit(1)

    for arq in arquivos:
        plt.close("all")
        process_file(arq, base_dir)

    print("\n[SUCESSO] Processamento completo executado com sucesso.")


if __name__ == "__main__":
    main()