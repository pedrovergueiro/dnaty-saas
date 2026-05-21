"""
Gera o relatório final em Markdown com todos os resultados reais.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
import numpy as np


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def generate_report(results_dir: str = "results", output_path: str = "DNATY_RESULTS.md") -> str:
    exp1_path = os.path.join(results_dir, "exp1_results.json")
    exp3_path = os.path.join(results_dir, "exp3_cl_results.json")

    has_exp1 = os.path.exists(exp1_path)
    has_exp3 = os.path.exists(exp3_path)

    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        "# dNaty — Resultados Experimentais Reais",
        "",
        f"> Gerado automaticamente em {now}",
        f"> Todos os valores são dados reais de execução — não ilustrativos.",
        "",
        "---",
        "",
    ]

    # ── Experimento 1 ─────────────────────────────────────────────────────────
    if has_exp1:
        exp1 = load_json(exp1_path)
        lines += [
            "## Experimento 1 — MNIST e FashionMNIST",
            "",
            "### Configuração",
            "",
        ]

        for dataset_name, data in exp1.items():
            s = data["summary"]
            dnaty_s = s["dnaty"]
            mlp_s = s["mlp"]
            ga_s = s["ga"]
            ttest = s["ttest_dnaty_vs_mlp"]

            lines += [
                f"### {dataset_name}",
                "",
                "| Método | Acurácia (média ± std) | Params (média) |",
                "|--------|------------------------|----------------|",
            ]

            dnaty_runs = data["dnaty"]
            mlp_runs = data["baselines"]
            ga_runs = data["baselines"]

            dnaty_params = int(np.mean([r["n_params"] for r in dnaty_runs]))
            mlp_params = int(np.mean([r["mlp_params"] for r in mlp_runs]))
            ga_params = int(np.mean([r["ga_params"] for r in ga_runs]))

            lines += [
                f"| **dNaty** | **{dnaty_s['mean']:.4f} ± {dnaty_s['std']:.4f}** | **{dnaty_params:,}** |",
                f"| MLP Fixo | {mlp_s['mean']:.4f} ± {mlp_s['std']:.4f} | {mlp_params:,} |",
                f"| GA Puro | {ga_s['mean']:.4f} ± {ga_s['std']:.4f} | {ga_params:,} |",
                "",
                f"**Teste-t pareado (dNaty vs MLP):** t={ttest['t']:.3f}, p={ttest['p']:.4f}, d={ttest['d']:.3f}"
                + (" ✓ significativo (p<0.05)" if ttest['p'] < 0.05 else " (não significativo)"),
                "",
            ]

            # Validação do Teorema 1
            dg_ok = s.get("theorem1_delta_grad_positive", False)
            dm_ok = s.get("theorem1_delta_mem_positive", False)
            lines += [
                "#### Validação do Teorema dNaty-Convergence",
                "",
                f"- δ_grad ≥ 0 em todas as gerações × 5 seeds: {'✓ CONFIRMADO' if dg_ok else '✗ VIOLADO'}",
                f"- δ_mem ≥ 0 após geração 3 × 5 seeds: {'✓ CONFIRMADO' if dm_ok else '✗ VIOLADO'}",
                "",
            ]

            # Tabela de convergência por geração (seed 0)
            if dnaty_runs:
                seed0 = dnaty_runs[0]
                hist = seed0.get("history", [])
                if hist:
                    lines += [
                        "#### Convergência por Geração (seed=0)",
                        "",
                        "| Geração | Acurácia | δ_grad | δ_mem | Params |",
                        "|---------|----------|--------|-------|--------|",
                    ]
                    step = max(1, len(hist) // 10)
                    for h in hist[::step]:
                        lines.append(
                            f"| {h['gen']} | {h['best_acc']:.4f} | {h['delta_grad']:.6f} | {h['delta_mem']:.6f} | {h['n_params']:,} |"
                        )
                    lines.append("")

    else:
        lines += [
            "## Experimento 1 — MNIST e FashionMNIST",
            "",
            "> ⚠️ Arquivo `results/exp1_results.json` não encontrado. Execute `run_experiments.py` primeiro.",
            "",
        ]

    # ── Experimento 3 ─────────────────────────────────────────────────────────
    if has_exp3:
        exp3 = load_json(exp3_path)
        s = exp3["summary"]
        lines += [
            "## Experimento 3 — Split-MNIST: Continual Learning",
            "",
            "### Métricas BWT / FWT / FM (média ± std, 5 seeds)",
            "",
            "| Método | BWT ↑ (0=ideal) | FWT ↑ | FM ↓ |",
            "|--------|-----------------|-------|------|",
            f"| **dNaty** | **{s['dnaty_bwt']['mean']:.4f} ± {s['dnaty_bwt']['std']:.4f}** | {s['dnaty_fwt']:.4f} | {s['dnaty_fm']:.4f} |",
            f"| EWC | {s['ewc_bwt']['mean']:.4f} ± {s['ewc_bwt']['std']:.4f} | — | — |",
            f"| MLP Fixo (sem CL) | {s['mlp_bwt']['mean']:.4f} ± {s['mlp_bwt']['std']:.4f} | — | — |",
            "",
        ]

        ttest = s.get("ttest_dnaty_vs_ewc_bwt", {})
        if ttest:
            lines += [
                f"**Teste-t pareado (dNaty vs EWC, BWT):** t={ttest.get('t', 0):.3f}, p={ttest.get('p', 1):.4f}, d={ttest.get('d', 0):.3f}"
                + (" ✓ significativo (p<0.05)" if ttest.get('p', 1) < 0.05 else " (não significativo)"),
                "",
            ]

        # Matriz R do seed 0
        if exp3["dnaty"]:
            R = np.array(exp3["dnaty"][0]["R"])
            T = R.shape[1]
            lines += [
                "### Matriz R[i,j] — dNaty seed=0",
                "",
                "R[i,j] = acurácia na tarefa j após treinar até tarefa i",
                "",
                "| Após tarefa \\ Tarefa | " + " | ".join([f"T{j}" for j in range(T)]) + " |",
                "|" + "---|" * (T + 1),
            ]
            for i in range(T):
                row = f"| Após T{i} | "
                row += " | ".join([f"{R[i,j]:.4f}" if j <= i else "—" for j in range(T)])
                row += " |"
                lines.append(row)
            lines.append("")

    else:
        lines += [
            "## Experimento 3 — Split-MNIST: Continual Learning",
            "",
            "> ⚠️ Arquivo `results/exp3_cl_results.json` não encontrado. Execute `run_experiments.py` primeiro.",
            "",
        ]

    # ── Metodologia ───────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Metodologia",
        "",
        "### Configuração dos Experimentos",
        "",
        "| Parâmetro | Valor |",
        "|-----------|-------|",
        "| Seeds | 5 (0, 1, 2, 3, 4) |",
        "| Gerações (N_GENERATIONS) | 30 |",
        "| População (N_POP) | 10 |",
        "| Passos locais (T_LOCAL) | 3 |",
        "| Decaimento memória (γ) | 0.99 |",
        "| Taxa de aprendizado | 1e-3 |",
        "| λ₁ (custo estrutural) | 1e-4 |",
        "| Top-k% micro-adaptação | 3% |",
        "| Operadores | 10 (add_neuron, remove_neuron, add_skip, change_activation, split_layer, merge_layers, prune_connections, duplicate_module, add_conv_block, depthwise_sep) |",
        "",
        "### O que foi implementado",
        "",
        "- **EpisodicMemory**: acumulação com decaimento γ, softmax com temperatura τ, pruning por relevância",
        "- **10 operadores estruturais**: todos com garantias formais, rollback automático se inválido",
        "- **NSGA-II corrigido**: índices inteiros, sem bug de hashability, crowding distance",
        "- **Treino local SAM+Adam**: L_total = CE + λ₁·C(A) + λ₂·S(θ,A)",
        "- **Micro-adaptação**: top-3% parâmetros por ‖∂L/∂θ_j‖",
        "- **Métricas CL**: BWT, FWT, FM (Lopez-Paz et al., 2017)",
        "- **Análise estatística**: teste-t pareado, Cohen's d, ANOVA",
        "",
        "### Baselines implementados",
        "",
        "- MLP Fixo (3 camadas, Adam, 20 epochs)",
        "- GA Puro (sem backprop, mutação de pesos aleatória)",
        "- EWC (Elastic Weight Consolidation, λ=1000)",
        "",
        "---",
        "",
        f"*Relatório gerado por `dnaty/analysis/report.py` em {now}*",
    ]

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Relatório salvo em: {output_path}")
    return content
