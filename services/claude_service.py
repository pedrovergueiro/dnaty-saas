"""
Claude API service — generates explanations and deployment code for compression results.

Falls back to template strings when ANTHROPIC_API_KEY is not configured,
so the compress endpoint works even without the key set.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_SYSTEM = (
    "Voce e um especialista em machine learning eficiente e edge AI. "
    "Responda sempre em portugues, de forma objetiva e pratica."
)


def explain_compression(result: dict) -> tuple[str, str]:
    """
    Return (explanation, deployment_code) for the given CompressResult dict.

    Tries Claude API first; falls back to templates if key is missing or call fails.
    """
    try:
        from config import settings
        api_key = getattr(settings, "anthropic_api_key", "")
    except Exception:
        api_key = ""

    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set — using template fallback")
        return _template_explanation(result), _template_code(result)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        explanation   = _call_explain(client, result)
        deploy_code   = _call_code(client, result)
        return explanation, deploy_code
    except Exception as exc:
        logger.warning("Claude API error: %s — falling back to template", exc)
        return _template_explanation(result), _template_code(result)


# ── Claude calls ──────────────────────────────────────────────────────────────

def _call_explain(client, result: dict) -> str:
    orig_p  = result.get("original_params", 0)
    comp_p  = result.get("compressed_params", 0)
    acc     = result.get("accuracy", 0) * 100
    flops_r = result.get("flops_reduction", 0) * 100
    arch    = result.get("arch", [])

    prompt = (
        f"Explique em 3-4 frases os resultados de compressao de modelo a seguir:\n\n"
        f"- Parametros: {orig_p:,} -> {comp_p:,} (reducao de {(1-comp_p/max(orig_p,1))*100:.1f}%)\n"
        f"- Reducao de FLOPs: {flops_r:.1f}%\n"
        f"- Acuracia final: {acc:.2f}%\n"
        f"- Arquitetura encontrada: MLP com camadas ocultas {arch}\n\n"
        f"Explique: o que foi feito, o que isso significa na pratica "
        f"(velocidade, memoria, dispositivos onde pode rodar), e quando usar."
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_code(client, result: dict) -> str:
    arch       = result.get("arch", [128, 64])
    input_size = result.get("input_size", 784)
    n_classes  = result.get("n_classes", 10)
    n_hidden   = len(arch)

    prompt = (
        f"Gere codigo Python limpo e funcional para carregar e usar um modelo "
        f"PyTorch comprimido pelo dNATY.\n\n"
        f"Especificacoes do modelo:\n"
        f"- Entrada: {input_size} features\n"
        f"- Camadas ocultas: {arch}\n"
        f"- Saidas: {n_classes} classes\n\n"
        f"Inclua: (1) reconstituicao da arquitetura DynamicMLP, "
        f"(2) carregamento dos pesos (.pt), "
        f"(3) inferencia de exemplo, "
        f"(4) export ONNX em 3 linhas.\n"
        f"Sem comentarios excessivos. Codigo direto ao ponto."
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── Template fallbacks ────────────────────────────────────────────────────────

def _template_explanation(result: dict) -> str:
    flops_pct = result.get("flops_reduction", 0) * 100
    orig_p    = result.get("original_params", 0)
    comp_p    = result.get("compressed_params", 0)
    acc       = result.get("accuracy", 0) * 100
    params_r  = (1 - comp_p / max(orig_p, 1)) * 100

    return (
        f"dNATY encontrou uma arquitetura {flops_pct:.1f}% mais eficiente em FLOPs "
        f"e {params_r:.1f}% menor em parametros, mantendo {acc:.2f}% de acuracia. "
        f"O modelo comprimido usa {comp_p:,} parametros (original: {orig_p:,}) e "
        f"e indicado para execucao em dispositivos com recursos limitados como "
        f"Raspberry Pi, microcontroladores e aplicacoes mobile."
    )


def _template_code(result: dict) -> str:
    arch       = result.get("arch", [128, 64])
    input_size = result.get("input_size", 784)
    n_classes  = result.get("n_classes", 10)
    n_hidden   = len(arch)
    acts       = '["relu"] * ' + str(n_hidden)

    return f"""import torch
from dnaty.core.arch import DynamicMLP

model = DynamicMLP(
    layer_sizes=[{input_size}] + {arch},
    activations={acts},
    n_classes={n_classes},
)
model.load_state_dict(torch.load("model_compressed.pt", map_location="cpu"))
model.eval()

with torch.inference_mode():
    x = torch.randn(1, {input_size})
    pred = model(x).argmax(dim=1).item()
    print(f"Predicao: {{pred}}")

# Export ONNX
torch.onnx.export(model, x, "model_compressed.onnx",
                  input_names=["input"], output_names=["logits"])
"""
