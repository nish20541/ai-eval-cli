import json
import os
import csv
from typing import Dict, List, Tuple

import click
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib import utils
import numpy as np
from tqdm import tqdm

@click.group()
def main():
    """GoldenEval CLI

    Simple, PM-friendly evaluations against golden responses.
    """

def _semantic_agreement(reference: str, candidate: str) -> float:
    """Cheap semantic agreement proxy without embeddings.

    Returns a score in [0,1] based on token overlap (Jaccard). Placeholder until
    we add embeddings-based similarity.
    """
    ref_tokens = set(reference.lower().split())
    cand_tokens = set(candidate.lower().split())
    if not ref_tokens or not cand_tokens:
        return 0.0
    intersection = len(ref_tokens & cand_tokens)
    union = len(ref_tokens | cand_tokens)
    return intersection / union


def _tone_alignment(expected_tone: str, text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Map expected tone to VADER polarity and compare.

    Returns [0,1] agreement with expected tone buckets.
    """
    if not expected_tone:
        return 0.5
    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    tone = expected_tone.lower()
    if tone in {"friendly", "polite", "positive"}:
        return max(0.0, min(1.0, (compound + 1.0) / 2.0))
    if tone in {"serious", "neutral", "concise"}:
        # Neutral aligns with compound near 0
        return 1.0 - abs(compound)
    if tone in {"negative", "critical"}:
        return max(0.0, min(1.0, (1.0 - compound) / 2.0))
    return 0.5


def _clarity_score(text: str) -> float:
    """Normalize Flesch reading ease to [0,1]."""
    try:
        fre = flesch_reading_ease(text)
    except Exception:
        return 0.5
    # Flesch ranges roughly 0-100+. Clip and scale.
    fre = max(0.0, min(100.0, float(fre)))
    return fre / 100.0


def _score_against_golden(golden: str, output: str, expected_tone: str, analyzer: SentimentIntensityAnalyzer) -> Dict[str, float]:
    return {
        "semantic": _semantic_agreement(golden, output),
        "clarity": _clarity_score(output),
        "tone": _tone_alignment(expected_tone, output, analyzer),
    }


def _weighted_overall(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total_weight = sum(weights.values()) or 1.0
    return sum(scores[k] * weights.get(k, 0.0) for k in scores) / total_weight


def _call_openai_model(client: OpenAI, model: str, prompt: str) -> str:
    """Call OpenAI model and return response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise assistant. Respond with only the answer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"<error: {e}>"


def _generate_edge_cases(client: OpenAI, base_prompt: str) -> List[Dict[str, str]]:
    """Use the LLM to propose edge-case variants for paraphrase, constraint, noisy, ambiguity."""
    system = (
        "You generate short edge-case variants for evaluation. "
        "Return 4 bullet points labeled with types: Paraphrase, Constraint, Noisy, Ambiguity."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Base prompt: {base_prompt}\nGenerate 4 edge-case variants."},
            ],
            temperature=0.6,
            max_tokens=300,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        text = (
            "- Paraphrase: " + base_prompt + "\n"
            "- Constraint: Please answer in exactly one sentence. " + base_prompt + "\n"
            "- Noisy: pls ans fast!!! " + base_prompt + " :)\n"
            "- Ambiguity: " + base_prompt.replace("?", "") + " (be specific)"
        )

    variants: List[Dict[str, str]] = []
    for line in text.split("\n"):
        norm = line.strip("- ")
        if not norm:
            continue
        if ":" in norm:
            typ, prompt = norm.split(":", 1)
            variants.append({"type": typ.strip().lower(), "prompt": prompt.strip()})
    # Ensure we only keep up to 4 known types if possible
    ordered = []
    type_order = ["paraphrase", "constraint", "noisy", "ambiguity"]
    seen = set()
    for t in type_order:
        for v in variants:
            if v["type"].startswith(t) and t not in seen:
                ordered.append(v)
                seen.add(t)
                break
    if not ordered:
        ordered = variants[:4]
    return ordered


def _map_score_to_1_5(semantic: float, clarity: float, intent: float) -> int:
    """Combine three [0,1] signals to a 1–5 integer score."""
    overall = 0.6 * semantic + 0.2 * clarity + 0.2 * intent
    # Map [0,1] to 1..5
    buckets = np.digitize([overall], [0.2, 0.4, 0.6, 0.8])[0] + 1
    return int(buckets)





def _run_edge_eval(items: List[Dict[str, str]], client: OpenAI, analyzer: SentimentIntensityAnalyzer, model_a: str, model_b: str, outdir: str, dataset: str) -> Tuple[str, str, str]:
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "results_edge.csv")
    md_path = os.path.join(outdir, "results_edge.md")

    base_wins_a = base_wins_b = base_ties = 0
    edge_wins_a = edge_wins_b = edge_ties = 0

    rows_md: List[str] = []

    # Limit to first 5 prompts
    items = items[:5]
    click.echo("📊 Starting AI evaluation...")
    click.echo(f"📁 Dataset: {len(items)} prompts (limited to first 5)")
    click.echo(f"🤖 Models: {model_a} vs {model_b}")
    click.echo("")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "type", "base_prompt", "edge_prompt", "golden", "tone",
            "model_a", "model_b", "score_a", "score_b", "winner",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, item in enumerate(tqdm(items, desc="Processing prompts", unit="prompt")):
            click.echo(f"\n🔍 Prompt {i+1}/{len(items)}")
            base_prompt = item.get("prompt", "")
            golden = item.get("golden", "")
            tone = item.get("tone", "")

            click.echo("  📝 Evaluating base prompt...")
            # Evaluate base prompt first
            a_out = _call_openai_model(client, model_a, base_prompt)
            b_out = _call_openai_model(client, model_b, base_prompt)
            a_scores = _score_against_golden(golden, a_out, tone, analyzer)
            b_scores = _score_against_golden(golden, b_out, tone, analyzer)
            # Intent proxy: semantic for now
            score_a = _map_score_to_1_5(a_scores["semantic"], a_scores["clarity"], a_scores["semantic"])
            score_b = _map_score_to_1_5(b_scores["semantic"], b_scores["clarity"], b_scores["semantic"])
            base_winner = "A" if score_a > score_b else ("B" if score_b > score_a else "Tie")
            if base_winner == "A": base_wins_a += 1
            elif base_winner == "B": base_wins_b += 1
            else: base_ties += 1
            writer.writerow({
                "type": "base",
                "base_prompt": base_prompt,
                "edge_prompt": "",
                "golden": golden,
                "tone": tone,
                "model_a": a_out,
                "model_b": b_out,
                "score_a": score_a,
                "score_b": score_b,
                "winner": base_winner,
            })
            rows_md.append("\n".join([
                "----------------------------------------",
                f"**Base Prompt:** \"{base_prompt}\"",
                f"**Golden:** \"{golden}\"",
                f"**Model A Response:** {a_out}",
                f"**Model B Response:** {b_out}",
                f"**Winner:** {('Model A' if base_winner=='A' else ('Model B' if base_winner=='B' else 'Tie'))}",
            ]))

            click.echo("  🔄 Generating edge cases...")
            # Edge cases
            edges = _generate_edge_cases(client, base_prompt)
            click.echo(f"  📋 Generated {len(edges)} edge cases")
            for j, ev in enumerate(edges):
                e_type = ev.get("type", "edge")
                e_prompt = ev.get("prompt", base_prompt)
                click.echo(f"    🎯 Edge {j+1} ({e_type}): Evaluating...")
                ea = _call_openai_model(client, model_a, e_prompt)
                eb = _call_openai_model(client, model_b, e_prompt)
                ea_scores = _score_against_golden(golden, ea, tone, analyzer)
                eb_scores = _score_against_golden(golden, eb, tone, analyzer)
                ea_score = _map_score_to_1_5(ea_scores["semantic"], ea_scores["clarity"], ea_scores["semantic"]) 
                eb_score = _map_score_to_1_5(eb_scores["semantic"], eb_scores["clarity"], eb_scores["semantic"]) 
                e_winner = "A" if ea_score > eb_score else ("B" if eb_score > ea_score else "Tie")
                if e_winner == "A": edge_wins_a += 1
                elif e_winner == "B": edge_wins_b += 1
                else: edge_ties += 1
                click.echo(f"      ✅ Edge {j+1} winner: {('Model A' if e_winner=='A' else ('Model B' if e_winner=='B' else 'Tie'))}")
                writer.writerow({
                    "type": e_type,
                    "base_prompt": base_prompt,
                    "edge_prompt": e_prompt,
                    "golden": golden,
                    "tone": tone,
                    "model_a": ea,
                    "model_b": eb,
                    "score_a": ea_score,
                    "score_b": eb_score,
                    "winner": e_winner,
                })
                rows_md.append("\n".join([
                    f"**Edge ({e_type.title()}):** \"{e_prompt}\"",
                    f"**Model A Response:** {ea}",
                    f"**Model B Response:** {eb}",
                    f"**Winner:** {('Model A' if e_winner=='A' else ('Model B' if e_winner=='B' else 'Tie'))}",
                ]))

    click.echo("\n📄 Writing reports...")
    with open(md_path, "w", encoding="utf-8") as mf:
        summary = [
            "📊 AI Evaluation Report",
            f"Dataset: {os.path.relpath(dataset, outdir)}",
            f"Models: {model_a} vs {model_b}",
            "",
            "Summary:",
            f"- Normal-case wins — A: {base_wins_a}, B: {base_wins_b}, Ties: {base_ties}",
            f"- Edge-case wins — A: {edge_wins_a}, B: {edge_wins_b}, Ties: {edge_ties}",
        ]
        recommendation = (
            "Model A for accuracy; Model B for robustness." if base_wins_a > base_wins_b and edge_wins_b > edge_wins_a else
            "Model B for accuracy; Model A for robustness." if base_wins_b > base_wins_a and edge_wins_a > edge_wins_b else
            ("Model A overall." if (base_wins_a + edge_wins_a) > (base_wins_b + edge_wins_b) else ("Model B overall." if (base_wins_b + edge_wins_b) > (base_wins_a + edge_wins_a) else "Tie overall."))
        )
        summary.append(f"✅ Recommendation: {recommendation}")
        mf.write("\n".join(summary + ["\n"] + rows_md + ["\n"]))

    # PDF export
    pdf_path = os.path.join(outdir, "results_edge.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=LETTER, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Evaluation Report", styles['Title']))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(f"Dataset: {os.path.relpath(dataset, outdir)}", styles['Normal']))
    story.append(Paragraph(f"Models: {model_a} vs {model_b}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Summary", styles['Heading2']))
    story.append(Paragraph(f"Normal-case wins — A: {base_wins_a}, B: {base_wins_b}, Ties: {base_ties}", styles['Normal']))
    story.append(Paragraph(f"Edge-case wins — A: {edge_wins_a}, B: {edge_wins_b}, Ties: {edge_ties}", styles['Normal']))
    story.append(Paragraph(f"Recommendation: {recommendation}", styles['Normal']))
    story.append(Spacer(1, 0.25*inch))

    story.append(Paragraph("Details", styles['Heading2']))
    for block in rows_md:
        for line in block.split("\n"):
            if line.strip() == "----------------------------------------":
                story.append(Spacer(1, 0.2*inch))
            elif "**Model A Response:**" in line or "**Model B Response:**" in line:
                # Make model responses more prominent
                clean_line = line.replace("**", "")
                story.append(Paragraph(clean_line, styles['Heading3']))
            elif "**Winner:**" in line:
                # Make winner more prominent
                clean_line = line.replace("**", "")
                story.append(Paragraph(clean_line, styles['Heading3']))
            else:
                # Remove markdown formatting for PDF
                clean_line = line.replace("**", "")
                story.append(Paragraph(clean_line, styles['Normal']))
        story.append(Spacer(1, 0.15*inch))

    try:
        doc.build(story)
        click.echo("✅ Evaluation complete!")
        return csv_path, md_path, pdf_path
    except Exception as e:
        click.echo(f"⚠️  PDF generation failed: {e}")
        click.echo("✅ Evaluation complete!")
        return csv_path, md_path, ""


@main.command()
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False, path_type=str), help="Path to dataset JSON")
@click.option("--model-a", required=True, type=str, help="OpenAI model name for Model A (e.g., gpt-4o-mini)")
@click.option("--model-b", required=True, type=str, help="OpenAI model name for Model B")
@click.option("--outdir", default=".", show_default=True, type=click.Path(file_okay=False, path_type=str), help="Output directory for results")
def evaluate(dataset: str, model_a: str, model_b: str, outdir: str) -> None:
    """Run AI model evaluation with edge-case generation and 1–5 scoring."""
    with open(dataset, "r", encoding="utf-8") as f:
        items: List[Dict[str, str]] = json.load(f)

    os.makedirs(outdir, exist_ok=True)

    client = OpenAI()
    analyzer = SentimentIntensityAnalyzer()

    # Run AI evaluation with edge cases
    csv_edge, md_edge, pdf_edge = _run_edge_eval(items, client, analyzer, model_a, model_b, outdir, dataset)
    if pdf_edge:
        click.echo(f"Wrote {csv_edge}, {md_edge}, and {pdf_edge} (AI evaluation).")
    else:
        click.echo(f"Wrote {csv_edge} and {md_edge} (AI evaluation).")


## Removed separate edge subcommand; EdgeEval is default in evaluate
