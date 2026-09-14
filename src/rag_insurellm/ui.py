"""Gradio UI for Insurellm Expert Assistant."""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr
import pandas as pd

from rag_insurellm.config import FINAL_K
from rag_insurellm.evaluate import load_tests

EXAMPLES = [
    "Who founded Insurellm?",
    "Who won the prestigious IIOTY award in 2023?",
    "Where is Insurellm's headquarters located?",
]

HEADER = """
<div class="ia-header">
  <div class="ia-brand">
    <div class="ia-logo">IL</div>
    <div>
      <div class="ia-title">Insurellm Expert Assistant</div>
      <div class="ia-sub">RAG-powered Q&amp;A over the Insurellm knowledge base</div>
    </div>
  </div>
  <div class="ia-badge">
    <div class="ia-badge-title">Built for portfolio</div>
    <div class="ia-badge-sub">Not a real Insurellm product</div>
  </div>
</div>
"""

SIDEBAR = """
<div class="ia-side">
  <div class="ia-side-title">
    <span class="ia-ico">📄</span> About this demo
  </div>
  <p>This is a RAG-based assistant over the Insurellm knowledge base. It answers questions using retrieved documents, not general web search.</p>
  <div class="ia-step">
    <div class="ia-step-ico">🔍</div>
    <div>
      <strong>Query rewriting</strong>
      <div>Your question is reformulated to improve retrieval quality.</div>
    </div>
  </div>
  <div class="ia-step">
    <div class="ia-step-ico">📚</div>
    <div>
      <strong>Retrieval + reranking</strong>
      <div>We retrieve relevant chunks and rerank them for better context.</div>
    </div>
  </div>
  <div class="ia-step">
    <div class="ia-step-ico">📝</div>
    <div>
      <strong>Source-grounded answers</strong>
      <div>Answers are generated from the retrieved documents, with citations so you can verify the facts.</div>
    </div>
  </div>
  <div class="ia-tip">
    <strong>💡 Tip</strong>
    <div>Always check the sources to verify information.</div>
  </div>
</div>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 12px 24px 24px !important; font-family: Inter, system-ui, sans-serif !important; }
.gradio-container, .main, body { background: #f4f6fb !important; }
footer, .footer { display: none !important; }
#ia-app { background: transparent; }

.ia-header { display: flex; justify-content: space-between; align-items: flex-start; padding: 8px 4px 18px; }
.ia-brand { display: flex; gap: 12px; align-items: center; }
.ia-logo { width: 40px; height: 40px; border-radius: 10px; background: #6d5efc; color: #fff; font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 14px; }
.ia-title { font-size: 22px; font-weight: 700; color: #1c1e26; line-height: 1.2; }
.ia-sub { font-size: 13px; color: #6b7280; margin-top: 2px; }
.ia-badge { background: #eef0f6; border-radius: 12px; padding: 8px 14px; text-align: right; }
.ia-badge-title { font-size: 12px; font-weight: 600; color: #374151; }
.ia-badge-sub { font-size: 11px; color: #9ca3af; }

.ia-side { background: #fff; border: 1px solid #e8eaf0; border-radius: 16px; padding: 20px 18px; }
.ia-side-title { font-weight: 700; font-size: 16px; margin-bottom: 10px; display: flex; gap: 8px; align-items: center; color: #1c1e26; }
.ia-side p { color: #4b5563; font-size: 13.5px; line-height: 1.5; margin: 0 0 16px; }
.ia-step { display: flex; gap: 10px; margin: 14px 0; color: #4b5563; font-size: 13px; line-height: 1.4; align-items: center; }
.ia-step strong { display: block; color: #1c1e26; font-size: 13.5px; margin-bottom: 2px; }
.ia-step-ico { width: 28px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; line-height: 1; font-size: 18px; }
.ia-tip { margin-top: 18px; background: #f7f4ff; border-radius: 12px; padding: 12px 14px; font-size: 13px; color: #4b5563; }
.ia-tip strong { color: #1c1e26; }

.ia-sources-wrap { margin: 4px 0 8px; }
.ia-sources-h { font-size: 13px; font-weight: 600; color: #374151; margin: 0 0 10px; }
.ia-cards {
  display: grid;
  grid-auto-flow: column;
  grid-template-rows: auto;
  grid-auto-columns: calc((100% - 20px) / 3);
  gap: 10px;
  overflow-x: auto;
  padding-bottom: 8px;
}
.ia-card { background: #fff; border: 1px solid #e8eaf0; border-radius: 12px; padding: 12px; min-height: 120px; cursor: pointer; }
.ia-card:hover { border-color: #6d5efc; }
.ia-dialog { border: none; border-radius: 16px; padding: 0; max-width: 720px; width: 92%; max-height: 80vh; }
.ia-dialog::backdrop { background: rgba(17, 24, 39, 0.45); }
.ia-dialog-inner { padding: 20px 22px 16px; }
.ia-dialog-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
.ia-dialog-top h3 { margin: 0; font-size: 16px; }
.ia-dialog-meta { font-size: 12px; color: #6b7280; margin-bottom: 10px; }
.ia-dialog pre { white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.5; color: #1c1e26; margin: 0; max-height: 56vh; overflow: auto; font-family: Inter, system-ui, sans-serif; }
.ia-dialog-close { border: none; background: #f3f4f6; border-radius: 8px; padding: 6px 10px; cursor: pointer; font-size: 13px; }
.ia-card-file { font-size: 12px; font-weight: 600; color: #1c1e26; }
.ia-card-type { font-size: 11px; color: #9ca3af; float: right; }
.ia-card-title { font-size: 13px; font-weight: 600; margin: 8px 0 6px; color: #1c1e26; }
.ia-card-sn { font-size: 12px; color: #6b7280; line-height: 1.4; }
.ia-card-pg { font-size: 11px; color: #9ca3af; margin-top: 8px; }

.ia-metrics { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 16px; }
.ia-metric { background: #fff; border: 1px solid #e8eaf0; border-radius: 14px; padding: 14px 16px; min-width: 140px; flex: 1; }
.ia-metric .lab { font-size: 12px; color: #6b7280; }
.ia-metric .val { font-size: 26px; font-weight: 700; margin-top: 4px; }
.ia-eval-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 8px; }
.ia-eval-head h3 { margin: 0; font-size: 18px; }
.ia-eval-head p { margin: 4px 0 0; color: #6b7280; font-size: 13px; }
.ia-eval-count { text-align: right; font-size: 13px; color: #6b7280; background: #eef4ff; border-radius: 12px; padding: 10px 14px; }
.ia-eval-count b { color: #1c1e26; display: block; font-size: 14px; }
.ia-eval-panel { background: transparent; }
.ia-run-card {
  border-radius: 16px;
  padding: 18px 16px 16px;
  text-align: center;
  cursor: pointer;
  user-select: none;
}
.ia-run-card-ret { background: #F8F5FF; border: 1px solid #EDE7FF; }
.ia-run-card-ans { background: #EFF6FF; border: 1px solid #DBEAFE; }
.ia-run-title {
  font-size: 15px;
  font-weight: 600;
  line-height: 1.3;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}
.ia-run-title svg { width: 16px; height: 16px; flex-shrink: 0; }
.ia-run-card-ret .ia-run-title { color: #7C3AED; }
.ia-run-card-ans .ia-run-title { color: #2563EB; }
.ia-run-sub { margin-top: 4px; font-size: 12px; font-weight: 400; color: #6B7280; }
#ia-run-ret, #ia-run-ans, #ia-app .ia-run-hidden {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  opacity: 0 !important;
  pointer-events: none !important;
  margin: 0 !important;
  padding: 0 !important;
  border: none !important;
}

#ia-app .tabs { border: none !important; background: transparent !important; }
#ia-app .tab-nav {
  display: inline-flex !important;
  width: fit-content !important;
  border: none !important;
  border-bottom: none !important;
  background: #eef0f6 !important;
  border-radius: 12px !important;
  padding: 4px !important;
  gap: 4px !important;
  margin-bottom: 16px !important;
  box-shadow: none !important;
}
#ia-app .tab-nav button {
  border-radius: 10px !important;
  padding: 8px 18px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  border: none !important;
  background: transparent !important;
  color: #4b5563 !important;
  box-shadow: none !important;
}
#ia-app .tab-nav button.selected,
#ia-app .tab-nav button[aria-selected="true"] {
  background: #4f46e5 !important;
  color: #fff !important;
}
#ia-app .tab-wrapper, #ia-app .tabitem { border: none !important; background: transparent !important; }
#ia-app .tab-nav::after, #ia-app .tab-nav button::after { display: none !important; }
#ia-app .chatbot { background: #fff; border-radius: 16px; border: 1px solid #e8eaf0; }
"""

HEAD = """
<script>
function iaClickRun(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const roots = [el, el.shadowRoot].filter(Boolean);
  for (const root of roots) {
    const btn = root.querySelector("button");
    if (btn) { btn.click(); return; }
  }
  el.click();
}
document.addEventListener("click", function (e) {
  if (!e.target || !e.target.closest) return;
  if (e.target.closest(".ia-run-card-ret")) iaClickRun("ia-run-ret");
  if (e.target.closest(".ia-run-card-ans")) iaClickRun("ia-run-ans");
});
</script>
"""

RUN_RETRIEVAL_CARD = """
<div class="ia-run-card ia-run-card-ret" role="button" tabindex="0">
  <div class="ia-run-title">
    <svg viewBox="0 0 24 24" fill="none" stroke="#7C3AED" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 3v6h-6"/></svg>
    Run Retrieval Evaluation
  </div>
  <div class="ia-run-sub">Evaluate retriever performance (MRR, nDCG, etc.)</div>
</div>
"""

RUN_ANSWER_CARD = """
<div class="ia-run-card ia-run-card-ans" role="button" tabindex="0">
  <div class="ia-run-title">
    <svg viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h5"/></svg>
    Run Answer Evaluation
  </div>
  <div class="ia-run-sub">Evaluate answer quality (LLM-as-a-judge)</div>
</div>
"""


def short_source(source: str) -> str:
    marker = "knowledge-base/"
    idx = source.replace("\\", "/").find(marker)
    return source[idx:] if idx >= 0 else Path(source).name


def source_cards(context) -> str:
    if not context:
        return "<div class='ia-sources-wrap'></div>"
    shown = context[:FINAL_K]
    cards, dialogs = [], []
    for i, doc in enumerate(shown, start=1):
        source = short_source(str(doc.metadata.get("source", "unknown")))
        name = Path(source).name
        kind = str(doc.metadata.get("type", "Document")).title()
        lines = [ln.strip() for ln in doc.page_content.splitlines() if ln.strip()]
        title = lines[0][:80] if lines else name
        snippet = " ".join(lines[1:])[:140] if len(lines) > 1 else doc.page_content[:140]
        dlg_id = f"ia-src-dlg-{i}"
        cards.append(
            f"""<div class="ia-card" onclick="document.getElementById('{dlg_id}').showModal()">
              <div><span class="ia-card-file">{html.escape(name)}</span>
              <span class="ia-card-type">{html.escape(kind)}</span></div>
              <div class="ia-card-title">{html.escape(title)}</div>
              <div class="ia-card-sn">{html.escape(snippet)}...</div>
              <div class="ia-card-pg">Source {i} · {html.escape(name)}</div>
            </div>"""
        )
        dialogs.append(
            f"""<dialog id="{dlg_id}" class="ia-dialog"
              onclick="if (event.target === this) this.close()">
              <div class="ia-dialog-inner">
                <div class="ia-dialog-top">
                  <h3>{html.escape(title)}</h3>
                  <form method="dialog"><button class="ia-dialog-close">Close</button></form>
                </div>
                <div class="ia-dialog-meta">{html.escape(kind)} · {html.escape(source)}</div>
                <pre>{html.escape(doc.page_content)}</pre>
              </div>
            </dialog>"""
        )
    return (
        f"<div class='ia-sources-wrap'><div class='ia-sources-h'>🔗 Sources ({len(shown)})</div>"
        f"<div class='ia-cards'>{''.join(cards)}</div>{''.join(dialogs)}</div>"
    )


def _metric_card(label: str, value_html: str, color: str) -> str:
    return (
        f'<div class="ia-metric"><div class="lab">{html.escape(label)}</div>'
        f'<div class="val" style="color:{color}">{value_html}</div></div>'
    )


def retrieval_metrics_html(retrieval: dict) -> str:
    if not retrieval:
        return "<div class='ia-metrics'></div>"
    return f"""
    <div class="ia-metrics">
      {_metric_card("Mean Reciprocal Rank (MRR)", f"{retrieval['mrr']:.5f}", "#6d5efc")}
      {_metric_card("Normalized DCG (nDCG)", f"{retrieval['ndcg']:.5f}", "#6d5efc")}
      {_metric_card("Keyword Coverage", f"{retrieval['keyword_coverage']:.1f}%", "#16a34a")}
    </div>
    """


def answer_metrics_html(answer: dict) -> str:
    if not answer:
        return "<div class='ia-metrics'></div>"
    return f"""
    <div class="ia-metrics">
      {_metric_card("Accuracy", f"{answer['accuracy']:.1f} / 5", "#6d5efc")}
      {_metric_card("Completeness", f"{answer['completeness']:.1f} / 5", "#6d5efc")}
      {_metric_card("Relevance", f"{answer['relevance']:.1f} / 5", "#16a34a")}
    </div>
    """


def retrieval_chart_df(retrieval: dict) -> pd.DataFrame:
    return pd.DataFrame((retrieval or {}).get("by_category") or [])


def answer_chart_df(answer: dict) -> pd.DataFrame:
    return pd.DataFrame((answer or {}).get("by_category") or [])


def retrieval_barplot(df: pd.DataFrame) -> gr.BarPlot:
    return gr.BarPlot(
        value=df,
        x="Category",
        y="Average MRR",
        title="Retrieval Performance by Category",
        y_lim=[0, 1],
        height=320,
    )


def answer_barplot(df: pd.DataFrame) -> gr.BarPlot:
    return gr.BarPlot(
        value=df,
        x="Category",
        y="Average Accuracy",
        title="Average Accuracy by Category",
        y_lim=[0, 5],
        height=320,
    )


def build_ui(chat, load_eval_results, run_retrieval_eval, run_answer_eval):
    n_tests = len(load_tests())
    saved = load_eval_results()
    retrieval = saved.get("retrieval", {})
    answer = saved.get("answer", {})

    with gr.Blocks(title="Insurellm Expert Assistant", elem_id="ia-app", fill_width=True) as demo:
        gr.HTML(HEADER)
        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=7):
                        chatbot = gr.Chatbot(
                            show_label=False,
                            height=420,
                            buttons=["copy"],
                            placeholder="Ask a question to get started.",
                        )
                        sources = gr.HTML(visible=False)
                        with gr.Row():
                            message = gr.Textbox(
                                placeholder="Ask a question about Insurellm...",
                                show_label=False,
                                scale=8,
                                container=False,
                            )
                            send = gr.Button("Send", variant="primary", scale=1)
                        gr.Examples(examples=EXAMPLES, inputs=message, label="💡 Example questions")
                    with gr.Column(scale=3):
                        gr.HTML(SIDEBAR)

            with gr.Tab("Evaluation"):
                with gr.Column(elem_classes=["ia-eval-panel"]):
                    gr.HTML(
                        f"""<div class="ia-eval-top">
                        <div class="ia-eval-head">
                          <h3>RAG Evaluation</h3>
                          <p>Evaluate retrieval and answer quality for the Insurellm RAG system.</p>
                        </div>
                        <div class="ia-eval-count"><b>Test set: {n_tests} questions</b>
                        Same knowledge base as the chat demo.</div>
                        </div>"""
                    )
                    with gr.Row():
                        retrieval_metrics = gr.HTML(retrieval_metrics_html(retrieval))
                        answer_metrics = gr.HTML(answer_metrics_html(answer))
                    with gr.Row():
                        retrieval_chart = retrieval_barplot(retrieval_chart_df(retrieval))
                        answer_chart = answer_barplot(answer_chart_df(answer))
                    with gr.Row():
                        gr.HTML(RUN_RETRIEVAL_CARD)
                        gr.HTML(RUN_ANSWER_CARD)
                    retrieval_btn = gr.Button(
                        "Run Retrieval Evaluation",
                        elem_id="ia-run-ret",
                        elem_classes=["ia-run-hidden"],
                    )
                    answer_btn = gr.Button(
                        "Run Answer Evaluation",
                        elem_id="ia-run-ans",
                        elem_classes=["ia-run-hidden"],
                    )

        def put_message(message, history):
            if not message or not str(message).strip():
                return "", history
            return "", history + [{"role": "user", "content": message}]

        send.click(put_message, [message, chatbot], [message, chatbot]).then(
            chat, chatbot, [chatbot, sources]
        )
        message.submit(put_message, [message, chatbot], [message, chatbot]).then(
            chat, chatbot, [chatbot, sources]
        )
        retrieval_btn.click(run_retrieval_eval, outputs=[retrieval_metrics, retrieval_chart])
        answer_btn.click(run_answer_eval, outputs=[answer_metrics, answer_chart])

    return demo
