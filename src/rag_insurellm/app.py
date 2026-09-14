"""Chat + evaluation handlers for Insurellm RAG."""

from __future__ import annotations

import os

import gradio as gr

from rag_insurellm.evaluate import (
    load_eval_results,
    run_answer_benchmark,
    run_retrieval_benchmark,
    save_eval_results,
)
from rag_insurellm.generate import answer_question
from rag_insurellm.ui import (
    CSS,
    HEAD,
    answer_barplot,
    answer_chart_df,
    answer_metrics_html,
    build_ui,
    retrieval_barplot,
    retrieval_chart_df,
    retrieval_metrics_html,
    source_cards,
)


def chat(history):
    last_message = history[-1]["content"][0]["text"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, gr.update(value=source_cards(context), visible=True)


def run_retrieval_eval(progress=gr.Progress()):
    retrieval = run_retrieval_benchmark(
        on_progress=lambda prog_value, count: progress(
            prog_value, desc=f"Evaluating test {count}..."
        )
    )
    save_eval_results({"retrieval": retrieval})
    return retrieval_metrics_html(retrieval), retrieval_barplot(retrieval_chart_df(retrieval))


def run_answer_eval(progress=gr.Progress()):
    answer = run_answer_benchmark(
        on_progress=lambda prog_value, count: progress(
            prog_value, desc=f"Evaluating test {count}..."
        )
    )
    save_eval_results({"answer": answer})
    return answer_metrics_html(answer), answer_barplot(answer_chart_df(answer))


def main():
    demo = build_ui(
        chat=chat,
        load_eval_results=load_eval_results,
        run_retrieval_eval=run_retrieval_eval,
        run_answer_eval=run_answer_eval,
    )
    theme = gr.themes.Soft(primary_hue="purple", font=["Inter", "system-ui", "sans-serif"])
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        theme=theme,
        css=CSS,
        head=HEAD,
    )


if __name__ == "__main__":
    main()
