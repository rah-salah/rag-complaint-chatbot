import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from src.rag_pipeline import load_vector_store, run_rag_pipeline
from groq import Groq

print("Loading...")
model = SentenceTransformer("all-MiniLM-L6-v2")
index, metadata = load_vector_store()
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
print("Ready.")

def answer_question(question, history):
    if not question.strip():
        return history, "", ""
    try:
        result = run_rag_pipeline(question, index, metadata, model, client)
        answer = result["answer"]
        sources = result["sources"]
        source_text = ""
        for i, src in enumerate(sources, 1):
            cid = src.get("complaint_id", "N/A")
            prod = src.get("product_category", "N/A")
            txt = src.get("text", "")
            source_text += f"**Source {i}** | ID: {cid} | Product: {prod}

{txt}

---

"
        history.append((question, answer))
        return history, source_text, ""
    except Exception as e:
        history.append((question, f"Error: {str(e)}"))
        return history, "", ""

with gr.Blocks(title="CrediTrust Complaint Analyst") as demo:
    gr.Markdown("# CrediTrust Financial — Complaint Analyst
Ask questions about customer complaints.")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            question_box = gr.Textbox(placeholder="e.g. What are the most common Credit Card complaints?", label="Your Question", lines=2)
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")
        with gr.Column(scale=1):
            sources_box = gr.Markdown(label="Retrieved Sources", value="*Sources will appear here.*")
    state = gr.State([])
    submit_btn.click(fn=answer_question, inputs=[question_box, state], outputs=[chatbot, sources_box, question_box])
    question_box.submit(fn=answer_question, inputs=[question_box, state], outputs=[chatbot, sources_box, question_box])
    clear_btn.click(fn=lambda: ([], "*Sources will appear here.*", "", []), outputs=[chatbot, sources_box, question_box, state])

if __name__ == "__main__":
    demo.launch()