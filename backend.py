import os
import json
import re
from datetime import datetime
from typing import List, Dict
from collections import Counter

import matplotlib.pyplot as plt

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


class DocumentationAssistant:
    """
    Strict AI Documentation Assistant:
    - Answers metadata questions
    - Answers content questions ONLY if present
    - Rejects all out-of-scope questions
    """

    def __init__(self, data_path="data", model_name="mistral"):
        self.data_path = data_path
        self.chat_history = []
        self.documents = []
        self.loaded_files = []

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # -------- LLM 1: ANSWER GENERATOR --------
        self.answer_llm = OllamaLLM(
            model=model_name,
        temperature=0.4,
        top_p=0.9,
        top_k=40,
        num_predict=200
        )

        # -------- LLM 2: VALIDATOR (STRICT) --------
        self.validator_llm = OllamaLLM(
            model=model_name,
            temperature=0.0,
            top_p=0.1,
            top_k=10,
            num_predict=5
        )

        self.vectorstore = None
        self.retriever = None

        self.prompt = PromptTemplate(
            template="""Answer ONLY using the context below.

Rules:
- Answer in 1–2 sentences.
- If the answer is not explicitly present, say exactly:
  Not found in documentation.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        self.validation_prompt = PromptTemplate(
            template="""
You are validating an AI answer.

Context:
{context}

Answer:
{answer}

Question:
{question}

Is the answer fully supported by the context?
Reply with ONLY one word: YES or NO.
""",
            input_variables=["context", "answer", "question"]
        )


    # ---------- INGESTION ----------

    def load_documents(self):
        docs = []
        self.loaded_files.clear()

        for file in os.listdir(self.data_path):
            path = os.path.join(self.data_path, file)

            if file.endswith(".pdf"):
                pages = PyPDFLoader(path).load()
                docs.extend(pages)
                self.loaded_files.append(file)

            elif file.endswith(".txt"):
                pages = TextLoader(path, encoding="utf-8").load()
                docs.extend(pages)
                self.loaded_files.append(file)

        self.documents = docs
        return docs

    def build_vectorstore(self):
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
        chunks = splitter.split_documents(self.documents)

        for c in chunks:
            c.page_content = re.sub(r"\s+", " ", c.page_content).strip()

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # 🔥 IMPORTANT: remove score_threshold
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )


    # ---------- INTENT CLASSIFICATION ----------

    def classify_intent(self, query: str) -> str:
        q = query.lower()

        if re.search(r"(file|document|pdf)", q):
            return "METADATA"

        return "CONTENT"
    
    def load_chat_history(self):
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r", encoding="utf-8") as f:
                self.chat_history = json.load(f)
        else:
            self.chat_history = []

    def save_chat_history(self):
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, indent=4)



    # ---------- HANDLERS ----------

    def handle_metadata(self, intent: str) -> Dict:
        if intent == "METADATA_LIST":
            return {
                "answer": "The following documents are loaded:\n" +
                          "\n".join(f"• {f}" for f in self.loaded_files),
                "sources": []
            }

        combined_text = " ".join(d.page_content for d in self.documents[:5])
        prompt = "Summarize the uploaded documentation:\n\n" + combined_text[:2000]
        summary = self.answer_llm.invoke(prompt).strip()

        return {
            "answer": summary,
            "sources": self.loaded_files
        }

    def handle_content_query(self, question: str) -> Dict:
        docs = self.retriever.invoke(question)

        # ❌ No context retrieved → refuse
        if not docs:
            return self.refusal()

        # -------- Build context --------
        context_blocks = []
        for d in docs[:6]:
            text = re.sub(r"\s+", " ", d.page_content).strip()
            if len(text) > 60:
                context_blocks.append(text)

        if not context_blocks:
            return self.refusal()

        context = "\n\n".join(context_blocks)

        # -------- LLM 1: GENERATE ANSWER --------
        generation_prompt = f"""
Answer the question using ONLY the documentation below.
If the documentation does not contain the answer, say exactly:
OUT_OF_CONTEXT

Documentation:
{context}

Question:
{question}

Answer:
"""
        answer = self.answer_llm.invoke(generation_prompt).strip()

        if "out_of_context" in answer.lower():
            return self.refusal()

        # -------- LLM 2: VALIDATE ANSWER --------
        validation_input = self.validation_prompt.format(
            context=context,
            answer=answer,
            question=question
        )

        verdict = self.validator_llm.invoke(validation_input).strip().upper()

        if verdict != "YES":
            return self.refusal()

        # -------- Determine dominant source --------
        source_counts = Counter(
            os.path.basename(d.metadata.get("source", "")) for d in docs
        )
        dominant_source = source_counts.most_common(1)[0][0]

        return {
            "answer": answer,
            "sources": [dominant_source]
        }

    def refusal(self) -> Dict:
        return {
            "answer": "Not found in documentation.",
            "sources": []
        }

    # ---------- RESPONSE ----------

    def format_response(self, result: Dict) -> str:
        if result["answer"] == "Not found in documentation.":
            return (
                "❌ Answer Not Available\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "This information is not present in the uploaded documents.\n"
            )

        response = (
            "📘 Answer\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{result['answer']}\n"
        )

        if result["sources"]:
            response += f"\n📄 Source: {result['sources'][0]}"

        return response


    # ---------- ANALYTICS ----------

    def show_analytics(self):
        if not self.chat_history:
            print("No analytics to show.")
            return

    # -------- Prepare data --------
        dates = [
            item["timestamp"].split(" ")[0]
            for item in self.chat_history
        ]
        date_counts = Counter(dates)

        feedbacks = [
            item["feedback"]
            for item in self.chat_history
        ]
        feedback_counts = Counter(feedbacks)

        total_questions = len(self.chat_history)

        # -------- Sort dates --------
        sorted_dates = sorted(date_counts.keys())
        sorted_counts = [date_counts[d] for d in sorted_dates]

        # -------- Plot 1: Questions over time --------
        plt.figure(figsize=(8, 4))
        plt.plot(sorted_dates, sorted_counts, marker="o")
        plt.title("📈 Questions Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Questions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # -------- Plot 2: Feedback distribution --------
        plt.figure(figsize=(6, 4))
        plt.bar(
            feedback_counts.keys(),
            feedback_counts.values()
        )
        plt.title("📊 Feedback Distribution")
        plt.xlabel("Feedback Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # -------- Text summary --------
        print("\n" + "=" * 40)
        print("📊 Analytics Summary")
        print("=" * 40)
        print(f"Total questions asked: {total_questions}")
        print(f"Good feedback: {feedback_counts.get('good', 0)}")
        print(f"Bad feedback: {feedback_counts.get('bad', 0)}")
        print(f"Unknown feedback: {feedback_counts.get('unknown', 0)}")
        print("=" * 40 + "\n")


    # ---------- RUN ----------

    def run(self):
        self.load_chat_history()   # ✅ LOAD OLD DATA

        self.load_documents()
        self.build_vectorstore()

        print("AI Assistant Ready (type 'exit' to quit)\n")

        while True:
            query = input("You: ").strip()

            if query.lower() == "exit":
                if input("Show analytics? (yes/no): ").lower() == "yes":
                    self.show_analytics()
                break

            intent = self.classify_intent(query)

            if intent == "METADATA":
                result = self.handle_metadata("METADATA_SUMMARY")
            else:
                result = self.handle_content_query(query)

            print("\n" + self.format_response(result))

            feedback = input("Helpful? (good/bad): ").lower()
            if feedback not in ["good", "bad"]:
                feedback = "unknown"

            self.chat_history.append({
                "query": query,
                "feedback": feedback,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        self.save_chat_history()   # ✅ SAVE ALL DATA



def main():
    DocumentationAssistant(data_path="data", model_name="mistral").run()


if __name__ == "__main__":
    main()
