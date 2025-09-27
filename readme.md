# SQCLIR Hindi - FIRE 2025 (Monolingual Track)

This repository contains our retrieval system implementation and relevance judgment (TRC) files for the **Spoken Query Cross-Language Information Retrieval (SQCLIR) task at FIRE 2025**.  
We focus on the **Hindi monolingual track**, where both spoken queries and documents are in Hindi.  

👉 Official Task Website: [SQCLIR 2025](https://sites.google.com/view/sqclir-2025)

---

## 📌 Contents
- `Retirieval files/` – Retrieval code (BM25, IndicBERT+FAISS).  
- `trec files/` – Topic relevance judgment files (for evaluation).   

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Pranesh4950/sqclir-hindi-fire2025.git
cd sqclir-hindi-fire2025
pip install -r requirements.txt
