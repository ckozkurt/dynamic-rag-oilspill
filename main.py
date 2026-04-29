import torch, os, time, json, psutil
import pandas as pd
from datasets import Dataset
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig

# --- PERFORMANS TAKİBİNİ BAŞLAT ---
GENEL_BASLANGIC_ZAMANI = time.time()

def get_gpu_memory():
    """GPU (VRAM) kullanımını GB cinsinden döndürür."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0

# =====================================================================
# 1. TEKRAR ÜRETİLEBİLİRLİK (REPRODUCIBILITY) AYARLARI VE LOGLARI
# =====================================================================
EXPERIMENT_CONFIG = {
    "model_name": "gemma3:27b",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens_predict": 16384,  
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "retrieval_k": 3
}

print("🚀 Sistem Başlatılıyor: Hatasız main4.py Altyapısı + Detaylı Loglama")

# --- VERİ VE VEKTÖR SİSTEMİ ---
loader = PyPDFDirectoryLoader("data/") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=EXPERIMENT_CONFIG["chunk_size"], 
    chunk_overlap=EXPERIMENT_CONFIG["chunk_overlap"]
)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EXPERIMENT_CONFIG["embedding_model"])
vectorstore = FAISS.from_documents(splits, embeddings)

# --- LLM TANIMLARI ---
llm = Ollama(
    model=EXPERIMENT_CONFIG["model_name"], 
    timeout=1200.0, # 126 PDF'in yoğun işlemleri için 20 dakika süre
    temperature=EXPERIMENT_CONFIG["temperature"],
    top_p=EXPERIMENT_CONFIG["top_p"],
    top_k=EXPERIMENT_CONFIG["top_k"],
    num_predict=EXPERIMENT_CONFIG["max_tokens_predict"]
)

queries = [
    "What novel ML algorithms can better distinguish oil spills from look-alikes in SAR imagery?",
    "How can we enhance the accuracy of volume and thickness estimations for marine oil spills by utilizing multi-band analysis from multi-spectral and hyperspectral optical remote sensing data?",
    "Can you explain how fusing Automatic Identification System (AIS) navigational data with satellite-based SAR imagery could optimize the real-time detection of ships conducting illegal oil discharges in the open ocean?",
    "What are the most effective modeling and filtering methods to minimize the misleading effects of environmental factors, such as sun glint, wind speed, and water quality, in optical remote sensing images of oil spills?",
    "Could you analyze the impact on simulation accuracy when integrating time-series remote sensing data into two-dimensional hydrodynamic models to predict the drift trajectories of oil spills caused by ocean currents and wind?",
    "How can we expand the capability of Thermal Infrared (IR) systems to monitor accident-induced oil spills at night, leveraging the differences in heat capacity and thermal inertia between oil and clean seawater?",
    "What would be the best approach to integrate ultraviolet (UV) induced fluorescence sensors into stationary buoy networks or vessels to identify specific oil types (e.g., heavy diesel vs. crude oil) based on their PAH content?",
    "What multi-sensor fusion techniques (e.g., SAR and Optical) are best for monitoring long-term oil spill damage to coastal ecosystems?",
    "How can passive UV imaging systems, which are highly effective at detecting oil sheens thinner than 300 µm, be integrated with radar-based systems to achieve ultra-precise oil spill mapping?",
    "What are the best strategies for using UAV-based monitoring systems to overcome the limitations of satellite revisiting periods, ensuring continuous tracking of strip-shaped illegal operational discharges from moving vessels?",
    "How can signal processing remove hydrodynamic noise from SAR data during stormy conditions to reveal masked oil spills?"
]

final_data = []
detailed_research_logs = []

print("\n📝 Cevaplar Üretiliyor ve Performans Logları Toplanıyor...")

for i, q in enumerate(queries):
    print(f"🔄 Soru {i+1}/11 işleniyor...")
    
    # =====================================================================
    # 2. RAG RETRIEVAL KALİTESİ (Geri Çağırma Metrikleri)
    # =====================================================================
    t_retrieval_start = time.time()
    
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(q, k=EXPERIMENT_CONFIG["retrieval_k"])
    
    retrieval_latency = time.time() - t_retrieval_start
    
    contexts = []
    chunk_logs = []
    context_window_usage = 0
    
    for doc, score in retrieved_docs_with_scores:
        contexts.append(doc.page_content)
        context_window_usage += len(doc.page_content)
        chunk_logs.append({
            "score_l2_distance": float(score),
            "metadata": doc.metadata,
            "chunk_length": len(doc.page_content)
        })

    prompt_template = f"""Context: {' '.join(contexts)}
Question: {q}
Instruction: Please answer the question above directly and concisely. Limit your response to a maximum of 3 or 4 sentences. Do not use long bulleted lists.
"""
    context_window_usage += len(prompt_template)
    
    # =====================================================================
    # 3. PERFORMANS VE KAYNAK KULLANIMI (Verimlilik Analizi)
    # =====================================================================
    t_gen_start = time.time()
    ttft = None
    token_count = 0
    response_text = ""
    
    for chunk in llm.stream(prompt_template):
        if ttft is None:
            ttft = time.time() - t_gen_start
        token_count += 1
        response_text += chunk
        
    t_gen_end = time.time()
    total_inference_time = t_gen_end - t_gen_start
    tps = token_count / total_inference_time if total_inference_time > 0 else 0

    final_data.append({
        "question": q,
        "answer": str(response_text).strip(),
        "contexts": contexts
    })
    
    detailed_research_logs.append({
        "question_id": i + 1,
        "question": q,
        "retrieval_metrics": {
            "retrieval_latency_sec": round(retrieval_latency, 4),
            "context_window_chars": context_window_usage,
            "retrieved_chunks": chunk_logs
        },
        "performance_metrics": {
            "time_to_first_token_sec": round(ttft, 3) if ttft else 0,
            "total_inference_time_sec": round(total_inference_time, 3),
            "tokens_per_second_tps": round(tps, 2),
            "estimated_token_count": token_count
        },
        "hardware_metrics": {
            "ram_usage_percent": psutil.virtual_memory().percent,
            "gpu_vram_gb": round(get_gpu_memory(), 2)
        }
    })

print("\n💾 Araştırma Verileri Diske Kaydediliyor...")

# Dosyaları okuma sırasına göre numaralandırıyoruz
with open("1_EXPERIMENT_CONFIG.json", "w", encoding="utf-8") as f:
    json.dump(EXPERIMENT_CONFIG, f, ensure_ascii=False, indent=4)

with open("2_RESEARCH_LOGS.json", "w", encoding="utf-8") as f:
    json.dump(detailed_research_logs, f, ensure_ascii=False, indent=4)

with open("3_QA_READABLE_LIST.txt", "w", encoding="utf-8") as f:
    for i, item in enumerate(final_data):
        f.write(f"Soru {i+1}: {item['question']}\n\nCevap: {item['answer']}\n")
        f.write("="*60 + "\n\n")

with open("4_RAGAS_RAW_RESULTS.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

# =====================================================================
# 4. RAGAs KALİTE DEĞERLENDİRMESİ
# =====================================================================
try:
    print("📊 Ragas Analizi Başlıyor (main4.py Kararlılığıyla)...")
    dataset = Dataset.from_list(final_data)
    
    # RAGAs değerlendirmesini boğan aşırı yükleri önlemek için max_workers=1
    custom_config = RunConfig(timeout=1200, max_workers=1)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm, 
        embeddings=embeddings,
        run_config=custom_config,
        raise_exceptions=False
    )
    
    clean_df = result.to_pandas()[['question', 'faithfulness', 'answer_relevancy']].copy()
    clean_df['faithfulness'] = clean_df['faithfulness'].round(3)
    clean_df['answer_relevancy'] = clean_df['answer_relevancy'].round(3)
    clean_df.to_csv("5_FINAL_ACADEMIC_REPORT.csv", index=False, encoding='utf-8-sig')
    
    print("\n🏆 ANALİZ BAŞARIYLA TAMAMLANDI!")

except Exception as e:
    print(f"❌ RAGAs sırasında kritik hata: {e}")

# =====================================================================
# 5. SİSTEM PERFORMANS METRİKLERİ
# =====================================================================
genel_sure_sn = time.time() - GENEL_BASLANGIC_ZAMANI
genel_sure_dk = genel_sure_sn / 60
ram_kullanimi = psutil.virtual_memory().percent
gpu_kullanimi = get_gpu_memory()

performans_raporu = f"""
=========================================
SİSTEM PERFORMANS RAPORU / 
=========================================
Toplam İşlem Süresi : {genel_sure_dk:.2f} Dakika ({genel_sure_sn:.1f} Saniye)
Sistem RAM Kullanımı: % {ram_kullanimi}
Maksimum GPU (VRAM) : {gpu_kullanimi:.2f} GB

=========================================
"""

print(performans_raporu)
with open("6_SISTEM_PERFORMANSI.txt", "w", encoding="utf-8") as f:
    f.write(performans_raporu)

print("✅ Tüm işlemler bitti! Raporlar 1'den 6'ya kadar proje klasörüne başarıyla kaydedildi.")
