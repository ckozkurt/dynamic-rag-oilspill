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

# --- PERFORMANS TAKİBİNİ BAŞLAT / START PERFORMANCE TRACKING ---
GENEL_BASLANGIC_ZAMANI = time.time() # GENERAL_START_TIME

def get_gpu_memory():
    """GPU (VRAM) kullanımını GB cinsinden döndürür.""" # Returns GPU (VRAM) usage in GB.
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0

# =====================================================================
# 1. TEKRAR ÜRETİLEBİLİRLİK (REPRODUCIBILITY) AYARLARI VE LOGLARI / REPRODUCIBILITY SETTINGS AND LOGS
# =====================================================================
EXPERIMENT_CONFIG = {
    "model_name": "gemma3:27b",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens_predict": 16384,   # <-- 126 PDF'te Faithfulness metriği yarım kalmasın diye 4096 yapıldı
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "retrieval_k": 3
}

print("🚀 Sistem Başlatılıyor / 🚀 System Starting Up") 

# --- VERİ VE VEKTÖR SİSTEMİ / DATA AND VECTOR SYSTEM---
loader = PyPDFDirectoryLoader("data/") # Gerçek testte burayı "data/" yapmayı unutma
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=EXPERIMENT_CONFIG["chunk_size"], 
    chunk_overlap=EXPERIMENT_CONFIG["chunk_overlap"]
)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EXPERIMENT_CONFIG["embedding_model"])
vectorstore = FAISS.from_documents(splits, embeddings)

# --- LLM TANIMLARI / LLM DEFINITIONS---
# main4.py'de hatasız çalışan o sade LLM yapısını kullanıyoruz. 
# Hiçbir ekstra Wrapper veya ChatOllama kalkanı YOK.
llm = Ollama(
    model=EXPERIMENT_CONFIG["model_name"], 
    timeout=1200.0, 
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

print("\n📝 Cevaplar Üretiliyor ve Performans Logları Toplanıyor...  \n📝 Responses are being generated and performance logs are being collected...")

for i, q in enumerate(queries):
    print(f"🔄 Soru {i+1}/11 işleniyor... \n🔄 Question {i+1}/11 is being processed...")
    
    # =====================================================================
    # 2. RAG RETRIEVAL KALİTESİ (Geri Çağırma Metrikleri) / RAG RETRIEVAL QUALITY (Recall Metrics)
    # =====================================================================
    t_retrieval_start = time.time()
    
    # Skorları ve belgeleri çek (FAISS varsayılan olarak L2 Distance döndürür) / Pull scores and documents (FAISS returns L2 Distance by default)
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
    # 3. PERFORMANS VE KAYNAK KULLANIMI (Verimlilik Analizi) / PERFORMANCE AND RESOURCE UTILIZATION (Efficiency Analysis)
    # =====================================================================
    t_gen_start = time.time()
    ttft = None
    token_count = 0
    response_text = ""
    
    # TTFT ve TPS ölçümü için LLM Stream kullanıyoruz / We use LLM Stream for TTFT and TPS measurement.
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

print("\n💾 Araştırma Verileri Diske Kaydediliyor...  \n💾 Research data is being saved to disk...")

# Dosyaları okuma sırasına göre numaralandırıyoruz # We number the files according to the order in which they are read.
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
# 4. RAGAS KALİTE DEĞERLENDİRMESİ
# =====================================================================
try:
    print("📊 Ragas Analizi Başlıyor / 📊 Ragas Analysis Begins")
    dataset = Dataset.from_list(final_data)
    
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
    
    print("\n🏆 ANALİZ BAŞARIYLA TAMAMLANDI! / 🏆 ANALYSIS SUCCESSFULLY COMPLETED!")

except Exception as e:
    print(f"❌ Ragas sırasında kritik hata: {e}  / Critical error during RAGAs: {e}")

# =====================================================================
# 5. SİSTEM PERFORMANS METRİKLERİ / SYSTEM PERFORMANCE METRICS
# =====================================================================
genel_sure_sn = time.time() - GENEL_BASLANGIC_ZAMANI # general_time_sec = time.time() - GENERAL_START_TIME
genel_sure_dk = genel_sure_sn / 60 # general_time_min = general_time_sec / 60
ram_kullanimi = psutil.virtual_memory().percent # ram_usage
gpu_kullanimi = get_gpu_memory() # gpu_usage

performans_raporu = f""" # performance_report
=========================================
SİSTEM PERFORMANS RAPORU / SYSTEM PERFORMANCE REPORT
=========================================
Toplam İşlem Süresi / Total Processing Time : {genel_sure_dk:.2f} Dakika ({genel_sure_sn:.1f} Saniye) 
Sistem RAM Kullanımı / System RAM Usage : % {ram_kullanimi}
Maximum GPU (VRAM) : {gpu_kullanimi:.2f} GB

=========================================
"""

print(performans_raporu)
with open("6_SISTEM_PERFORMANSI.txt", "w", encoding="utf-8") as f:
    f.write(performans_raporu)

print("✅ Tüm işlemler bitti! Raporlar 1'den 6'ya kadar proje klasörüne başarıyla kaydedildi. \n ✅ All operations completed! Reports 1 through 6 have been successfully saved to the project folder.")
