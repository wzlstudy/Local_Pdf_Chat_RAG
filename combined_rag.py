import os
import socket
import json
import webbrowser
import logging
from io import StringIO
import time
import re
import markdown
from typing import List, Dict
import hashlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pdfminer.high_level import extract_text_to_fp
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 加载环境变量
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # 请在.env中设置 SERPAPI_KEY
SEARCH_ENGINE = "google"  # 可根据需要改为其他搜索引擎

# 初始化日志
logging.basicConfig(level=logging.INFO)

# 初始化嵌入模型
# 为了实现多源向量统一，请确保 PDF 和网络结果使用相同的嵌入空间
# 如果有本地部署的 "text-embedding-3-small" 模型，也可替换使用，但需保证与 PDF 向量一致
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化 ChromaDB 客户端以及共享集合
CHROMA_CLIENT = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
COLLECTION = CHROMA_CLIENT.get_or_create_collection("rag_docs")

# 设置重试 session
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('http://', HTTPAdapter(max_retries=retries))

#########################################
# SerpAPI 网络查询及向量化处理函数
#########################################
def serpapi_search(query: str, num_results: int = 5) -> list[dict]:
    """
    执行 SerpAPI 搜索，并返回解析后的结构化结果
    """
    if not SERPAPI_KEY:
        raise ValueError("未设置 SERPAPI_KEY 环境变量。请在.env文件中设置您的 API 密钥。")
    try:
        params = {
            "engine": SEARCH_ENGINE,
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "hl": "zh-CN",  # 中文界面
            "gl": "cn"
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        search_data = response.json()
        return _parse_serpapi_results(search_data)
    except Exception as e:
        logging.error(f"网络搜索失败: {str(e)}")
        return []

def _parse_serpapi_results(data: dict) -> List[Dict[str, str]]:
    """解析 SerpAPI 返回的原始数据"""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            result = {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "timestamp": item.get("date")  # 若有时间信息，可选
            }
            results.append(result)
    # 如果有知识图谱信息，也可以添加置顶（可选）
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"),
            "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"),
            "source": "knowledge_graph"
        })
    return results

def update_web_results(query: str, num_results: int = 5) -> list[dict]:
    """
    基于 SerpAPI 搜索结果，向量化并存储到 ChromaDB
    为网络结果添加元数据，ID 格式为 "web_{index}"
    """
    results = serpapi_search(query, num_results)
    if not results:
        return []
    # 删除旧的网络搜索结果
    existing_ids = COLLECTION.get()['ids']
    web_ids = [doc_id for doc_id in existing_ids if doc_id.startswith("web_")]
    if web_ids:
        COLLECTION.delete(ids=web_ids)
    docs = []
    metadatas = []
    ids = []
    for idx, res in enumerate(results):
        text = f"标题：{res.get('title', '')}\n摘要：{res.get('snippet', '')}"
        docs.append(text)
        meta = {"source": "web", "url": res.get("url", ""), "title": res.get("title")}
        meta["content_hash"] = hashlib.md5(text.encode()).hexdigest()[:8]
        metadatas.append(meta)
        ids.append(f"web_{idx}")
    embeddings = EMBED_MODEL.encode(docs)
    metadatas = [{"source": "pdf"} if not meta else meta for meta in metadatas]
    COLLECTION.add(ids=ids, embeddings=embeddings.tolist(), documents=docs, metadatas=metadatas)
    return results

#########################################
# PDF 文档处理（本地知识库更新）
#########################################
def extract_text(filepath: str) -> str:
    """使用 PDFMiner 提取 PDF 文本"""
    output = StringIO()
    with open(filepath, 'rb') as file:
        extract_text_to_fp(file, output)
    return output.getvalue()

def process_pdf(file, progress=gr.Progress()):
    """
    处理 PDF 文档：提取文本、分割文本、生成嵌入、存储到 ChromaDB 中
    每个文本块添加元数据 source: pdf
    """
    try:
        progress(0.2, desc="解析PDF...")
        text = extract_text(file.name)
        
        progress(0.4, desc="分割文本...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        
        progress(0.6, desc="生成嵌入...")
        embeddings = EMBED_MODEL.encode(chunks)
        
        progress(0.8, desc="存储向量...")
        # 删除旧的 PDF 文档数据（根据 id 前缀 pdf_）
        existing_ids = COLLECTION.get()['ids']
        pdf_ids = [doc_id for doc_id in existing_ids if doc_id.startswith("pdf_")]
        if pdf_ids:
            COLLECTION.delete(ids=pdf_ids)
        ids = [f"pdf_{i}" for i in range(len(chunks))]
        metadatas = [{"source": "pdf"} for _ in chunks]
        metadatas = [{"source": "pdf", "content_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]} for chunk in chunks]
        metadatas = [{"source": "pdf"} if not meta else meta for meta in metadatas]
        COLLECTION.add(ids=ids, embeddings=embeddings.tolist(), documents=chunks, metadatas=metadatas)
        
        progress(1.0, desc="完成!")
        return "PDF处理完成，已存储 {} 个文本块".format(len(chunks))
    except Exception as e:
        return f"处理失败: {str(e)}"

#########################################
# 新增矛盾检测函数（需放在调用前）
#########################################
def detect_conflicts(sources):
    """精准矛盾检测算法"""
    key_facts = {}
    for item in sources:
        # 从item字典中获取excerpt
        facts = extract_facts(item['excerpt'])
        for fact, value in facts.items():
            if fact in key_facts:
                if key_facts[fact] != value:
                    return True
            else:
                key_facts[fact] = value
    return False

def extract_facts(text):
    """从文本提取关键事实（示例逻辑）"""
    facts = {}
    # 提取数值型事实
    numbers = re.findall(r'\b\d{4}年|\b\d+%', text)
    if numbers:
        facts['关键数值'] = numbers
    # 提取技术术语
    if "产业图谱" in text:
        facts['技术方法'] = list(set(re.findall(r'[A-Za-z]+模型|[A-Z]{2,}算法', text)))
    return facts

#########################################
# 多源结果整合及问答生成函数
#########################################
def combined_query_answer(question, progress=gr.Progress()):
    """
    基于本地 PDF 与网络搜索结果回答用户问题（带详细进度）
    """
    try:
        # 增强版时间敏感性检测
        time_keywords = {
            "时间相关": ["最新", "今年", "当前", "最近", "刚刚", "日前", "近日", "近期"],
            "年份模式": r"\b(20\d{2}|今年|明年|去年)\b",
            "时间副词": ["最近", "目前", "现阶段", "当下", "此刻"]
        }
        
        # 使用正则表达式增强检测
        time_sensitive = (
            any(word in question for word in time_keywords["时间相关"]) or
            re.search(time_keywords["年份模式"], question) or
            any(adv in question for adv in time_keywords["时间副词"])
        )
        
        # 阶段1：初始化处理
        progress(0.05, desc="🔄 正在分析问题类型...")
        
        # 阶段2：网络搜索处理
        if time_sensitive:
            progress(0.1, desc="🌐 正在获取最新网络结果 (0/3)")
            update_steps = [
                "执行搜索请求",
                "解析搜索结果",
                "向量化存储"
            ]
            for i, step in enumerate(update_steps):
                progress(0.1 + i*0.1, desc=f"🌐 {step} ({i+1}/3)")
            results = serpapi_search(question)  # 真实耗时操作
            
            progress(0.3, desc="🌐 解析搜索结果")
            parsed = _parse_serpapi_results(results)
            
            progress(0.5, desc="🌐 向量化存储")
            update_web_results(question)  # 使用已定义的更新函数
        else:
            progress(0.4, desc="⏩ 跳过网络搜索")

        # 阶段3：向量检索
        progress_steps = [
            (0.4, "生成问题嵌入"),
            (0.5, "检索本地知识库"),
            (0.6, "排序多源结果")
        ]
        for percent, desc in progress_steps:
            progress(percent, desc=f"🔍 {desc}")

        # 阶段4：生成回答
        progress(0.7, desc="💡 正在构建提示词")
        question_embedding = EMBED_MODEL.encode([question]).tolist()
        
        progress(0.8, desc="🤖 调用大模型生成回答")
        query_results = COLLECTION.query(
            query_embeddings=question_embedding,
            n_results=10,
            include=["documents", "metadatas"]
        )
        
        # 修复文档和元数据的对齐问题
        combined_items = []
        documents = query_results.get("documents", [[]])[0]  # 防止空值
        metadatas = query_results.get("metadatas", [[]])[0]  # 防止空值
        
        # 确保文档和元数据数量一致
        max_length = max(len(documents), len(metadatas))
        for idx in range(max_length):
            doc = documents[idx] if idx < len(documents) else ""
            meta = metadatas[idx] if idx < len(metadatas) else {}
            
            # 确保元数据是字典类型
            safe_meta = meta if isinstance(meta, dict) else {}
            source_type = safe_meta.get("source", "unknown")
            
            combined_items.append({
                "type": source_type,
                "url": safe_meta.get("url", ""),
                "excerpt": (doc[:200] + "...") if doc else "",
                "title": safe_meta.get("title", "无标题")  # 新增标题字段
            })

        # 修改排序逻辑（确保网络结果优先）
        if time_sensitive:
            sorted_items = sorted(
                combined_items,
                key=lambda x: (x["type"] != "web", -len(x["excerpt"]))
            )
        else:
            sorted_items = sorted(
                combined_items,
                key=lambda x: (-len(x["excerpt"]), x["type"])
            )
        
        # 修改后的上下文构建部分
        context_parts = []
        for idx, item in enumerate(sorted_items, 1):
            if item["type"] == "web":
                context_parts.append(f"[网络结果 {idx}] {item['excerpt']} (链接: {item['url']})")
            else:
                context_parts.append(f"[本地文档 {idx}] {item['excerpt']}")
        context = "\n\n".join(context_parts)
        
        # 构建提示词模板，并提醒模型注意矛盾检测
        prompt = (
            f"请根据以下本地文档和网络搜索结果回答问题：\n{context}\n\n"
            "注意：若本地与网络结果存在矛盾，请分别标明并说明数据来源。\n\n"
            f"问题：{question}"
        )
        
        # 新增矛盾检测模块
        conflict_detected = detect_conflicts(sorted_items)
        
        if conflict_detected:
            credible_sources = [s for s in sorted_items if evaluate_source_credibility(s) > 0.7]
            if credible_sources:
                prompt += "\n注意：以下高可信来源建议优先参考：\n"
                prompt += "\n".join(f"- {s['url']}" for s in credible_sources)
        
        progress(0.95, desc="✅ 正在格式化最终答案")
        time.sleep(0.5)
        
        progress(1.0, desc="🎉 处理完成！")
        result = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120,
            headers={'Connection': 'close'}
        ).json().get("response", "未获取到有效回答")
        return format_answer(result, sorted_items)
        
    except Exception as e:
        progress(1.0, desc="❌ 遇到错误")
        return f"系统错误: {str(e)}"

def format_answer(response_text, sources):
    """生成自适应主题的HTML回答"""
    return f"""
    <style>
        :root {{
            --background: #f8f9fa;
            --text: #333;
            --code-bg: #f4f4f4;
            --border: #e0e0e0;
        }}
        
        @media (prefers-color-scheme: dark) {{
            :root {{
                --background: #2d2d2d;
                --text: #e0e0e0;
                --code-bg: #1e1e1e;
                --border: #404040;
            }}
        }}

        .answer-container {{
            padding: 20px;
            background: var(--background);
            color: var(--text);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}
        
        .source-item {{
            margin: 10px 0;
            padding: 10px;
            border-left: 3px solid var(--border);
            background: rgba(255, 255, 255, 0.05);
        }}
        
        pre {{
            background: var(--code-bg);
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px solid var(--border);
        }}
        
        a {{ color: #58a6ff; }}
    </style>
    <div class="answer-container">
        <div class="answer-content">{markdown.markdown(response_text)}</div>
        <div class="sources-section">
            <h3>📖 参考来源</h3>
            {_format_sources(sources)}
        </div>
    </div>
    """

def _format_sources(sources):
    """格式化来源信息"""
    items = []
    for idx, source in enumerate(sources, 1):
        badge_color = "#4CAF50" if source['type'] == 'web' else "#2196F3"
        items.append(f"""
        <div class="source-item">
            <div class="source-header">
                <span class="source-badge" style="background:{badge_color}">
                    {source['type']}
                </span>
                <a href="{source['url']}" target="_blank">{source.get('title', '来源'+str(idx))}</a>
            </div>
            <div class="excerpt">{source['excerpt']}</div>
        </div>
        """)
    return "\n".join(items)

#########################################
# 环境与端口检测函数
#########################################
def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0

def check_environment():
    """
    环境检查：
    1. 检查大模型服务是否加载（以 deepseek-r1:7b 为例）；
    2. 检查 Ollama 服务状态。
    """
    try:
        model_check = session.post(
            "http://localhost:11434/api/show",
            json={"name": "deepseek-r1:7b"},
            timeout=10
        )
        if model_check.status_code != 200:
            print("模型未加载！请先执行：")
            print("ollama pull deepseek-r1:7b")
            return False
        response = session.get(
            "http://localhost:11434/api/tags",
            proxies={"http": None, "https": None},
            timeout=5
        )
        if response.status_code != 200:
            print("Ollama服务异常，返回状态码:", response.status_code)
            return False
        return True
    except Exception as e:
        print("Ollama连接失败:", str(e))
        return False

#########################################
# Gradio 界面构建
#########################################
with gr.Blocks(
    css="""
    .gradio-container { max-height: 90vh !important; overflow-y: auto; }
    .left-panel, .right-panel { height: auto !important; }
    .answer-box { max-height: 60vh; overflow-y: auto; }
    .progress-tracker-container { position: static; }
    """
) as demo:
    gr.Markdown("# 🧠 多源文档问答系统")
    
    with gr.Row():
        # 左侧面板高度限制
        with gr.Column(scale=1, elem_classes="left-panel"):
            gr.Markdown("## 📂 文档处理区")
            with gr.Group():
                file_input = gr.File(label="上传PDF文档", file_types=[".pdf"])
                upload_btn = gr.Button("🚀 开始处理", variant="primary")
                upload_status = gr.Textbox(label="处理状态", interactive=False)
            
            gr.Markdown("## ❓ 提问区")
            with gr.Group():
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=4,
                    placeholder="例如：本文档的主要观点是什么？",
                    elem_id="question-input"
                )
                ask_btn = gr.Button("🔍 开始提问", variant="primary")
                status_display = gr.HTML("", elem_id="status-display")
        
        # 右侧面板高度限制
        with gr.Column(scale=3, elem_classes="right-panel"):
            gr.Markdown("## 📝 答案展示")
            progress_steps = gr.HTML()
            real_time_status = gr.HTML()  # 将状态组件移入右侧面板
            answer_output = gr.HTML(elem_classes="answer-box")
    
    # 加载提示
    # gr.HTML("""
    # <div id="loading" style="text-align:center;padding:20px;">
    #     <h3>🔄 系统初始化中，请稍候...</h3>
    # </div>
    # """)
    
    # 按钮事件绑定
    upload_btn.click(
        fn=process_pdf,
        inputs=file_input,
        outputs=upload_status
    )
    ask_btn.click(
        fn=combined_query_answer,
        inputs=question_input,
        outputs=answer_output,
        show_progress="full",  # 改为完整进度条
        api_name="ask_question"
    )

#########################################
# 主程序启动
#########################################
if __name__ == "__main__":
    if not check_environment():
        exit(1)
    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)
    if not selected_port:
        print("所有端口都被占用，请手动释放端口")
        exit(1)
    try:
        ollama_check = session.get("http://localhost:11434", timeout=5)
        if ollama_check.status_code != 200:
            print("Ollama服务未正常启动！")
            print("请先执行：ollama serve 启动服务")
            exit(1)
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port,
            server_name="0.0.0.0",
            show_error=True,
            ssl_verify=False
        )
    except Exception as e:
        print(f"启动失败: {str(e)}") 

def generate_answer_with_context(question, context):
    """改进后的回答生成逻辑"""
    # 新增矛盾检测模块
    conflict_detected = detect_conflicts(context['sources'])
    
    prompt = f"""
    根据以下信息回答问题：
    {json.dumps(context['documents'], ensure_ascii=False, indent=2)}
    问题：{question}
    
    回答要求：
    1. 区分信息来源于【本地文档】或【网络结果】
    2. 仅当明确矛盾时说明差异（当前检测状态：{'发现矛盾需说明' if conflict_detected else '未发现明确矛盾'}）
    3. 时间敏感问题优先使用最新网络结果
    """
    # ...后续生成逻辑不变...

def evaluate_source_credibility(source):
    """评估来源可信度"""
    credibility_scores = {
        "gov.cn": 0.9,
        "weixin": 0.7,
        "zhihu": 0.6
    }
    domain = re.search(r'//([^/]+)', source['url']).group(1)
    return credibility_scores.get(domain, 0.5) 