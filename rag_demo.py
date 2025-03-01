import gradio as gr
from pdfminer.high_level import extract_text_to_fp
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import json
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import socket
import webbrowser
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime
import hashlib
import re
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # 在.env中设置 SERPAPI_KEY
SEARCH_ENGINE = "google"  # 可根据需要改为其他搜索引擎

# 在文件开头添加超时设置
import requests
requests.adapters.DEFAULT_RETRIES = 3  # 增加重试次数

# 在文件开头添加环境变量设置
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化

# 在文件最开头添加代理配置
import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'  # 新增代理绕过设置

# 初始化组件
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CHROMA_CLIENT = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
COLLECTION = CHROMA_CLIENT.get_or_create_collection("rag_docs")

logging.basicConfig(level=logging.INFO)

print("Gradio version:", gr.__version__)  # 添加版本输出

# 在初始化组件后添加：
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
def serpapi_search(query: str, num_results: int = 5) -> list:
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

def _parse_serpapi_results(data: dict) -> list:
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

def update_web_results(query: str, num_results: int = 5) -> list:
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
    COLLECTION.add(ids=ids, embeddings=embeddings.tolist(), documents=docs, metadatas=metadatas)
    return results

# 检查是否配置了SERPAPI_KEY
def check_serpapi_key():
    """检查是否配置了SERPAPI_KEY"""
    return SERPAPI_KEY is not None and SERPAPI_KEY.strip() != ""

# 添加文件处理状态跟踪
class FileProcessor:
    def __init__(self):
        self.processed_files = {}  # 存储已处理文件的状态
        
    def clear_files(self):
        """清空所有文件记录"""
        self.processed_files = {}
        
    def add_file(self, file_name):
        self.processed_files[file_name] = {
            'status': '等待处理',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chunks': 0
        }
        
    def update_status(self, file_name, status, chunks=None):
        if file_name in self.processed_files:
            self.processed_files[file_name]['status'] = status
            if chunks is not None:
                self.processed_files[file_name]['chunks'] = chunks
                
    def get_file_list(self):
        return [
            f"📄 {fname} | {info['status']}"
            for fname, info in self.processed_files.items()
        ]

file_processor = FileProcessor()

#########################################
# 矛盾检测函数
#########################################
def detect_conflicts(sources):
    """精准矛盾检测算法"""
    key_facts = {}
    for item in sources:
        facts = extract_facts(item['text'] if 'text' in item else item.get('excerpt', ''))
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

def evaluate_source_credibility(source):
    """评估来源可信度"""
    credibility_scores = {
        "gov.cn": 0.9,
        "edu.cn": 0.85,
        "weixin": 0.7,
        "zhihu": 0.6,
        "baidu": 0.5
    }
    
    url = source.get('url', '')
    if not url:
        return 0.5  # 默认中等可信度
    
    domain_match = re.search(r'//([^/]+)', url)
    if not domain_match:
        return 0.5
    
    domain = domain_match.group(1)
    
    # 检查是否匹配任何已知域名
    for known_domain, score in credibility_scores.items():
        if known_domain in domain:
            return score
    
    return 0.5  # 默认中等可信度

def extract_text(filepath):
    """改进的PDF文本提取方法"""
    output = StringIO()
    with open(filepath, 'rb') as file:
        extract_text_to_fp(file, output)
    return output.getvalue()

def process_multiple_pdfs(files, progress=gr.Progress()):
    """处理多个PDF文件"""
    if not files:
        return "请选择要上传的PDF文件", []
    
    try:
        # 清空向量数据库
        progress(0.1, desc="清理历史数据...")
        try:
            # 获取所有现有文档的ID
            existing_data = COLLECTION.get()
            if existing_data and existing_data['ids']:
                COLLECTION.delete(ids=existing_data['ids'])
            logging.info("成功清理历史向量数据")
        except Exception as e:
            logging.error(f"清理历史数据时出错: {str(e)}")
            return f"清理历史数据失败: {str(e)}", []
        
        # 清空文件处理状态
        file_processor.clear_files()
        
        total_files = len(files)
        processed_results = []
        total_chunks = 0
        
        for idx, file in enumerate(files, 1):
            try:
                file_name = os.path.basename(file.name)
                progress((idx-1)/total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")
                
                # 添加文件到处理器
                file_processor.add_file(file_name)
                
                # 处理单个文件
                text = extract_text(file.name)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_text(text)
                
                if not chunks:
                    raise ValueError("文档内容为空或无法提取文本")
                
                # 生成文档唯一标识符
                doc_id = f"doc_{int(time.time())}_{idx}"
                
                # 生成嵌入
                embeddings = EMBED_MODEL.encode(chunks)
                
                # 存储向量，添加文档源信息
                ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]
                
                COLLECTION.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=chunks,
                    metadatas=metadatas
                )
                
                # 更新处理状态
                total_chunks += len(chunks)
                file_processor.update_status(file_name, "处理完成", len(chunks))
                processed_results.append(f"✅ {file_name}: 成功处理 {len(chunks)} 个文本块")
                
            except Exception as e:
                error_msg = str(e)
                logging.error(f"处理文件 {file_name} 时出错: {error_msg}")
                file_processor.update_status(file_name, f"处理失败: {error_msg}")
                processed_results.append(f"❌ {file_name}: 处理失败 - {error_msg}")
        
        # 添加总结信息
        summary = f"\n总计处理 {total_files} 个文件，{total_chunks} 个文本块"
        processed_results.append(summary)
        
        # 获取更新后的文件列表
        file_list = file_processor.get_file_list()
        
        return "\n".join(processed_results), file_list
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"整体处理过程出错: {error_msg}")
        return f"处理过程出错: {error_msg}", []

def stream_answer(question, enable_web_search=False, progress=gr.Progress()):
    """改进的流式问答处理流程，支持联网搜索"""
    try:
        # 如果启用了联网搜索，先进行网络搜索
        if enable_web_search:
            if not check_serpapi_key():
                yield "⚠️ 联网功能启用失败：未配置SERPAPI_KEY。请在.env文件中添加您的API密钥。", "错误"
                return
                
            progress(0.3, desc="正在进行网络搜索...")
            try:
                web_results = update_web_results(question)
                if not web_results:
                    progress(0.4, desc="网络搜索未返回结果，继续使用本地知识...")
            except Exception as e:
                progress(0.4, desc="网络搜索失败，使用本地知识...")
                logging.error(f"网络搜索错误: {str(e)}")
                yield f"网络搜索过程中出现错误: {str(e)}，将使用本地知识库回答", "搜索失败"
        
        progress(0.5, desc="生成问题嵌入...")
        query_embedding = EMBED_MODEL.encode([question]).tolist()
        
        progress(0.6, desc="检索相关内容...")
        results = COLLECTION.query(
            query_embeddings=query_embedding,
            n_results=5,  # 增加检索结果数量
            include=['documents', 'metadatas']
        )
        
        # 组合上下文，包含来源信息
        context_with_sources = []
        sources_for_conflict_detection = []
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            source_type = metadata.get('source', '本地文档')
            
            source_item = {
                'text': doc,
                'type': source_type
            }
            
            if source_type == 'web':
                url = metadata.get('url', '未知URL')
                title = metadata.get('title', '未知标题')
                context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
                source_item['url'] = url
                source_item['title'] = title
            else:
                source = metadata.get('source', '未知来源')
                context_with_sources.append(f"[本地文档: {source}]\n{doc}")
                source_item['source'] = source
            
            sources_for_conflict_detection.append(source_item)
        
        # 检测矛盾
        conflict_detected = detect_conflicts(sources_for_conflict_detection)
        
        # 获取可信源
        if conflict_detected:
            credible_sources = [s for s in sources_for_conflict_detection 
                               if s['type'] == 'web' and evaluate_source_credibility(s) > 0.7]
        
        context = "\n\n".join(context_with_sources)
        
        # 添加时间敏感检测
        time_sensitive = any(word in question for word in ["最新", "今年", "当前", "最近", "刚刚"])
        
        prompt_template = """基于以下{context_type}：
        {context}
        
        问题：{question}
        请用中文给出详细回答，并在回答末尾标注信息来源。{time_note}{conflict_note}"""
        
        prompt = prompt_template.format(
            context_type="本地文档和网络搜索结果" if enable_web_search else "本地文档",
            context=context,
            question=question,
            time_note="注意这是时间敏感的问题，请优先使用最新信息。" if time_sensitive and enable_web_search else "",
            conflict_note="\n注意：检测到信息源之间可能存在矛盾，请在回答中明确指出不同来源的差异。" if conflict_detected else ""
        )
        
        progress(0.7, desc="生成回答...")
        full_answer = ""
        
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": True
            },
            timeout=120,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode()).get("response", "")
                full_answer += chunk
                yield full_answer, "生成回答中..."
                
        yield full_answer, "完成!"
        
    except Exception as e:
        yield f"系统错误: {str(e)}", "遇到错误"

def query_answer(question, enable_web_search=False, progress=gr.Progress()):
    """问答处理流程，支持联网搜索"""
    try:
        logging.info(f"收到问题：{question}，联网状态：{enable_web_search}")
        
        # 如果启用了联网搜索，先进行网络搜索
        if enable_web_search:
            if not check_serpapi_key():
                return "⚠️ 联网功能启用失败：未配置SERPAPI_KEY。请在.env文件中添加您的API密钥。"
                
            progress(0.2, desc="正在进行网络搜索...")
            try:
                web_results = update_web_results(question)
                if not web_results:
                    progress(0.3, desc="网络搜索未返回结果，继续使用本地知识...")
            except Exception as e:
                progress(0.3, desc="网络搜索失败，使用本地知识...")
                logging.error(f"网络搜索错误: {str(e)}")
        
        progress(0.4, desc="生成问题嵌入...")
        # 生成问题嵌入
        query_embedding = EMBED_MODEL.encode([question]).tolist()
        
        progress(0.6, desc="检索相关内容...")
        # Chroma检索
        results = COLLECTION.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=['documents', 'metadatas']
        )
        
        # 组合上下文，包含来源信息
        context_with_sources = []
        sources_for_conflict_detection = []
        
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            source_type = metadata.get('source', '本地文档')
            
            source_item = {
                'text': doc,
                'type': source_type
            }
            
            if source_type == 'web':
                url = metadata.get('url', '未知URL')
                title = metadata.get('title', '未知标题')
                context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
                source_item['url'] = url
                source_item['title'] = title
            else:
                source = metadata.get('source', '未知来源')
                context_with_sources.append(f"[本地文档: {source}]\n{doc}")
                source_item['source'] = source
            
            sources_for_conflict_detection.append(source_item)
        
        # 检测矛盾
        conflict_detected = detect_conflicts(sources_for_conflict_detection)
        
        # 获取可信源
        if conflict_detected:
            credible_sources = [s for s in sources_for_conflict_detection 
                              if s['type'] == 'web' and evaluate_source_credibility(s) > 0.7]
        
        context = "\n\n".join(context_with_sources)
        
        # 添加时间敏感检测
        time_sensitive = any(word in question for word in ["最新", "今年", "当前", "最近", "刚刚"])
        
        prompt_template = """基于以下{context_type}：
        {context}
        
        问题：{question}
        请用中文给出详细回答，并在回答末尾标注信息来源。{time_note}{conflict_note}"""
        
        prompt = prompt_template.format(
            context_type="本地文档和网络搜索结果" if enable_web_search else "本地文档",
            context=context,
            question=question,
            time_note="注意这是时间敏感的问题，请优先使用最新信息。" if time_sensitive and enable_web_search else "",
            conflict_note="\n注意：检测到信息源之间可能存在矛盾，请在回答中明确指出不同来源的差异。" if conflict_detected else ""
        )
        
        progress(0.8, desc="生成回答...")
        # 调用Ollama
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120,  # 延长到2分钟
            headers={'Connection': 'close'}  # 添加连接头
        )
        response.raise_for_status()  # 检查HTTP状态码
        
        progress(1.0, desc="完成!")
        # 确保返回字符串并处理空值
        result = response.json()
        return str(result.get("response", "未获取到有效回答"))
    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except KeyError:
        return "响应格式异常，请检查模型服务"
    except Exception as e:
        progress(1.0, desc="遇到错误")  # 确保进度条完成
        return f"系统错误: {str(e)}"

# 修改界面布局部分
with gr.Blocks(
    title="本地RAG问答系统",
    css="""
    /* 全局主题变量 */
    :root[data-theme="light"] {
        --text-color: #2c3e50;
        --bg-color: #ffffff;
        --panel-bg: #f8f9fa;
        --border-color: #e9ecef;
        --success-color: #4CAF50;
        --error-color: #f44336;
        --primary-color: #2196F3;
        --secondary-bg: #ffffff;
        --hover-color: #e9ecef;
        --chat-user-bg: #e3f2fd;
        --chat-assistant-bg: #f5f5f5;
    }

    :root[data-theme="dark"] {
        --text-color: #e0e0e0;
        --bg-color: #1a1a1a;
        --panel-bg: #2d2d2d;
        --border-color: #404040;
        --success-color: #81c784;
        --error-color: #e57373;
        --primary-color: #64b5f6;
        --secondary-bg: #2d2d2d;
        --hover-color: #404040;
        --chat-user-bg: #1e3a5f;
        --chat-assistant-bg: #2d2d2d;
    }

    /* 全局样式 */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .gradio-container {
        max-width: 1200px !important;
        color: var(--text-color);
        background-color: var(--bg-color);
    }

    /* 主题切换按钮 */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        background: var(--panel-bg);
        color: var(--text-color);
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .theme-toggle:hover {
        background: var(--hover-color);
    }

    /* 面板样式 */
    .left-panel {
        padding-right: 20px;
        border-right: 1px solid var(--border-color);
        background: var(--bg-color);
    }

    .right-panel {
        height: 100vh;
        background: var(--bg-color);
    }

    /* 文件列表样式 */
    .file-list {
        margin-top: 10px;
        padding: 12px;
        background: var(--panel-bg);
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 答案框样式 */
    .answer-box {
        min-height: 500px !important;
        background: var(--panel-bg);
        border-radius: 8px;
        padding: 16px;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 输入框样式 */
    textarea {
        background: var(--panel-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 14px !important;
    }

    /* 按钮样式 */
    button.primary {
        background: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    button.primary:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* 标题和文本样式 */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }

    .footer-note {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 13px;
        margin-top: 12px;
    }

    /* 加载和进度样式 */
    #loading, .progress-text {
        color: var(--text-color);
    }

    /* 聊天记录样式 */
    .chat-container {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 16px;
        max-height: 600px;
        overflow-y: auto;
        background: var(--bg-color);
    }

    .chat-message {
        padding: 12px 16px;
        margin: 8px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
    }

    .chat-message.user {
        background: var(--chat-user-bg);
        margin-left: 32px;
        border-top-right-radius: 4px;
    }

    .chat-message.assistant {
        background: var(--chat-assistant-bg);
        margin-right: 32px;
        border-top-left-radius: 4px;
    }

    .chat-message .timestamp {
        font-size: 12px;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 4px;
    }

    .chat-message .content {
        white-space: pre-wrap;
    }

    /* 按钮组样式 */
    .button-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }

    .clear-button {
        background: var(--error-color) !important;
    }

    /* API配置提示样式 */
    .api-info {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
    }
    """
) as demo:
    gr.Markdown("# 🧠 智能文档问答系统")
    
    with gr.Row():
        # 左侧操作面板
        with gr.Column(scale=1, elem_classes="left-panel"):
            gr.Markdown("## 📂 文档处理区")
            with gr.Group():
                file_input = gr.File(
                    label="上传PDF文档",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("🚀 开始处理", variant="primary")
                upload_status = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    lines=2
                )
                file_list = gr.Textbox(
                    label="已处理文件",
                    interactive=False,
                    lines=3,
                    elem_classes="file-list"
                )

        # 右侧对话区
        with gr.Column(scale=3, elem_classes="right-panel"):
            gr.Markdown("## 📝 对话记录")
            
            # 对话记录显示区
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500,
                elem_classes="chat-container",
                show_label=False
            )
            
            # 问题输入区
            with gr.Group():
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=3,
                    placeholder="请输入您的问题...",
                    elem_id="question-input"
                )
                with gr.Row():
                    # 添加联网开关
                    web_search_checkbox = gr.Checkbox(
                        label="启用联网搜索", 
                        value=False,
                        info="打开后将同时搜索网络内容（需配置SERPAPI_KEY）"
                    )
                    
                with gr.Row():
                    ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button", scale=1)
                status_display = gr.HTML("", elem_id="status-display")
            
            # 添加API配置提示信息
            api_info = gr.HTML(
                """
                <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
                    <p>📢 <strong>联网功能说明：</strong></p>
                    <p>1. 需要在项目目录下的<code>.env</code>文件中配置<code>SERPAPI_KEY=您的密钥</code></p>
                    <p>2. 可以在<a href="https://serpapi.com/" target="_blank">SerpAPI官网</a>获取免费密钥</p>
                </div>
                """
            )
            
            gr.Markdown("""
            <div class="footer-note">
                *回答生成可能需要1-2分钟，请耐心等待<br>
                *支持多轮对话，可基于前文继续提问
            </div>
            """)

    # 调整后的加载提示
    gr.HTML("""
    <div id="loading" style="text-align:center;padding:20px;">
        <h3>🔄 系统初始化中，请稍候...</h3>
    </div>
    """)

    # 进度显示组件调整到左侧面板下方
    with gr.Row(visible=False) as progress_row:
        gr.HTML("""
        <div class="progress-text">
            <span>当前进度：</span>
            <span id="current-step" style="color: #2b6de3;">初始化...</span>
            <span id="progress-percent" style="margin-left:15px;color: #e32b2b;">0%</span>
        </div>
        """)

    def clear_chat_history():
        return [], ""  # 清空对话历史和输入框

    # 修改问答处理函数
    def process_chat(question, history, enable_web_search):
        if not question:
            return history, ""
        
        history = history or []
        history.append([question, None])
        
        try:
            for response, status in stream_answer(question, enable_web_search):
                if status != "遇到错误":
                    history[-1][1] = response
                    yield history, ""
                else:
                    history[-1][1] = f"❌ {response}"
                    yield history, ""
        except Exception as e:
            history[-1][1] = f"❌ 系统错误: {str(e)}"
            yield history, ""

    # 检查SERPAPI配置状态并更新提示信息
    def update_api_info(enable_web_search):
        if not enable_web_search:
            return """
            <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
                <p>📢 <strong>联网功能已关闭</strong></p>
                <p>开启联网功能可获取最新网络信息</p>
            </div>
            """
        
        if check_serpapi_key():
            return """
            <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);border-left:4px solid #4CAF50;">
                <p>✅ <strong>联网功能已启用</strong></p>
                <p>SERPAPI_KEY已配置，可以进行网络搜索</p>
            </div>
            """
        else:
            return """
            <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);border-left:4px solid #f44336;">
                <p>❌ <strong>联网功能启用失败</strong></p>
                <p>未检测到SERPAPI_KEY配置，请在项目目录下的<code>.env</code>文件中添加：</p>
                <pre style="background:var(--code-bg);padding:5px;border-radius:3px;">SERPAPI_KEY=您的API密钥</pre>
                <p>可以在<a href="https://serpapi.com/" target="_blank">SerpAPI官网</a>获取免费密钥</p>
            </div>
            """

    # 更新事件处理
    web_search_checkbox.change(
        fn=update_api_info,
        inputs=web_search_checkbox, 
        outputs=api_info
    )
    
    ask_btn.click(
        fn=process_chat,
        inputs=[question_input, chatbot, web_search_checkbox],
        outputs=[chatbot, question_input],
        show_progress=False
    ).then(
        fn=lambda: "",
        outputs=status_display
    )

    clear_btn.click(
        fn=clear_chat_history,
        outputs=[chatbot, question_input],
        show_progress=False
    )

    # 添加文件处理按钮事件
    upload_btn.click(
        fn=process_multiple_pdfs,
        inputs=file_input,
        outputs=[upload_status, file_list]
    )

# 修改JavaScript注入部分
demo._js = """
function gradioApp() {
    // 设置默认主题为暗色
    document.documentElement.setAttribute('data-theme', 'dark');
    
    const observer = new MutationObserver((mutations) => {
        document.getElementById("loading").style.display = "none";
        const progress = document.querySelector('.progress-text');
        if (progress) {
            const percent = document.querySelector('.progress > div')?.innerText || '';
            const step = document.querySelector('.progress-description')?.innerText || '';
            document.getElementById('current-step').innerText = step;
            document.getElementById('progress-percent').innerText = percent;
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});
}

function toggleTheme() {
    const root = document.documentElement;
    const currentTheme = root.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', newTheme);
}

// 初始化主题
document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.setAttribute('data-theme', 'dark');
});
"""

# 修改端口检查函数
def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0  # 更可靠的检测方式

def check_environment():
    """环境依赖检查"""
    try:
        # 添加模型存在性检查
        model_check = session.post(
            "http://localhost:11434/api/show",
            json={"name": "deepseek-r1:7b"},
            timeout=10
        )
        if model_check.status_code != 200:
            print("模型未加载！请先执行：")
            print("ollama pull deepseek-r1:7b")
            return False
            
        # 原有检查保持不变...
        response = session.get(
            "http://localhost:11434/api/tags",
            proxies={"http": None, "https": None},  # 禁用代理
            timeout=5
        )
        if response.status_code != 200:
            print("Ollama服务异常，返回状态码:", response.status_code)
            return False
        return True
    except Exception as e:
        print("Ollama连接失败:", str(e))
        return False

# 方案2：禁用浏览器缓存（添加meta标签）
gr.HTML("""
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
""")

# 恢复主程序启动部分
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

