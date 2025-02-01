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

def extract_text(filepath):
    """改进的PDF文本提取方法"""
    output = StringIO()
    with open(filepath, 'rb') as file:
        extract_text_to_fp(file, output)
    return output.getvalue()

def process_pdf(file, progress=gr.Progress()):
    """PDF处理全流程"""
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
        # 清空现有数据的正确方式
        existing_ids = COLLECTION.get()['ids']
        if existing_ids:
            COLLECTION.delete(ids=existing_ids)
        # 存入新数据
        ids = [str(i) for i in range(len(chunks))]
        COLLECTION.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks
        )
        
        progress(1.0, desc="完成!")
        return "PDF处理完成，已存储 {} 个文本块".format(len(chunks))
    except Exception as e:
        return f"处理失败: {str(e)}"

def stream_answer(question, progress=gr.Progress()):
    """流式问答处理流程"""
    try:
        progress(0.3, desc="生成问题嵌入...")
        query_embedding = EMBED_MODEL.encode([question]).tolist()
        
        progress(0.5, desc="检索相关内容...")
        results = COLLECTION.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        context = "\n".join(results['documents'][0])
        prompt = f"""基于以下上下文：
        {context}
        
        问题：{question}
        请用中文给出详细回答："""
        
        progress(0.7, desc="生成回答...")
        full_answer = ""
        
        # 流式请求
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": True  # 启用流式
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

def query_answer(question, progress=gr.Progress()):
    """问答处理流程"""
    try:
        logging.info(f"收到问题：{question}")
        progress(0.3, desc="生成问题嵌入...")
        # 生成问题嵌入
        query_embedding = EMBED_MODEL.encode([question]).tolist()
        
        progress(0.5, desc="检索相关内容...")
        # Chroma检索
        results = COLLECTION.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        # 构建提示词
        context = "\n".join(results['documents'][0])
        prompt = f"""基于以下上下文：
        {context}
        
        问题：{question}
        请用中文给出详细回答："""
        
        progress(0.7, desc="生成回答...")
        # 调用Ollama
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",
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

# 构建Gradio界面
with gr.Blocks(
    title="本地RAG问答系统",
    css="""
    .gradio-container {max-width: 800px !important}
    .progress-text {color: #666; font-size: 0.9em;}
    """
) as demo:
    gr.Markdown("# 本地RAG问答系统")
    gr.Markdown("""
    ## 使用说明
    1. 上传PDF文档进行处理
    2. 等待处理完成后即可开始提问
    3. 系统会从文档中检索相关内容并生成答案
    """)
    
    # 在gr.Markdown后添加加载提示
    gr.HTML("""
    <div id="loading" style="text-align:center;padding:20px;">
        <h3>系统初始化中，请稍候...</h3>
    </div>
    """)
    
    # 新增统一进度显示组件
    with gr.Row(visible=False) as progress_row:
        gr.HTML("""
        <div class="progress-text">
            <span id="current-step">初始化...</span>
            <span id="progress-percent">0%</span>
        </div>
        """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="上传PDF文档")
            upload_btn = gr.Button("处理文档", variant="primary")
            upload_status = gr.Textbox(label="处理状态", interactive=False)
        
        with gr.Column():
            question_input = gr.Textbox(label="输入问题", lines=3, placeholder="请输入您的问题...")
            ask_btn = gr.Button("提问", variant="primary")
            # 修改为逐步输出的文本框
            answer_output = gr.Textbox(label="答案", interactive=False, lines=10, 
                                     autoscroll=True, elem_classes="stream-output")
            # 新增实时状态显示
            status_display = gr.HTML("")
    
    upload_btn.click(
        fn=process_pdf,
        inputs=file_input,
        outputs=upload_status
    )
    
    ask_btn.click(
        fn=stream_answer,
        inputs=question_input,
        outputs=[answer_output, status_display],
        show_progress="hidden"  # 使用自定义进度显示
    )

    # 修改JavaScript注入部分为兼容写法
    demo._js = """
    function gradioApp() {
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
            json={"name": "deepseek-r1:1.5b"},
            timeout=10
        )
        if model_check.status_code != 200:
            print("模型未加载！请先执行：")
            print("ollama pull deepseek-r1:1.5b")
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

