import os
import openai

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# ===== Chroma 新版用法 =====
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, Embeddings


# ============【1】自訂 EmbeddingFunction，避免調用到 openai.OpenAI(...) ============ #
class MyCustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Chroma 會透過此 class 的 __call__ 方法來產生 Embedding。
    我們直接用 openai.Embedding.create() 生成向量，避免使用 OpenAIEmbeddingFunction 內部的舊調用。
    """
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model_name = model_name

    def __call__(self, texts: Documents) -> Embeddings:
        """
        :param texts: List of strings
        :return: List of embedding vectors (List of Lists of floats)
        """
        openai.api_key = self.api_key
        embeddings = []
        for t in texts:
            response = openai.Embedding.create(input=t, model=self.model_name)
            emb_vec = response['data'][0]['embedding']
            embeddings.append(emb_vec)
        return embeddings


# ============【2】Flask + LINE Bot 初始化 ============ #
app = Flask(__name__)

# 從環境變數讀取
openai.api_key = os.getenv("OPENAI_API_KEY")
line_bot_api = LineBotApi(os.getenv("CHANNELeAKXjsayP5rGy5pfk6QFoIFbZtJBTFkeu41eDbtWCaGjy8+0msFjWi5Uels5FpQRn0c3EfeHYmD3UT6LXgRrrjoaUlSvKLqBqOFqfawESVpoEU6765MBWl2tkpn0j2l2dja7yzxO+vf66ARnw3GRfgdB04t89/1O/w1cDnyilFU=_ACCESS_TOKEN"))
handler1 = WebhookHandler(os.getenv("CHANNEL_SECRET"))

# ============【3】連線到 Chroma - PersistentClient ============ #
# 注意：如果你想要「臨時」不存檔，可換成 `chromadb.EphemeralClient()`
client = chromadb.PersistentClient(path="./chroma_db")

# 建立自訂 embedding function
my_embedding_fn = MyCustomEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)

# 取得（或建立）Collection
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=my_embedding_fn
)

# ============【4】impersonated_role：你的系統角色設定 (System Prompt) ============ #
position = '資訊系統操作與專業客服'
impersonated_role = f"""
    我現在要你當心統科技有限公司的{position}。
    記住本公司的產品有直播錄影系統、PBL互動教學教室系統、各類專業行雲端學習平台等等。
    當你不知道答案的時候，可以請使用者參閱公司官網 https://www.zsystem.com.tw/, 
    或請留下使用者資訊(email,聯絡電話)並整理使用者問題後請使用者確認我們將請專人與他聯繫。
    另外我們有一個很獨特的雲端學習平台,它只針對保經代公司與保險業務人員的學習平台,
    提供產險,壽險,相關法令課程與其他專業銷售資格的課程內容,
    業務人員在這個平台學習與測驗後,他們的學習歷程將會被提供到保經商公會與產壽險公司進行通報,
    能與公會資料結合自動化計算出業務人員的年度受訓區間,
    並指派應該上哪些課程,簡化了行政端的行政流程與整理的時間。
    你是一個具有專業保險行業知識背景的客服，了解所有產險與壽險教訓訓練知識，
    熟知所有法令規章，且同時也具備專業資訊系統操作知識。
    為了提供更完整的服務，你熟悉產險1~5年及壽險1~5年的專業課程內容...
    ...
    （此處省略部分重複內容）
    ...
    只能使用繁體中文，並全程只能使用台灣慣用用語！
    你不是一個AI模型！
"""


# ============【5】LINE Webhook 路由 & 處理函式 ============ #
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler1.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler1.add(MessageEvent, message=TextMessage)
def handle_message(event):
     # 在這裡印出 event 內容，觀察是否有正確接收
    print("Received event:", event)
    user_text = event.message.text

    # (A) 先在 Chroma 中檢索與 user_text 相似的段落
    #     你可以改 n_results=5 或其他數字
    results = collection.query(query_texts=[user_text], n_results=3)

    # results["documents"] 通常是 List of List
    retrieved_docs = results["documents"][0] if results and len(results["documents"]) > 0 else []
    reference_text = "\n".join(retrieved_docs)

    # (B) 組合要給 GPT-4 的 Prompt
    additional_context = f"以下為與使用者問題相似的檔案內容供你參考：\n{reference_text}"
    messages = [
        {"role": "system", "content": impersonated_role},
        {"role": "system", "content": additional_context},
        {"role": "user", "content": user_text}
    ]

    # (C) 呼叫 OpenAI ChatCompletion
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.5
        )
        assistant_reply = response["choices"][0]["message"]["content"].strip()
        ret_msg = f"Zsystem智能AI助理：\n\n{assistant_reply}"

    except Exception as e:
        ret_msg = f"發生錯誤: {str(e)}"

    # (D) 回覆使用者
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=ret_msg))


# ============【6】主程式入口 ============ #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
