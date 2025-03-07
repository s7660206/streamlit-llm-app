import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 環境変数からAPIキーを取得
openai_api_key = os.getenv("OPENAI_API_KEY")

# LLMの初期化（temperature=0で安定した回答）
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# 各専門家のツール関数を定義
def get_programming_expert_advice(query: str) -> str:
    """プログラミングに関する質問に回答するツール関数"""
    system_template = """
    あなたは優秀なプログラミング専門家です。技術的な質問に対して詳細かつ正確に回答してください。
    """
    messages = [
        SystemMessage(content=system_template),
        HumanMessage(content=query)
    ]
    response = llm(messages)
    return response.content

programming_expert_tool = Tool.from_function(
    func=get_programming_expert_advice,
    name="プログラミング専門家",
    description="プログラミングに関する質問に日本語で回答します。"
)

def get_medical_expert_advice(query: str) -> str:
    """医療に関する質問に回答するツール関数"""
    system_template = """
    あなたは優秀な医療専門家です。医療に関する質問に対して専門的かつ正確に回答してください。
    """
    messages = [
        SystemMessage(content=system_template),
        HumanMessage(content=query)
    ]
    response = llm(messages)
    return response.content

medical_expert_tool = Tool.from_function(
    func=get_medical_expert_advice,
    name="医療専門家",
    description="医療に関する質問に日本語で回答します。"
)

# Toolオブジェクトのリストを作成
tools = [
    programming_expert_tool,
    medical_expert_tool
]

# Agentsの初期化
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def get_llm_response(input_text: str, expert_choice: str) -> str:
    """
    入力テキストと専門家の選択値に応じた問い合わせを、
    LangChain AgentのToolを利用してLLMからの回答を取得する関数です。
    """
    # ユーザーの選択に応じ、ツール名を明示する指示をプロンプトに追加
    if expert_choice == "A":
        prompt = f"Use the ProgrammingExpert tool. {input_text}"
    elif expert_choice == "B":
        prompt = f"Use the MedicalExpert tool. {input_text}"
    else:
        prompt = input_text

    # Agent経由で問い合わせし、結果を返す
    response = agent.run(prompt)
    return response

# アプリの概要・操作方法の説明
st.title("専門家対話アプリ")
st.markdown("""
このアプリは、入力フォームに記入したテキストをLangChain Agent経由でLLMにプロンプトとして渡し、その回答結果を画面上に表示します。  
また、ラジオボタンで以下の専門家の中から振る舞いを選択できます。

- **A: プログラミング専門家**  
  プログラミングや技術的な質問に対して専門的に回答します。
- **B: 医療専門家**  
  医療や健康に関する質問に対して専門的に回答します。

""")

# 専門家の選択（ラジオボタン）
expert_choice = st.radio(
    "専門家の種類を選択してください：",
    options=["A", "B"],
    index=0,
    format_func=lambda x: "プログラミング専門家" if x == "A" else "医療専門家"
)

# 入力フォーム
user_input = st.text_input("質問を入力してください:")

# 送信ボタン押下時に、入力テキストと選択値をもとにLLMへ問い合わせ、回答結果を表示
if st.button("送信"):
    if user_input.strip() == "":
        st.error("質問を入力してください。")
    else:
        with st.spinner("LLMに問い合わせ中です..."):
            answer = get_llm_response(user_input, expert_choice)
        st.markdown("### 回答")
        st.write(answer)
