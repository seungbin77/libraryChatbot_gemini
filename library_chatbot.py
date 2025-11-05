import os
import sys
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

# ✅ pysqlite3 패치 (ChromaDB용)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ✅ LangChain 및 관련 모듈 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_chroma import Chroma


# ✅ Gemini API 키 설정
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# ✅ PDF 경로 및 고유 벡터DB 경로 지정
PDF_PATH = r"/mount/src/librarychatbot_gemini/안전한 바다여행_최종.pdf"
PDF_NAME = os.path.splitext(os.path.basename(PDF_PATH))[0]
VECTOR_DIR = f"./chroma_db_{PDF_NAME}"  # PDF마다 고유 폴더

# ✅ Streamlit 캐시 초기화 옵션
if st.button("🔄 캐시 및 임베딩 데이터 초기화"):
    if os.path.exists(VECTOR_DIR):
        import shutil
        shutil.rmtree(VECTOR_DIR)
        st.success("✅ 이전 ChromaDB 데이터 삭제 완료!")
    st.cache_resource.clear()
    st.success("✅ Streamlit 캐시 초기화 완료! 앱을 새로고침하세요.")

# ✅ PDF 로드 및 분할
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# ✅ ChromaDB 생성
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    st.info("🤖 임베딩 모델 로드 중... (첫 실행 시 모델 다운로드)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 및 저장 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=VECTOR_DIR
    )
    st.success("💾 새로운 벡터 데이터베이스 생성 완료!")
    return vectorstore

# ✅ 기존 데이터가 있으면 로드, 없으면 새로 생성
@st.cache_resource
def get_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(VECTOR_DIR):
        st.info("📂 기존 안전한 바다여행 벡터DB 로드 중...")
        return Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    else:
        return create_vector_store(_docs)

# ✅ 전체 초기화 (RAG 체인)
@st.cache_resource
def initialize_components(selected_model):
    pages = load_and_split_pdf(PDF_PATH)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 질문 재구성용 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    # 질문-답변용 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer short, accurate, and polite. \
    Please answer in Korean and use emojis naturally with your answer. \
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# ✅ Streamlit UI
st.header("🌊 안전한 바다여행 Q&A 챗봇 💬")

if not os.path.exists(VECTOR_DIR):
    st.info("🔄 첫 실행입니다. PDF를 임베딩 중입니다... (약 5분 소요)")
else:
    st.info(f"📂 '{PDF_NAME}' 벡터 데이터베이스를 불러왔습니다!")

option = st.selectbox(
    "Select Gemini Model",
    ("gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite"),
    index=0,
    help="Gemini 2.0 Flash가 가장 빠르고 효율적입니다"
)

try:
    with st.spinner("🔧 챗봇 초기화 중... 잠시만 기다려주세요"):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇이 준비되었습니다!")
except Exception as e:
    st.error(f"⚠️ 초기화 중 오류 발생: {str(e)}")
    st.stop()

chat_history = StreamlitChatMessageHistory(key="chat_messages")
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "안전한 바다여행에 대해 궁금한 점을 물어보세요! 🌊😊"
    }]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("질문을 입력하세요..."):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "safe_sea_chat"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)
            answer = response['answer']
            st.write(answer)
            with st.expander("📘 참고 문서 확인"):
                for doc in response['context']:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
