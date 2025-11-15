import os
import sys
import streamlit as st
import nest_asyncio

# Streamlit 비동기 충돌 방지
nest_asyncio.apply()

# ================================
# 1. LangChain & Chroma 임포트
# ================================
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

# 🔥 최신 Chroma Settings 사용 (dict 절대 사용 X)
from langchain_chroma import Chroma, Settings


# ================================
# 2. Google Gemini API Key
# ================================
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요.")
    st.stop()


# ================================
# 3. PDF 설정
# ================================
PDF_PATH = r"/mount/src/librarychatbot_gemini/안전한 바다여행_최종.pdf"
PDF_NAME = os.path.splitext(os.path.basename(PDF_PATH))[0]

VECTOR_DIR = f"./chroma_db_{PDF_NAME}"


# ================================
# 4. Streamlit UI — 캐시 초기화
# ================================
if st.button("🔄 ChromaDB / 캐시 초기화"):
    import shutil
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    st.cache_resource.clear()
    st.success("♻️ 초기화 완료! 새로고침하세요.")


# ================================
# 5. PDF 로드 & 분할
# ================================
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


# ================================
# 6. Chroma Settings (필수!!)
# ================================
def get_chroma_settings():
    return Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=VECTOR_DIR,
        anonymized_telemetry=False
    )


# ================================
# 7. ChromaDB 생성
# ================================
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(_docs)

    st.info(f"📄 {len(split_docs)}개 청크로 분할 완료.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 임베딩 생성 중...")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="default",
        client_settings=get_chroma_settings()
    )

    st.success("💾 ChromaDB 저장 완료!")
    return vectorstore


# ================================
# 8. 기존 DB 로드 or 생성
# ================================
@st.cache_resource
def get_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(VECTOR_DIR):
        st.info("📂 기존 ChromaDB 로드 중...")
        return Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="default",
            client_settings=get_chroma_settings()
        )
    else:
        return create_vector_store(_docs)


# ================================
# 9. RAG 초기화
# ================================
@st.cache_resource
def initialize_components(selected_model):
    pages = load_and_split_pdf(PDF_PATH)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # ---- 질문 재구성 프롬프트 ----
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        Given the chat history and the latest user question, rewrite it as a standalone question. 
        Do NOT answer.
        """),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    # ---- QA 프롬프트 ----
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are an assistant for question-answering tasks.
        Use the retrieved context.
        If you don’t know the answer, say you don't know.
        Answer in Korean with emojis.
        {context}
        """),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return rag_chain


# ================================
# 10. UI
# ================================
st.header("🌊 안전한 바다여행 Q&A 챗봇 💬")

if not os.path.exists(VECTOR_DIR):
    st.info("🔄 첫 실행 — 벡터 데이터 생성 중입니다.")
else:
    st.info(f"📂 '{PDF_NAME}' 벡터DB 준비됨!")


selected_model = st.selectbox(
    "Select Gemini model",
    ("gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite")
)


# RAG 로드
try:
    with st.spinner("🔧 시스템 로딩 중..."):
        rag_chain = initialize_components(selected_model)
    st.success("✨ 준비 완료!")
except Exception as e:
    st.error(f"⚠️ 초기화 오류: {e}")
    st.stop()


# ================================
# 11. 대화 히스토리 불러오기
# ================================
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversation_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer"
)


# ================================
# 12. 기존 메시지 출력
# ================================
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


# ================================
# 13. 사용자 입력
# ================================
if user_input := st.chat_input("질문을 입력하세요..."):
    st.chat_message("human").write(user_input)

    with st.chat_message("ai"):
        with st.spinner("답변 생성 중..."):
            response = conversation_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "sea_chat"}}
            )

            st.write(response["answer"])

            # 참고 문서 표시
            if "context" in response:
                with st.expander("📘 참고 문서"):
                    for doc in response["context"]:
                        st.markdown(doc.metadata.get("source", "출처 없음"))
