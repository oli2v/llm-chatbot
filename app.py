import streamlit as st

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import utils
from streaming import StreamHandler

st.set_page_config(page_title="eXalt chatbot")
st.header("eXalt chatbot")
st.write("Ask anything about eXalt!")


class ContextDocumentChatbot:

    def __init__(self):
        self.llm_model = "mistral"

    @st.cache_resource
    def setup_chain(_self):
        vector_store = Chroma(
            persist_directory="/home/olivier/llm-playground/chroma_store",
            embedding_function=FastEmbedEmbeddings(),
        )
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5},
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        llm = ChatOllama(model="mistral")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            """
            You are a chatbot tasked with responding to questions about the company eXalt.
            You should never answer a question with a question.
            Your response must be synthetic and no longer than 3 sentences.
            Given a question, you should respond with the most relevant information using the context below:\n
            -----
            {context}
            -----
            """
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_messages(
                    [
                        system_message_prompt,
                        human_message_prompt,
                    ]
                ),
            },
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, "user")
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    obj = ContextDocumentChatbot()
    obj.main()
