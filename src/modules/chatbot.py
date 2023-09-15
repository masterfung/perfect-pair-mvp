import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

# fix Error: module 'langchain' has no attribute 'verbose'
import langchain
langchain.verbose = False


class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    qa_template = """
        You are a helpful agent who is tasked to help find sell my wines to the restaurant. You will utilize the pasted menu and beverage list to understand their food, beverage lists, and price point. You will review the restaurant menu and beverage list and understand the price point of each dish, the types of wines they carry, the grape varieties of the wines, and types of food offered. The rules you must follow are: 

        1. we want to place our wines onto the restaurant's list. we will only use from the CSV document which is called `inventory`
        2. be creative with wine suggestions by look at their list and suggest wine varietals that compliment best with their food and the wines from the CSV
        3. the price of the bottles must be within the price range of the restaurant's wine list, called `wine cost range`. The lowest cost wine bottle is called `lowest restaurant wine bottle` and highest cost bottle is called `highest restaurant wine bottle`.
        4. bottle pricing is the `price` column in the CSV and this value is the `retail pricing`. 
        5. Restaurants pay in `wholesale cost`. `wholesale cost` equals (`retail pricing` times (0.6667)). For example, restaurants generally price bottles between 2 to 3 times the cost of wholesale price. The rule of thumb is that highest cost bottles are generally around 2 times `wholesale cost` while the most inexpensive bottles are closer to 3 times. For example, if the most expensive bottle cost $69 dollars, then you must pick wholesale pricing of bottle around $23 dollars. The formula arise from: (`highest restaurant wine bottle` divided by 2) times (2/3), which we will call `maximum restaurant pricing`. `lowest restaurant pricing` will use the (`lowest restaurant wine bottle` divided by 3) times (2/3). Only pick wines between the the lowest and high restaurant pricing. The value within `lowest restaurant pricing` and `highest restaurant pricing` is called `cost range`.

        The template you will follow is to first give me an overview of price point range for their entrees and beverage list (segment by beverage types), what wines by varietals, and the price points of their beverage ranges. Afterwards, suggests grape varietals and wine regions that are missing from their menu that could better compliment their food menu than their existing list. Lastly, pick several wines from the `inventory` that fits the `cost range`. Remember that all future prompts need to obey these ranges and can only be based from the CSV.

        context: {context}
        =========
        question: {question}
        ======
        """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=[
                               "context", "question"])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name,
                         temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                      retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query,
                       "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        # count_tokens_chain(chain, chain_input)
        return result["answer"]


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(
            f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result
