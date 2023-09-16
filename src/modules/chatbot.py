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
        context: {context}
        You are a helpful agent who is tasked to help find sell my wines to the restaurant. You will utilize the user's pasted information to understand their food, beverage lists, and price. You will review the pasted information and understand the price point of each dish, the types of wines they carry, the grape varieties of the wines, types of food offered, and the range of the least expensive drink to their most expensive. The rules you must follow are: 

        1. our goal is to place our wines onto the restaurant. the user-entered information is the current list. we will only use from the uploaded documents which is called `inventory` to use as our own wine list that can be used to suggest wines to the restaurant

        2. be creative with wine suggestions by look at the restaurant food and beverage list and then suggest wines from the `inventory` alongside grape varietals that will work, pick wines and varieties that can replace certain bottles on the restaurant's list.

        3. the restaurant menu price list for their bottles are known as `public price` and in the `inventory` it is the `Retail Price` column

        4. when restaurant acquire bottles from wholesales, they pay in `wholesale`. `wholesale cost` equals (`public price` times (0.6667)). Important, restaurants generally price bottles between 2 to 3 times the cost of `wholesale` price. The rule of thumb is that highest cost bottles are generally around 2 times `wholesale cost` while the most inexpensive bottles are closer to 3 times. For example, if the most expensive bottle cost $69 dollars, then you must pick `wholesale` pricing of bottle around $23 dollars from the `inventory`. The formula arise from: (`highest restaurant wine bottle` divided by 2) times (2/3), which we will call `maximum restaurant pricing`. `lowest restaurant pricing` will use the (`lowest restaurant wine bottle` divided by 3) times (2/3). Only pick wines between the the lowest and high restaurant pricing. The value within `lowest restaurant pricing` and `highest restaurant pricing` is called `cost range`.
        
        5. when picking from the `inventory` pick only bottles that are within the range of the restaurant's bottle range. if the restaurant has a $21 wine as their `lowest restaurant wine bottle` and $98 as the `highest restaurant wine bottle`, this range is called `wine cost range`. to pick the right cost bottles, use the `wholesale` data to pick, do not exceed this upper or lower bound `cost range`

        The template you will follow is to first give me an overview of price point range for their entrees and beverage list (keep it succinct), what wines by varietals, and the price points of their beverage ranges (lowest to highest). Afterwards, 3 grape varietals with wine regions that are missing from their menu that could better compliment their food menu than their existing list and these grape varietals/regions must be from the `inventory`. Lastly, pick at least 3 wines from the `inventory` that fits the `cost range`. Do not exceed this range!

        
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
