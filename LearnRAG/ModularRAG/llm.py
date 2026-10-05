from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

class LLM:
    def __init__(self, model_name : str, api_key : str):
        self.model_name = model_name
        self.api_key = api_key
        self.llm = ChatGroq(model=self.model_name,temperature=0.1,max_tokens=1024)
        print(f"Initialized Groq LLM with model: {self.model_name}")
        print()

    def generate_response(self, context : str, query : str) -> str:
        prompt_template = PromptTemplate(
            input_variables=["context","question"],
            template="""
            You are a helpful AI assisstent. Use the following context to answer the question accurately and concisely.
            Context:
            {context}
            Question:
            {question}
            Answer:
            Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so.
            """
        )
        prompt_formatted = prompt_template.format(context=context,question=query)
        try:
            messages = [HumanMessage(content=prompt_formatted)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error occured while generating response by Groq LLM model {self.model_name} : {e}")
            print()