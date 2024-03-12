
# import packages
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os
import re

def main():
    print("Welcome to chatbot!")
    # find path to the current file
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    # setup working directory
    os.chdir(dname)

    # load and check openai_api_key
    #user_api_key = os.environ.get('OPENAI_API_KEY')
    user_api_key = input("Please provide OPENAI_API_KEY:")
    #print(f"Provided OPENAI_API_KEY:\n{user_api_key}")
    print("    ...done")

    # load external csv file into row-based documents
    print("Loading provided csv file...")
    csv_file_path = 'data/alerts.csv'
    loader = CSVLoader(file_path=csv_file_path, encoding="utf-8")
    data = loader.load()
    print("    ...done")

    # embedding of the information provided in the csv file
    print("Chatbot initilization...")
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    print("    ...done")

    print("Learning from provided csv...")
    vectors = FAISS.from_documents(data, embeddings)
    print("    ...done")
    print("#################################################")
    print("")

    # print welcome text
    print("##################################################################################")
    print("##################### >> WELCOME TO KD CUSTOMIZED CHATBOT << #####################")
    print("##################################################################################")
    print("")

    # define chain
    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature=0.0,
                         model_name='gpt-3.5-turbo',
                         openai_api_key=user_api_key),
        retriever=vectors.as_retriever(),
        return_source_documents=True
                         )
    
    # send welcome message by chat
    print(f"> Bot: Hi, I am chatbot trained by your csv file with alerts! How can I help you?")

    # defince tabs for pretty printing chatbot answers
    ls1 = "    "

    # chat history
    history = []
    
    # setup chat hearing in the infinite loop
    while True:
        user_input = input("> You: ")
        if (user_input.lower()=='exit' or user_input.lower()=='quit' or user_input.lower()=='bye'):
            print(f"> Bot:\n{ls1}-[ANSWER]:\n{ls1}Goodbye!")
            break

        # get result from chatbot
        result = chain.invoke({"question": user_input, "chat_history": history})

        # capture alrt id's
        alert_id = []
        source_docs = result['source_documents']
        for i in source_docs:
            li = list(str(i).split("\\n"))[0]
            li = re.sub('\D', '', li)
            alert_id.append(li)

        # print chatbot answer
        response = result["answer"]
        print(f"> Bot:\n{ls1}-[INFO]:\n{ls1}Text generated based on AlertId's: {alert_id}\n{ls1}-[ANSWER]:\n{ls1}{response}")
        
        # flush alert_id
        alert_id = None

# run main function
if __name__ == "__main__":
    main()