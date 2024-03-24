# Importations standard
import os
from operator import itemgetter
import gradio as gr
import base64
from mimetypes import guess_type
import os
import cv2
import numpy as np
import sys
import sounddevice
import gtts
from playsound import playsound 
import speech_recognition as sr
from gtts import gTTS

# Importations tierces
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant, FAISS
from langchain_text_splitters import CharacterTextSplitter
from openai import AzureOpenAI

# Reconnaissance vocale
recognizer = sr.Recognizer()

# Configuration des variables d'environnement
os.environ["AZURE_OPENAI_API_KEY"] = ("INSERT YOUR API KEY HERE")
os.environ["AZURE_OPENAI_ENDPOINT"] = ("INSERT YOUR ENDPOINT HERE")

api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key= os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = 'DEPLOYEMENT_NAME'
api_version = 'API_VERSION'

# Initialisation des objets
embeddings = AzureOpenAIEmbeddings(azure_deployment="DEPLOYEMENT_NAME", openai_api_version="API_VERSION")
modeltemp = AzureChatOpenAI(openai_api_version="API_VERSION", azure_deployment="DEPLOYEMENT_NAME", temperature=0, max_tokens=250)

# Reconnaissance vocale
def transcribe(audio):
    try:
        sample_rate = audio[0]
        audio_data = audio[1]
        
        # Convert stereo audio data to mono
        audio_data_mono = np.mean(audio_data, axis=1)
        
        # Convert data to bytes
        audio_data_bytes = audio_data_mono.astype(np.int16).tobytes()
        
        # Set sample width to 2 bytes (16 bits)
        sample_width = 2
        
        audio_instance = sr.AudioData(audio_data_bytes, sample_rate=sample_rate, sample_width=sample_width)
        text = recognizer.recognize_google(audio_instance, language='fr-FR', show_all=False)
        question = ask_question(text, memory, loaded_memory, standalone_question, retrieved_documents, answer)
        text_to_speech(question)
        return question
    except sr.UnknownValueError:
        return "Impossible de reconnaître le discours"
    except sr.RequestError as e:
        return "Erreur lors de la requête vers Google Speech Recognition: {0}".format(e)

# Text to speech
def text_to_speech(text):
    # Créer un objet gTTS
    tts = gTTS(text=text, lang='fr')
    # Sauvegarder le fichier audio
    tts.save("output.mp3")
    # Jouer le fichier audio
    playsound("output.mp3")
    # Supprimer le fichier audio après la lecture
    os.remove("output.mp3")

#image
def image_to_data_url(image: np.ndarray, mime_type="image/jpeg"):
    # Convert the image array into bytes
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    byte_im = im_buf_arr.tobytes()

    # Encode the bytes into base64
    base64_encoded_data = base64.b64encode(byte_im).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def describe_image(image_path):
    data_url = image_to_data_url(image_path)

    client = AzureOpenAI(
        api_key=api_key,  
        api_version=api_version,
        base_url=f"{api_base}openai/deployments/{deployment_name}/extensions",
    )

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": f"Respond with only the personality of {personality} and speak like {personality}, still you'll have all the expertise of the linked document and will answer to the asked question while not breaking character ; Decribe this picture: "  
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ] } 
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

# Utilisation de l'interface Gradio pour poser des questions
def ask_question_wrapper(question, str):
    return ask_question(question, memory, loaded_memory, standalone_question, retrieved_documents, answer)

# Liste des noms de fichiers PDF
pdf_files = ["YOUR_PDF_FILE_PATH"]
#"C:/Users/ulyss/HACKATON/s71500_et200mp_system_manual_en-US_en-US.pdf"

# Initialiser une liste vide pour stocker les documents
docs = []

# Parcourir la liste des noms de fichiers
for pdf_file in pdf_files:
    # Charger et diviser chaque fichier PDF
    loader = PyPDFLoader(pdf_file)
    docs += loader.load_and_split()

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

#first question
template = """Answer the question based only on the following context et avec exclsuivement la personnalitée de {personality} et en parlant exclusivement de la maniere de {personality}:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original languageet avec exclsuivement la personnalitée de {personality} et en parlant exclusivement de la maniere de {personality}: et en parlant à la première personne.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)      

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

#memory
memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
global personality
personality = "Intelligence Artificielle Lenny Barre qui est un assistant virtuel intelligent et serviable"
def update(name):
    global personality
    personality = f"{name}"
    return f"Welcome to Gradio, {personality}!"


with gr.Blocks() as demo:
    with gr.Tab("Chatbot"):
        gr.Markdown("Appuye sur run après avoir donnez la personnalité")
        with gr.Row():
            inp = gr.Textbox(placeholder="Donnez la personnalité a votre IA")
        btn = gr.Button("Run")
        btn.click(fn=update, inputs=inp)
        gr.ChatInterface(fn=ask_question_wrapper, title="LENNY BARRE")
    with gr.Tab("Image"):
        gr.Interface(fn=describe_image, inputs="image", outputs="text")
    with gr.Tab("Audio"):
        gr.Interface(fn=transcribe, inputs="microphone", outputs="text")
# Maintenant, nous calculons la question autonome
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        "personality": lambda x: personality,  # Ajout de la personnalité
    }
    | CONDENSE_QUESTION_PROMPT
    | modeltemp
    | StrOutputParser(),
}

#Combine the documents
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """
    Combines a list of documents into a single document string.

    Args:
        docs (list): A list of documents to be combined.
        document_prompt (str, optional): The prompt to be added before each document. Defaults to DEFAULT_DOCUMENT_PROMPT.
        document_separator (str, optional): The separator to be added between each document. Defaults to "\n\n".

    Returns:
        str: The combined document string.
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
    "personality": lambda x: personality,  # Ajout de la personnalité
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
    "personality": lambda x: personality,  # Ajout de la personnalité
}

# Création de la réponse
answer = {
    "answer": final_inputs | ANSWER_PROMPT | modeltemp,
    "docs": itemgetter("docs"),
    "personality": lambda x: personality,  # Ajout de la personnalité
}

def ask_question(question, memory, loaded_memory, standalone_question, retrieved_documents, answer):
    """
    Asks a question and retrieves the answer using the provided memory and retrieval models.

    Args:
        question (str): The question to ask.
        memory: The memory object used for saving and loading context.
        loaded_memory: The loaded memory variables.
        standalone_question: The standalone question model.
        retrieved_documents: The retrieved documents.
        answer: The answer model.

    Returns:
        str: The answer to the question.
    """

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    inputs = {"question": question}
    result = final_chain.invoke(inputs)
    memory.save_context(inputs, {"answer": result["answer"].content})
    memory.load_memory_variables({})
  
    return result["answer"].content
    
demo.launch()
