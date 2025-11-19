"""
GK_Altrist_V3_GitHubReady

Created by Ratanapara Choorat, Goood
Since 2024-06-15

This is the replica file from my own Project named "Altrist", This is third iteration of the project here, and this is the API Handler file.

What it is? a file manages the backend logic for the AI and Database connections. It handles the indexing of documents into ChromaDB, retrieving relevant information (RAG) using vector search, and sending the final prompts to Anthropic's Claude for both text and image analysis.
"""

# Import necessary library for this section of the code
from importlib import metadata
from anthropic import Anthropic
import time
import chromadb
import nltk
import re
import os
import config
from gk_file_processing import encode_image, extract_text_from_pdf, prep_data_for_upsert, check_file_type
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('paraphrase-MiniLM-L6-V2') # classify the Sentence transformer we run
nltk.download('punkt_tab') # Model needed for NLP or Natural Language Process

# Use environment variables for API keys and being the backbone of the program
anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Initialize ChromaDB and save to the disk for local
client = chromadb.PersistentClient(path="./chroma_db/")
# Create Collection for ChromaDB to store the vector data
chroma_collection = client.get_or_create_collection(name="gk_altrist_collection")

#This is where program index the file and prepare to upload to the pinecone database
def index_pdf_files(pdf_files):
    for pdf_file in pdf_files:
        try:
            pages = extract_text_from_pdf(pdf_file) # Finds the page that it has for the documents
            data = prep_data_for_upsert(pages, pdf_file) #Get the text chunk ready for upsert to the index

            if not data:
                print(f"No data to upsert for '{pdf_file}'")
                continue

            print(f"Prepared {len(data)} vectors for '{pdf_file}'") # Tell user that data is ready to upsert

            # Since we change the Database Structure to ChromaDB
            # We first unzip the data to create the list of ids, metadatas, and embeddings
            ids = [item[0] for item in data]
            embeddings = [item[1] for item in data]
            metadatas = [item[2] for item in data]
            documents = [item[2]['text'] for item in data] # Requires for ChromaDB to have documents field from the text chunk

            # Upsert the data into ChromaDB
            chroma_collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            print(f"Upserted {len(data)} vectors for '{pdf_file}' into ChromaDB")

        except Exception as e:
            print(f"Error processing '{pdf_file}': {e}")
    print("Finished indexing PDF files.")

#Grab the results needed from the documents sorting the order by KNN within 10 nearest member
def retrieve_relevant_info(user_input, top_k=10):
    try:
        if chroma_collection is None:
            raise ValueError("ChromaDB collection is not initialized")

        query_embedding = model.encode([user_input]).tolist() # This will convert the user prompt to the vector based format and then use to find the KNN
        results = chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        print(f"Raw query results: {results}") #Display the information that it sorted

        if not results['ids'][0]:
            return "No relevant information found in the document." # Can't find any relevant info from the docs
        
        #This will return the responses of the file using the result it got and display each value
        relevant_docs = []

        for i in range(len(results['ids'][0])):
            doc_text = results['documents'][0][i]
            source = results['metadatas'][0][i].get('source', 'Unknown Source')
            page = results['pages'][0][i] if 'pages' in results['metadatas'][0][i] else 'Unknown Page'  
            relevant_docs.append(f"Source: {source}, Page: {page}\nContent: {doc_text}")

        if not relevant_docs:
            return "No relevant text found in matching documents." #Can't find relevant info on the docs

        return "\n\n---\n\n".join(relevant_docs) # Return users with the result
    except Exception as e: # This will only run when it can't get the document's information
        print(f"Error in retrieve_relevant_info: {str(e)}")
        return f"Error retrieving information: {str(e)}"

"""

This will create the response needed according of what file type that user inputs

"""
def process_pdf_query(query, client): #PDF path
    try:
        print(f"Querying index with: {query}") #Showing what is user asking
        relevant_info = retrieve_relevant_info(query) #Retrieve information from what user says and find the nearest neighbor
        print(f"Retrieved info: {relevant_info}") #Show the data that user gets from the documents

        #This will check if user retrieve any informtion or not, if not it will look up the general information in websites, etc.
        if relevant_info:
            context = f"Relevant document information:\n{relevant_info}"
        else:
            context = "No relevant information found in the document. I'll try to answer based on general knowledge."

        # Construct the message correctly for Anthropic's API and send the information needed for the AI
        messages = [
            {
                "role": "user",
                "content": f"You are a helpful assistant. If no relevant information is found in the document, please state that clearly, and then try to answer based on your general knowledge. If you can't answer, suggest how the user might rephrase their query or what kind of document might contain the information they're looking for.\n\nContext: {context}\n\nQuestion: {query}"
            }
        ]

        #Create the respond that it needed to be generated and to be show as the final product
        response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=messages
        )
        return response.content #Show only the information needed to respond, aka the text part

    except Exception as e:
        return f"Error processing PDF query: {str(e)}. Please try uploading the file again or check if the file contains the information you're looking for." #This will trigger when it's too irrelevant or too board or any issues with index

def process_image_query(file_content, file_name, query, client): #Image Path
    try:
        #This will use to encode the image to UTF-8 unicode and will check for the file path as well to able to put into the API
        base64_image, media_type = encode_image(file_content, file_name)

        #This will generate the responses needed for user when input the image
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Describe this image in detail, and then answer the following question about it: {query}"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type, #They can handle JPG, PNG, or any widely known name
                                "data": base64_image #Send the UTF-8 data to the API
                            }
                        }
                    ]
                }
            ]
        )
        return response.content #Responses the text chunk

    except Exception as e:
        return f"Error processing image query: {str(e)}" #Fail to process the image due to various reason

def extract_plaintext_from_claude(response_content):
  """
  Extracts plain text from a Claude API response, handling potential formatting variations.

  Args:
      response_content (str): The content string from Claude's response.

  Returns:
      str: The extracted plain text.
  """

  try:
    if response_content and hasattr(response_content[0], 'text') :
      # If the response content is a list of objects with a 'text' attribute
      return response_content[0].text
    else:
        print("Response content format unexpected, returning original content.")
        return str(response_content)  # Return original content if format is unexpected
  except Exception as e:
      print(f"Error extracting plain text: {e}")
      return f"Error: {str(response_content)}"  # Return original content if error occurs
