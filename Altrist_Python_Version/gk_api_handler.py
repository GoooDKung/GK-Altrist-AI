# Import necessary library for this section of the code
from anthropic import Anthropic
import time
import nltk
import re
import os
import config
from gk_file_processing import encode_image, extract_text_from_pdf, prep_data_for_upsert, check_file_type
from pinecone import Pinecone, ServerlessSpec, PineconeException
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('paraphrase-MiniLM-L6-V2') # classify the Sentence transformer we run
nltk.download('punkt_tab') # Model needed for NLP or Natural Language Process

# Use environment variables for API keys and being the backbone of the program
anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Set Pinecone API Key for the Database
pinecone_client = Pinecone(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_ENVIRONMENT
)

try:
    indexes = pinecone_client.list_indexes() # We lists of the indexes of pinecone db here the code here and check if it's ready to use
    print("Pinecone client initialized successfully")
except Exception as e:
    print(f"Error initializing Pinecone client: {str(e)}") # Fail to intialize due to database down, etc.

#Tell program to make pinecone index and create as the specs below
def create_index_if_not_exists(index_name, dimension): # This will create new index to the desired configuration, we use cosine similatiry for this project
    try:
        if index_name not in pinecone_client.list_indexes():
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{index_name}' created successfully.") # Tell user that the index is finished
        else:
            print(f"Index '{index_name}' already exists.") # When user already have the exact same naming for the index

        return pinecone_client.Index(index_name) # Return the pinecone index that we created
    except Exception as e:
        print(f"Error creating or accessing index: {str(e)}") # Flagged error due to various reason like variable call not properly
        return None

#This is where program index the file and prepare to upload to the pinecone database
def index_pdf_files(index, pdf_files, index_name):
    for pdf_file in pdf_files:
        try:
            pages = extract_text_from_pdf(pdf_file) # Finds the page that it has for the documents
            data = prep_data_for_upsert(pages, pdf_file) #Get the text chunk ready for upsert to the index

            print(f"Prepared {len(data)} vectors for '{pdf_file}'") # Tell user that data is ready to upsert

            batch_size = 25  # Reduced batch size
            total_vectors = len(data) #Show how many list it needs to upload to the documents
            vectors_upserted = 0


            #This will try to upsert the information receive as the batch and tries to upserting with the retries maximum at 5 and tries slower to give Pinecone time to think
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                max_retries = 5 # Retries need to make the code to upsert
                retry_delay = 1  # Initial retry delay in seconds

                for attempt in range(max_retries):
                    try:
                        upsert_response = index.upsert(vectors=batch) # Upsert or send data to Pinecone
                        vectors_upserted += len(batch) # Says how many chunk or upserted to the database and update to every iteration
                        print(f"Successfully upserted batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size} for '{pdf_file}'")
                        print(f"Upsert response: {upsert_response}")
                        time.sleep(1)  # Small delay between batches
                        break
                    except PineconeException as pe:
                        if attempt < max_retries - 1:
                            print(f"Pinecone error upserting batch for '{pdf_file}'. Retrying... (Attempt {attempt + 1}/{max_retries})")
                            print(f"Error details: {str(pe)}")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            print(f"Failed to upsert batch for '{pdf_file}' after {max_retries} attempts: {str(pe)}")
                    except Exception as e:
                        print(f"Unexpected error upserting batch for '{pdf_file}': {str(e)}")
                        print("Continuing with the next batch...")
                        break

            print(f"Finished processing '{pdf_file}' into '{index_name}'") #Tell user that the info is up to the Pinecone
            print(f"Total vectors upserted: {vectors_upserted} out of {total_vectors}") # This will tell how much data they got for upserting

            # Verify the index status after processing each file
            try:
                index_stats = index.describe_index_stats()
                print(f"Total vectors in index after processing '{pdf_file}': {index_stats.total_vector_count}") # Show how many index that the documents got
            except Exception as e:
                print(f"Error getting index stats: {str(e)}") #It can't receive the information of the index due to can't create one or improperly create one

        except Exception as e:
            print(f"Error processing file '{pdf_file}': {str(e)}") # it can't responses due to the file complexity

    print("Finished indexing all PDF files.") # Shows that it able to check the information and create the vector it needed

#Grab the results needed from the documents sorting the order by KNN within 10 nearest member
def retrieve_relevant_info(index, user_input, top_k=10):
    try:
        if index is None:
            raise ValueError("Index is not initialized") #When Pinecone failed to create new index or use the previous index

        query_embedding = model.encode([user_input]).tolist() #This will convert the user prompt to the vector based format and then use to find the KNN
        results = index.query(
            vector=query_embedding[0], #Vector checking using the first one
            top_k=top_k,
            include_metadata=True
        )
        print(f"Raw query results: {results}") #Display the information that it sorted

        if not results['matches']:
            return "No matching documents found in the index." #Can't find relevant info on the docs

        #This will return the responses of the file using the result it got and display each value
        relevant_docs = [f"Page {result['metadata']['page']}:\n{result['metadata']['text']}" for result in results['matches'] if 'metadata' in result and 'text' in result['metadata']]

        if not relevant_docs:
            return "No relevant text found in matching documents." #Can't find relevant info on the docs

        return "\n\n---\n\n".join(relevant_docs) # Return users with the result
    except Exception as e: # This will only run when it can't get the document's information
        print(f"Error in retrieve_relevant_info: {str(e)}")
        return f"Error retrieving information: {str(e)}"

"""

This will create the response needed according of what file type that user inputs

"""
def process_pdf_query(query, index, client): #PDF path
    try:
        if index is None:
            return "Error: PDF index not properly initialized. Please upload a PDF file first." #This refers to No Pinecone Index

        print(f"Querying index with: {query}") #Showing what is user asking
        relevant_info = retrieve_relevant_info(index, query) #Retrieve information from what user says and find the nearest neighbor
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
        model="claude-3-5-sonnet-20240620",
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
            model="claude-3-5-sonnet-20240620",
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
    if isinstance(response_content, dict) and 'text' in response_content:
        text = response_content['text']
    elif isinstance(response_content, str):
        text = response_content
    else:
        text = str(response_content)

    # Remove TextBlock wrappers
    text = re.sub(r"TextBlock\(text='(.*?)', type='text'\)", r"\1", text, flags=re.DOTALL)

    # Remove outer brackets
    text = text.strip('[]')

    # Replace \n with actual newlines
    text = text.replace('\\n', '\n')

    # Remove any remaining escape characters
    text = text.replace('\\', '')

    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)

    # Remove any remaining Markdown formatting
    text = re.sub(r"```(?:\w+\n)?(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"[_*`]", "", text)

    # Trim leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # Remove any numbered list formatting
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)

    return text.strip()

  except Exception as e:
      print(f"Error extracting plain text: {e}")
      return response_content  # Return original content if error occurs
