
# Import necessary library for File_Processing
import fitz
import nltk
import base64
import os
from PIL import Image
from sentence_transformers import SentenceTransformer

# Downloading the Model for Sentence Transformer
model = SentenceTransformer('paraphrase-MiniLM-L6-V2') # classify the Sentence transformer we run
nltk.download('punkt_tab') # Model needed for any NLP or Natural Language Process

# This section is where we will be checking what file path is input receiving and tries to open the file
def is_pdf(file_path):
    return file_path.lower().endswith('.pdf')

# If it checks and found that image opne the image and return yes
def is_image(file_path):
    try:
        Image.open(file_path)
        return True
    except:
        return False

# This is the function that handles the entire pdf file process
def extract_text_from_pdf(pdf_path): #This will get the text from pdf to plain text
    doc = fitz.open(pdf_path)
    text = []
    print(f"PDF has {len(doc)} pages")
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks") #extract the text we got from the block we can see
        page_text = "\n".join([block[4] for block in blocks if block[4].strip()]) # join each block of text to the page
        text.append(page_text) # Add the text we need correlate to the page text we got and ready for the upsert process
        print(f"Extracted {len(page_text)} characters from page {page_num + 1}") # Grab the amount of text we grab by character to the page we gpt

    print(f"Total pages processed: {len(text)}") # Tell how many text grab
    return text # Return the text that is splited for the upsert

#Preparing the data as the chunk and ready to upload to database
def prep_data_for_upsert(pages, file_path, max_chunk_size=500): # Tell program how we going to create the text chunk
    all_chunks = [] # Variable to store the text chunk
    for page_num, text in enumerate(pages): # for each page that it able to extract
        page_chunks = [] # create the chunk for each page
        sentences = nltk.sent_tokenize(text)  # Use NLTK for sentence tokenization
        current_chunk = '' #Display the text that it able to extract
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:  # +1 for space
                current_chunk += sentence + ' ' #Create space to separate into chunks
            else:
                if current_chunk:
                    page_chunks.append(current_chunk.strip()) #Add this text to the chunk
                current_chunk = sentence + ' '
        if current_chunk:
            page_chunks.append(current_chunk.strip()) # When it hits the final chunk, add this chunk to the page as well,
        all_chunks.extend([(chunk, page_num) for chunk in page_chunks]) # Iterate through the each chunk receive for each page to all of them
        print(f"Created {len(page_chunks)} chunks from page {page_num + 1}")

    print(f"Total chunks created: {len(all_chunks)} from {len(pages)} pages") #Tell how many chunk it creates

    # This will encode all of the chunks to the vector base and then send the information needed to add to the vector and return the data
    embeddings = model.encode([chunk for chunk, _ in all_chunks])
    data_to_upsert = [
        (f'{file_path}_p{page_num}_c{i}', embedding.tolist(), {'text': chunk, 'source': file_path, 'page': page_num})
        for i, ((chunk, page_num), embedding) in enumerate(zip(all_chunks, embeddings))
    ]
    print(f"Prepared {len(data_to_upsert)} vectors for upsert")
    return data_to_upsert # Return the list of items that will be upsert as vector and ready to upload to Pinecone Database

# This which check for the file path that user uploads, it it is PDF, return yes, other wise check case by case
def check_file_type(file_path):
    if not os.path.exists(file_path):
        return "not_found"
    elif file_path.lower().endswith('.pdf'):
        return "pdf"
    elif is_image(file_path):
        return "image"
    else:
        return "unsupported"

# Image processing functions
def encode_image(file_path):
    with open(file_path, "rb") as image_file:
        file_content = image_file.read()

    base64_image = base64.b64encode(file_content).decode('utf-8')
    file_extension = os.path.splitext(file_path)[1].lower()
    media_type = f"image/{file_extension[1:]}"  # Remove the dot from the extension
    return base64_image, media_type
