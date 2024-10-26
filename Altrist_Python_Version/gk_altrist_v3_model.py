"""
GK_Altrist_V3_GitHubReady

Created by Ratanapara Choorat, Goood

This is the replica file from my own Project named "Altrist", This is third iteration of the project here, and this is the main loop file, check api_handler and file_processing to see the behind the scenes

For Short Explaination, this is LLM (Large Language Model) and RAG (Retrival Augumented Generation) program to help user with the documents and image to gather useful information from it with the help of AI
"""

"""
This is the Library we use for the program and try to import all the library, please run the install line first before import to the program below
"""

# We need to import all the library and all the functions we need to create the function
import os
from dotenv import load_dotenv
from gk_file_processing import (
    check_file_type,
    extract_text_from_pdf,
    prep_data_for_upsert,
    encode_image
)
from gk_api_handler import (
    create_index_if_not_exists,
    index_pdf_files,
    process_pdf_query,
    process_image_query,
    extract_plaintext_from_claude,
    anthropic_client,
    pinecone_client
)

# Load dotenv (.env) variable to the project, which contains all the API key we need to run the code
load_dotenv("/workspaces/47936376/GK_Altrist_Python/.env")

# Global variable declared here
index = None
current_file_path = None
file_type = None

"""
This is the section where all the file is check whether it's PDF or Image or None of the above

"""
def process_file(file_path, index_name='gdk-l1m-pdf', dimension=384):
    global index, file_type
    # We run the Check file type to determine what user added to the code and return the value to file_type
    file_type = check_file_type(file_path)

    if file_type == "not_found":
        return "not_found", None, None
    elif file_type == "unsupported":
        return "unsupported", None, None
    elif file_type == "pdf":
        try:
            # Create index if it doesn't exist
            index = create_index_if_not_exists(index_name, dimension)

            # Process PDF
            pages = extract_text_from_pdf(file_path)
            data = prep_data_for_upsert(pages, file_path)

            # Index the PDF file
            index_pdf_files(index, [file_path], index_name)

            return "pdf", index, None
        except Exception as e:
            print(f"Error processing PDF file: {str(e)}")
            return "error", None, None
    elif file_type == "image":
        return "image", None, None

    return "unknown", None, None

"""
    This is where all the magic happen, the main loop

    When we run the code, user will welcome with the introduction to the robot itself, and then we will ask user for the file path of the file they want to process, whether it's image or pdf file

    and then the code will run in their own section and then display the final response from the GPT chat itself, in this case, will come from Claude
"""

def main():
    # We declared the variable we declared above and make sure it's universal to every code we do here
    global current_file_path, file_type, index

    print("\nWelcome to Altrist, The Friendly and Helpful Chat System to assist you with Image and PDF file!")
    print("Type 'Exit' at any point of this program to exit from the program when finish using")

    while True:
        # This will be from the start, when user start's new session with chatbot, it will always asks for new file or exit here
        if current_file_path is None:
            file_path = input("\n Please enter the file path you want to upload (or 'exit' to exit from the program): ")

            # When user type exit to terminate program
            if file_path.lower() == 'exit':
                print("Thank you for choosing Altrist, See you next time")
                break

            # We have 3 new variable that we will be using according to process_file function, currently third one is redundant but kept for future use
            result, new_index, _ = process_file(file_path)

            # We will check the result from the provided code above, we want to make sure that the file is upload properly, otherwise flag an error to user for basic error handling
            if result == "not_found":
                print("File Not Found In System, Please check the file path or upload the file again")
            elif result == "unsupported":
                print("File Not Supported. Please upload a File that ends with .pdf or any image file type like (.png or .jpg)")
            elif result == "pdf":
                current_file_path = file_path
                index = new_index
                file_type = "pdf"
                print("PDF index created, ready to query and process")
                # print(f"Debug: Final index value = {index}")  # Debug print
            elif result == "image":
                current_file_path = file_path
                file_type = "image"
                print("Image ready to query")
            elif result == "error":
                print("Unexpected Error while processing and check for file type. Please try again")
            else:
                print("Unknown error occured, Please Try Again.")
        else:
            # After the file uploading, we will now allow user to ask the question to the chatbot
            query = input("\nEnter your question that you want to ask related to the file ('exit' to quit or 'new' to upload a new file and start new session): ")

            # Handling the Exit Clause or New Session Clause from user
            if query.lower() == 'exit':
                print("Thank you for choosing Altrist, See you next time")
                break
            elif query.lower() == "new":
                print("Starting new session")
                # We want to make it blank so it doesn't conflict with the current information from before
                current_file_path = None
                file_type = None
                index = None
                continue

            """

            Next, we will be handling the file processing, that's include getting information to database and grabbing the vector from the embedding process and calculation done by the code itself to use as information sort by KNN or K-Nearest-Neighbour from top 10 result from documents for PDF file only.

            Then, we will be sending the information from relevant information from user or Image information send to Claude via the API and the function created in API Handler and return with the final results

            """

            if file_type == "pdf":
                if index is None:
                    # This error flags on certain condition, usually during the file upload process for first time is blank due to the same index from first time
                    print("Error to process with the index: PDF index not initialized. Please check your pinecone whether there's already created index,\n delete existing index and then upload the PDF File again")
                    current_file_path = None
                    continue
                # Call the pdf processing to get the final response
                result = process_pdf_query(query, index, anthropic_client)
            elif file_type == "image":
                # Read the image file content
                with open(current_file_path, 'rb') as image_file:
                    file_content = image_file.read()

                # Get the file name from the path
                file_name = os.path.basename(current_file_path)

                # Call the image process to get the response from Claude
                result = process_image_query(file_content, file_name, query, anthropic_client)
            else:
                print("Unsupported file type for querying, Pleaes upload new file.")
                current_file_path = None
                continue

            # We want to make sure that we got plain text response from Claude, so we will run this function to polish and make sure about it
            plaintext_response = extract_plaintext_from_claude(result)
            print("\nResult:", plaintext_response)

        # print(f"Debug: Final index value = {index}")  # Debug print

if __name__ == "__main__":
    main()




