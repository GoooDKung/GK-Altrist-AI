# Altrist Proj.

Welcome to the Altrist Project, codename: GK-L1M Project, an LLM (Large Language Model) and RAG (Retrival Augmented Generation) System to response to the documents user uploads to the file code itself.
The file provided at the top are the builds that. If you ask what happen to V1, well it just history at this point, ask owner to know why. 

If you are wondering what LLM or RAG is about, here's an explanation:

	 LLM, or Large Language Model, is the more advanced version of AI or Artificial Intelligence with the training of text information that enabling to do more tasks like helping you with making a story or 	 suggesting a math solution with the prompt giving enough sufficient information about the internet is needed to enable them to give a human-like response.
	
	 RAG or Retrieval Augmented Generation is one technique used to make artificial intelligence create more robust and more accurate informative responses, enabling themself to generate the information 		needed prompt by the user and the attached files required to allow such features. It is done by storing the information from the documents in smaller chunks and then upload to the Database, where we also 	do the embedding processing, which makes it more dimensional, and store it in vector format, which later converts the vector and sorts to KNN or K-Nearest Neighbour to finally use as information for 		querying and get the final responses that are more accurate.


At this moment, I have implemented 2 separate system, one for Jupyter Notebook, another for Regular Python file. 

## HOW TO USE THE PROGRAM FOR JUPYTER NOTEBOOK:

A. The code only works in Google Colab at the moment, at https://colab.research.google.com, then upload a file to make it work for V2, **from V3 and above are now allow direct upload from the file explorer/finder**

B. Documents are only for English At the moment, the system won't be very stable for other languages yet until localization of the language.

C. You have to supply your own key to make the code work, since there's changes in the Large Language Model Service during Version 2 and 3, C.1 will be Version 2, and C.2 will be more newer version using Claude. *Don't worry about it ChatGPT User, there will be other version that will allow to use ChatGPT instead of Claude as preferred language*

  C.1 Please supply your OpenAI Key at https://platform.openai.com, when making the API please choose an model to the selected model in the web or make an match to the code itself or can edit the program to match the version of the program.
  
  C.2 For Claude or the Version 3, Please go to https://www.anthropic.com/claude, and select **Get API Access** To receive the API Key, then Generate the key, note that you might need to match the version of the Bot in the code to the Claude's console

D. You also need to have your own key for Pinecone to make the code running, you will go to https://www.pinecone.io to access the website and follow insturction to make new account and then replace the comments to make the code running as it's intended

E. You are allow to use file above for controlled environment and test out the results as you like.

## HOW TO USE THE PROGRAM FOR PYTHON VERSION: 

Before we continue anything, I'll give explanation to each file available to each file for the feature itself.

1. ** gk_altrist_v3_model.py ** -> This is the primary function of the code where all the magic happens; this is the file that you can run on a terminal with the following, *python gk_altrist_v3_model.py*.
                                    The program usage will ask users for the path of the file; the file here can be uploaded directly to the compiler of your choice, preferably Virtual Studio Code,
                                    and then it will ask the user for the query, which asks you the question that you want to prompt to a chatbot, in this case, Claude. Finally, it will generate the responses
                                    using the KNN or K-Nearest Neighbour based on the first ten results of the Databasee databases via the Pinecone Database. Then it's done!

 2. ** gk_api_handler.py ** ->      These API-related functions are executed within this file. There are 2 APIs related to this project: Anthropic API Key or Claude API Key and Pinecone Database.
                                    These API keys are essential to the project where it receives and sends information, especially when sending the PDF file format and getting the responses from Claude. This is 				    the step we receive the file from the file_processing library and then transfer it to be polished to upload and ready to use for a query for a bot or upload the image as UTF-8  				    encoding to then upload to Claude.

 3. ** gk_file_processing.py ** -> The file_processing file here is where we manage all the files that we received from the primary function and have separate functions:

                                    PDF -> After identifying as a PDF File, the file will be directed to each page and get the total number of pages; then, it will grab text via the PDF Reader and then put it 					into chunks of text and down into smaller ones. Then afterward, it will be added to the list and then sent to Upsert, which groups down to a group of 25 elements from any 					information we receive; we want to make it smaller to make the program run faster and more efficiently. Afterward, it will be sent to the API_Handler section to prepare 					data for the cloud Database.

                                    Image -> The Image section is more straightforward; after identifying the file format as image, it will convert the image to base64, first encode it as UTF-8 code, and then 					identify the file format of the code, which is later sent to API_Handler will handle the converted UTF-8 file and the format itself.

 4. ** config.py ** ->              This is where we set up all the codes, which are equally important as the code itself. This is where we want to add our API key to enable Anthropic's AI, Claude, and the  	 				    Database, Pinecone, to be functional. To get the API key mentioned here, you going to need to follow these steps:

                                        1. First, open Anthropic's website via this link: https://www.anthropic.com/api. Then, you need to create an account by clicking the "Start Building" Button; it will ask 					for the credentials needed to make an account, then go to the settings and grab the API key for Claude.
                                        2. Then, for Pinecone, access this link: https://www.pinecone.io/; then login or sign up for your account. After this, we go to the API Keys section and then create a new 					API key there
                                        3. Open config.py to copy both API keys into the code, with the Anthropic API key going first and then Pinecone after in the second.


	## HOW TO RUN:
		 1. Follow the guide of config.py to get the API Key needed for the project
		 2. Upload a file to the Program of the Complier of your choice
		 3. Run the Program using this in the command line, *python gk_altrist_v3_model.py*.
		 4. Copy the file path used for the program itself
		 5. Ask the question
		 6. Wait for a response
	
Enjoy the using Altrist, we will be frequently update the code and hopefully making further development such as mobile compatible or local LLM to able to run offline and on the road. Feel free to contact me at @gooodkung in discord if you have any question.

