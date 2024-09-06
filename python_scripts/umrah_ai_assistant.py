
import openai
from conversational_ai_with_rag import retrieve_relevant_documents, re_rank_documents, generate_response
from serpapi import GoogleSearch
import json
import re
import pandas as pd
from tqdm import tqdm


openai.api_key = ''
serpapi_key = ""
fine_tuned_model_id = ''

params_general_search = {
  "api_key": serpapi_key,
  "engine": "google",
  "q": "jadwal sholat masjid nabaQwi",
  "location": "Austin, Texas, United States",
  "google_domain": "google.com",
  "gl": "us",
  "hl": "en"
}

params_local_food = {
  "api_key": serpapi_key,
  "engine": "google_local",
  "google_domain": "google.com",
  "q": "indonesian restaurant in madinah"
}

params_local_tourism = {
  "api_key": serpapi_key,
  "engine": "google_local",
  "google_domain": "google.com",
  "q": "tempat wisata di riyadh"
}

params_direction = {
  "api_key": serpapi_key,
  "engine": "google_maps_directions",
  "q": "Coffee",
  "hl": "en",
  "start_addr": "makkah",
  "end_addr": "madinah",
  "travel_mode": "3"
}

params_flights = {
  "api_key": serpapi_key,
  "engine": "google_flights",
  "hl": "en",
  "gl": "us",
  "departure_id": "MAN",
  "arrival_id": "JED",
  "outbound_date": "2024-08-01",
  "return_date": "2024-08-07",
  "currency": "USD"
}

params = {
  "api_key": serpapi_key,
  "engine": "google_hotels",
  "q": "hotel in jeddah",
  "hl": "en",
  "gl": "us",
  "check_in_date": "2024-08-01",
  "check_out_date": "2024-08-02",
  "currency": "USD",
  "adults": "2",
  "children": "1",
  "children_ages": "2"
}

def get_result(params):
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def determine_usefulness_and_scraping(query, documents):
    context = "\n\n".join(documents)
    examples = """
    Example 1: {"engine": "google", "q": "jadwal sholat masjid nabawi", "location": "Austin, Texas, United States", "google_domain": "google.com", "gl": "us", "hl": "en"}
    
    Example 2: {"engine": "google_local", "google_domain": "google.com", "q": "indonesian restaurant in madinah"}
    
    Example 3: {"engine": "google_local", "google_domain": "google.com", "q": "tempat wisata di riyadh"}
    
    Example 4: {"engine": "google_maps_directions", "q": "Coffee", "hl": "en", "start_addr": "makkah", "end_addr": "madinah", "travel_mode": "3"}
    
    Example 5: {"engine": "google_flights", "hl": "en", "gl": "us", "departure_id": "MAN", "arrival_id": "JED", "outbound_date": "2024-08-01", "return_date": "2024-08-07", "currency": "USD"}
    
    Example 6: {"engine": "google_hotels", "q": "hotel in jeddah", "hl": "en", "gl": "us", "check_in_date": "2024-08-01", "check_out_date": "2024-08-02", "currency": "USD", "adults": "2", "children": "1", "children_ages": "2"}
    """
    
    messages = [
        {"role": "system", "content": "You are an AI assistant. Your tasks are to: 1) Determine whether the retrieved documents are useful for answering the user's main query. If not, ignore the documents; if they are helpful, use them as context. 2) Assess whether the user's query requires web scraping and identify the type of web scraping needed from the given five types: google search, google local, google maps directions, google flights, or google hotels. 3) Generate the appropriate parameters for the SerpAPI call based on the user's query and examples provided. 4) Your output should ALWAYS be in a dictionary format like this:\n {'user_query': ... , \n'Context from RAG' (if any): ..., \n 'Scraping Params': ..., }\nYou will communicate this output to the fine-tuned model to generate the final response."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": context},
        {"role": "system", "content": "Here are some examples of search parameters for different types of queries: " + examples}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500,
    )
    result = response.choices[0].message.content
    
    # Process results
    try:
        result = eval(result)
        useful_docs = result['Context from RAG']
        scraping_params = result['Scraping Params']
    except:
        try:
            useful_docs = result.split('Context from RAG (if any):')[1].split('Scraping Params')[0]
            scraping_params = result.split('Scraping Params: ')[1]
        except:
#             print("Error parsing scraping params:", result)
            scraping_params = {}
            useful_docs= ""
    return {"useful_docs": useful_docs, "scraping_params": scraping_params}

def perform_web_scraping(params, serpapi_key=serpapi_key):
    serpapi_key = serpapi_key
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def generate_final_response(query, relevant_texts, scraping_results='', fine_tuned_model=fine_tuned_model_id):
    web_scraping_data = json.dumps(scraping_results)
    if len(web_scraping_data)>10000:
        web_scraping_data = web_scraping_data[:10000]
    if relevant_texts:
        if len(relevant_texts)>5000:
            relevant_texts = relevant_texts[:5000]
        
    # Construct the context, emphasizing the use of web-scraped data
    context = f"Use the following real-time web-scraped data to provide an accurate response: {web_scraping_data}\n\nAdditional context from relevant documents: {relevant_texts}"
    
    # Messages to instruct the AI model
    messages = [
        {"role": "system", "content": (
            "You are an AI assistant with expertise in Umrah & Hajj. You will receive a query, real-time web-scraped data, and additional context from relevant documents. "
            "If the web-scraped data is relevant and helpful for answering the query, PRIORITIZE using it. "
            "ALWAYS integrate the web-scraped data into your response when applicable. "
            "If the query is not related to Hajj or Umrah but can be answered using the web-scraped data, use the web-scraped data to provide the response, clearly stating it as the source. "
            "If no relevant information is available in either the documents or web-scraped data, respond respectfully stating it's outside your expertise."
        )},
        {"role": "user", "content": query},
        {"role": "assistant", "content": context}
    ]
    
    response = openai.chat.completions.create(
        model=fine_tuned_model,
        messages=messages,
        max_tokens=2048,
    )
    return response.choices[0].message.content

def ai_umrah_assistant_response(query):
    # Step 1-3: Retrieve and re-rank documents
    initial_relevant_texts = retrieve_relevant_documents(query)
    re_ranked_texts = re_rank_documents(query, initial_relevant_texts)

    # Use GPT-3.5 to determine document usefulness and scraping needs
    analysis = determine_usefulness_and_scraping(query, re_ranked_texts[0][:1500])

    # Extract information from the analysis
    useful_docs = analysis['useful_docs']
#     print('USEFUL DOCS FROM RAG: ', useful_docs)
    
    scraping_results = ''
    if analysis['scraping_params']:
        try: 
            scraping_params = {'api_key':serpapi_key}
            scraping_params.update(analysis['scraping_params'])
    #         print('SCRAPING PARAMS by GPT: ', scraping_params)

            scraping_results = {}
            if scraping_params:
                # Perform web scraping
                scraping_results = perform_web_scraping(scraping_params)
    #             print('SCRAPING RESULTS : ', scraping_results)
        except:
            if analysis['scraping_params']['q']:
                scraping_result = f"an attempt to search '{analysis['scraping_params']['q']}' is failed"
            else:
                scraping_result = 'An error occured when attempting to do web browsing, please try again.'

    # Generate the final response
    response = generate_final_response(query, useful_docs, scraping_results)
#     print("--"*40, "\nGenerated Response:", response)
    return response