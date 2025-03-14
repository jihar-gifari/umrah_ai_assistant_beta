
import os
import openai
import streamlit as st
from streamlit_chat import message

# # Fetch your API key securely
# openai.api_key = st.secrets["OPENAI_API_KEY"]

# # Instantiate the OpenAI client
# client = openai.OpenAI(api_key=openai.api_key)

# Fetch API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Correct OpenAI client initialization
openai.api_key = api_key 

client = OpenAI()

st.set_page_config(page_title="AI Umrah Assistant", layout="wide")
st.title('AI Umrah Assistant')

# Initialize chat_log as a list in the session state if it doesn't exist
if 'chat_log' not in st.session_state:
    st.session_state['chat_log'] = []

if 'user_input_field' not in st.session_state:
    st.session_state['user_input_field'] = ""
    
# Function to interact with the fine-tuned model
def ask(model, prompt, chat_log):
    # Construct the list of messages for the API call
    messages = chat_log + [{"role": "user", "content": prompt}]
    
    # Make the API call using the new client instance method
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=150  # Adjust the number of tokens as needed
    )
    
    # Extract the message content from the response
    message_content = response.choices[0].message.content
    
    # Append the assistant's response to the chat log
    chat_log.append({"role": "assistant", "content": message_content})
    
    return message_content

def display_chat():
    # Empty the chat display area first
    st.session_state['displayed_chat_log'] = []
    
    # Display the chat log
    for idx, msg in enumerate(st.session_state['chat_log']):
        # Pass a unique key for each message using the index in the list
        key = f"msg_{idx}"
        if msg not in st.session_state['displayed_chat_log']:
            if msg["role"] == "user":
                message(msg["content"], key=key, is_user=True)
            else:
                message(msg["content"], key=key, is_user=False)
            # Add the message to the list of displayed messages to avoid duplicates
            st.session_state['displayed_chat_log'].append(msg)

def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Append the user's message to the chat log
        st.session_state['chat_log'].append({"role": "user", "content": user_input})
        
        # Get the assistant's response
        model_name = "ft:gpt-3.5-turbo-0125:personal:umroh-ai-trial:9BFSDZQX"
        response = ask(model_name, user_input, st.session_state['chat_log'])
        
        # Append the assistant's response to the chat log
        st.session_state['chat_log'].append({"role": "assistant", "content": response})

        # Clear the input box and rerun the app to refresh the display
        st.session_state.user_input = ""

# Place the display_chat function inside the chat_input_area function
def chat_input_area():
    # Display the chat log
    display_chat()
    
    # Inline CSS to style the button and input with the same height
    button_css = """
    <style>
    div.stButton > button:first-child {
        height: 2em; /* Match the input field height */
        width: 2em;  /* Set the width to match height for a square appearance */
        border-radius: 1.5em; /* Adjust for a circular appearance */
        font-size: 1.5em; /* Adjust the size of the icon */
        margin-top: 0.15em; /* You may need to adjust this for alignment */
    }
    /* Adjust the style of the input box to match the button */
    div.stTextInput > div > div > input {
        height: 3em;
        margin-top: 0.15em; /* This margin is for vertical centering */
    }
    /* This ensures vertical alignment is centered in the column */
    div.stTextInput, div.stButton {
        display: flex;
        align-items: center;
    }
    </style>
    """

    # Display custom CSS
    st.markdown(button_css, unsafe_allow_html=True)
    
    # Use columns for layout
    cols = st.columns([0.9, 0.1])
    with cols[0]:
        # Text input for user message
        user_input = st.text_input("", placeholder="Type your message here...", key="user_input")
    with cols[1]:
        # Button to send message
        send_button = st.button("⬆️", on_click=send_message)

# Call the chat_input_area function to show the input area and the chat log
chat_input_area()
