import datetime

import pandas as pd
import streamlit as st
import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import GenerativeModel

# Initialize Vertex AI outside the functions for efficiency
vertexai.init(project="proven-reality-415301", location="us-central1")
model = GenerativeModel("gemini-1.0-pro-001")  # Initialize model outside for reusability


# Function to generate insights, optimized for clarity and reusability
def generate_insights(data: str) -> str:
    responses = model.generate_content(
        """You are a financial analyst and you are required to summarize the key insights of given NPS Survey Results.

Show the Date Range of the Survey

Show total count of days covered

Show me the number of rows analyzed

Please write in a professional and business-neutral tone.
Add action items for the teams involved
The summary should only be based on the information presented in the data.""" + data,
        # Use a placeholder for the prompt
        generation_config={
            "max_output_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=True,
    )
    text = ""
    for response in responses:
        text += response.text

    return text


def chatbot(raw_df, data: str, question: str, last_message: str) -> str:
    responses = model.generate_content(
        """You are a Net Promoter Expert and Analyst and you are to answer the question based on the NPS Survey Results.
            Please answer  in a professional and business-neutral tone. 
            Answer accurately and clearly.
            Using the following data:
        """ + "Raw Dataframe: " + raw_df + "Insights: " + data + " \n" + question + "last message: " + last_message,
        # Use a placeholder for the prompt
        generation_config={
            "max_output_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=True,
    )
    text = ""
    for response in responses:
        text += response.text
    return text


def clear_chat_state():
    st.session_state["chat"] = ""


# Function to handle file upload and data processing
def handle_upload() -> dict:
    # dictionary to store two results
    results = {}
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])  # Restrict to CSV
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Date range inputs
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        # Convert start_date and end_date to datetime objects
        start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
        end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())

        # Filter data based on date range
        if start_date and end_date:
            df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors='coerce')
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        # Paginate the filtered data
        page_size = 10
        total_rows = len(df)
        total_pages = (total_rows // page_size) + 1

        page_number = st.number_input("Page Number", min_value=1, max_value=total_pages, value=1)

        start_index = (page_number - 1) * page_size
        end_index = start_index + page_size

        df_paginated = df.iloc[start_index:end_index]
        results["df"] = df
        results["df_paginated"] = df_paginated
        return results
    else:
        # Clear session state when no file is uploaded
        st.session_state = {}
        return None


# Main function with state management and UI logic
def main():
    st.markdown("<h1 style='text-align: center;'>NPS Survey Insights Generator</h1>", unsafe_allow_html=True)
    # Initialize insights in session state for insights
    if "insights" not in st.session_state:
        st.session_state["insights"] = ""

    if "chat" not in st.session_state:
        st.session_state["chat"] = ""

    df = handle_upload()  # Get paginated data

    if df is not None:
        # Display table and insights
        st.table(df["df_paginated"])  # Set width as needed

        insights = st.session_state["insights"]

        if st.button("Generate Insights"):
            with st.spinner("Generating insights..."):
                text_rep = df['df'].to_string()
                insights = generate_insights(text_rep)
                st.session_state["insights"] = insights
        insights_str = str(insights)

        # show only the download button if there is already generated insights
        if insights_str:
            st.sidebar.download_button(
                label="Download Insights",
                data=insights_str,
                file_name="insights.txt",
                mime="text/plain",
            )
        # st.sidebar.markdown(insights_str, unsafe_allow_html=True)
        with st.sidebar:
            st.markdown(insights_str, unsafe_allow_html=True)

            # add container and divider
            st.divider()
            with st.container():
                st.subheader("Ask Me Anything")
                question = st.chat_input("Say something")
                if question:
                    # add spinner loading
                    with st.spinner("Generating response..."):
                        bot_response = chatbot(df["df"].to_string(), insights_str, question, st.session_state["chat"])
                        st.session_state["chat"] = bot_response

            message = st.chat_message("ai", avatar="🤖")
            with message:
                message.write("Ask me anything!")
                message.write(st.session_state["chat"])
                clear_button = st.button("Clear chat")
                if clear_button:
                    clear_chat_state()
                    st.experimental_rerun()


if __name__ == '__main__':
    main()
