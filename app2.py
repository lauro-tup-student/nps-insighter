import datetime
from concurrent.futures import ThreadPoolExecutor

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

Show total count of days covered

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


def process_chunk(chunk):
    insights = generate_insights(chunk.to_string())
    return insights


def app():
    with st.container(border=True):
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            # Read the CSV file in chunks
            chunksize = 200
            all_results = []
            filtered_chunk_str = pd.DataFrame()

            # Date range inputs
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")

            if st.button("Generate Insights"):
                with st.spinner("Generating insights..."):
                    with ThreadPoolExecutor(max_workers=20) as executor:
                        for chunk in pd.read_csv(uploaded_file, chunksize=chunksize):
                            chunk['Date'] = pd.to_datetime(chunk['Date'], format="%m/%d/%Y", errors='coerce').dt.date
                            filtered_chunk = chunk[(chunk['Date'] >= start_date) & (chunk['Date'] <= end_date)]

                            # add chunk to filtered_chunk_str
                            filtered_chunk_str = pd.concat([filtered_chunk_str, filtered_chunk], ignore_index=True)

                            # Submit each chunk for processing in parallel
                            future = executor.submit(process_chunk, filtered_chunk)
                            all_results.append(future)

                            # Wait for all tasks to finish and collect results
                        final_results = [future.result() for future in all_results]
                    all_results = "\n".join(final_results)

                    with st.container(border=True):
                        st.subheader(
                            "Rows from " + str(start_date) + " to " + str(end_date) + "\n" + generate_insights(all_results))
                        st.dataframe(filtered_chunk_str, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    app()
