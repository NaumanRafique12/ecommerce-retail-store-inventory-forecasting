import random
import streamlit as st
from test_metrics import predict, process_tokens


# make the title of the page
st.title("Welcome to the Sentiment Analysis App")

# make the subheader of the page
st.subheader("Testing Demo Target for Prometheus")

# enter the text in the text area
input_text = st.text_area("Enter your text here", "Type something...")


# call the dummy predict button
btn = st.button("Analyze Sentiment")

if btn:
    # split the text into tokens
    tokens = input_text.split(" ")
    # process the tokens
    number_of_tokens = process_tokens(tokens)
    st.write(f"Number of tokens processed: {number_of_tokens}")
    random_delay = random.uniform(0.5, 2.0)  # Random delay between 0.5 and 2 seconds
    st.write(f"Simulating prediction with a delay of {random_delay:.2f}")
    
    # call the predict function
    sentiment_result = predict(random_delay)
    sentiment = sentiment_result["sentiment"]
    st.write(f"The sentiment of the text is: **{sentiment}**")