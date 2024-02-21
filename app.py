import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib
matplotlib.use('TkAgg')
# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function to perform sentiment analysis
def perform_sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()
    res = {}
    for i, row in df.iterrows():
        if pd.notna(row['Review Text']):
            text = row['Review Text']
            myid = i
            res[myid] = sia.polarity_scores(text)
    sentiment_df = pd.DataFrame(res).T
    # Merge the sentiment DataFrame with the original DataFrame
    merged_df = pd.concat([df, sentiment_df], axis=1)
    
    # Define thresholds
    neutral_threshold_low = -0.33
    neutral_threshold_high = 0.33
    sentiment_classifications=[]
    # Categorize compound scores and add classifications to the list
    for key, value in res.items():
        compound_score = value['compound']
        if compound_score < neutral_threshold_low:
            sentiment_classifications.append('Negative')
        elif compound_score > neutral_threshold_high:
            sentiment_classifications.append('Positive')
        else:
            sentiment_classifications.append('Neutral')
            
    
    merged_df['Sentiment Classification'] = sentiment_classifications


    return merged_df

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.head(500)
    return df

def calculate_overall_sentiment(df):
    # Calculate average compound score
    average_compound_score = df['compound'].mean()
    return average_compound_score

def main():
    st.title('Sentiment Analysis of Mouthwash Reviews')
    
    # Select file to load
    selected_file = st.selectbox("Select Product", ['colgateCavityProtection.csv', 'mouthwash.csv', 'ListerineReviews.csv', 'MaxfreshMouthwash.csv', 'toothbrush.csv', 'toothpaste.csv', 'ListerineTotalCareMildTaste.csv'])

    # Load data
    df = load_data(f'./input/{selected_file}')
    
    # Perform sentiment analysis
    df_with_sentiment = perform_sentiment_analysis(df)
    
    # Display DataFrame with sentiment scores
    st.subheader('Data with Sentiment Scores')
    st.dataframe(df_with_sentiment)

    # Display overall sentiment
    overall_sentiment = calculate_overall_sentiment(df_with_sentiment)
    st.subheader('Overall Sentiment')
    if overall_sentiment > 0.33:
        st.write(f'Overall sentiment: Positive (Compound Score: {overall_sentiment})')
    elif overall_sentiment < -0.33:
        st.write(f'Overall sentiment: Negative (Compound Score: {overall_sentiment})')
    else:
        st.write(f'Overall sentiment: Neutral (Compound Score: {overall_sentiment})')
    
    # Display compound score distribution
    st.subheader('Compound Score Distribution')
    st.bar_chart(df_with_sentiment['compound'])
    
    # Display positive, neutral, and negative score distribution
    st.subheader('Positive, Neutral, and Negative Score Distribution')
    fig, axs = plt.subplots(1, 3, figsize=(15, 9))
    sns.barplot(data=df_with_sentiment, x='Index', y='pos', ax=axs[0])
    sns.barplot(data=df_with_sentiment, x='Index', y='neu', ax=axs[1])
    sns.barplot(data=df_with_sentiment, x='Index', y='neg', ax=axs[2])
    axs[0].set_title('Positive')
    axs[1].set_title('Neutral')
    axs[2].set_title('Negative')
    st.pyplot(fig)
    

    # Histogram of compound scores
    plt.figure(figsize=(8, 6))
    plt.hist(df_with_sentiment['compound'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Histogram of Compound Sentiment Scores')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')

    # Display the figure using Streamlit
    st.pyplot(plt.gcf())
    
    # Boxplot of sentiment scores by rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Rating', y='compound', hue='Rating', data=df_with_sentiment, palette='viridis', legend=False)
    plt.title('Boxplot of Sentiment Scores by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Compound Score')

    # Display the figure using Streamlit
    st.pyplot(plt.gcf())
    
    


if __name__ == '__main__':
    main()
