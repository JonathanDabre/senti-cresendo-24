import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")

# Function to get sentiment score for a given text
# Function to get sentiment score for a given text
def map_sentiment_score_to_category(score):
    category_mapping = {0: "Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Positive"}
    return category_mapping.get(score, "Undefined")

def get_sentiment_score(text):
    # Convert text to string to handle non-string data
    text = str(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax(dim=1).item()
    
    return predicted_label


# Load the CSV file into a DataFrame
csv_file = "Listerine_reviews.csv"
df = pd.read_csv(csv_file)

# Apply the get_sentiment_score function to each review and add sentiment score to a new column
df['Sentiment Score'] = df['Review Text'].apply(get_sentiment_score)
df['Sentiment Category'] = df['Sentiment Score'].apply(map_sentiment_score_to_category)

# Save the updated DataFrame back to CSV
output_csv_file = "updated_Listerine_review.csv"
df.to_csv(output_csv_file, index=False)

print("Sentiment scores added to the CSV file successfully.")
