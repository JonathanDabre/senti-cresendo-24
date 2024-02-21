from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")


text = "ok ok product"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

outputs = model(**inputs)

predicted_label = outputs.logits.argmax(dim=1).item()


print("Predicted sentiment label:", predicted_label)
