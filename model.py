import pickle
import pandas as pd
import numpy as np

class SentimentRecommenderModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
        self.data = pd.read_csv("dataset/sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))
    
    """function to filter the product recommendations using the sentiment model and get the top 5 recommendations"""

    def getSentimentRecommendations(self, user):
        if (user in self.user_final_rating.index):
            # get the product recommedation using the trained ML model
            recommendations = list(
                self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            filtered_data = self.cleaned_data[self.cleaned_data.id.isin(
                recommendations)]
            # preprocess the text before tranforming and predicting
            #filtered_data["reviews_text_cleaned"] = filtered_data["reviews_text"].apply(lambda x: self.preprocess_text(x))
            # transfor the input data using saved tf-idf vectorizer
            X = self.vectorizer.transform(
                filtered_data["reviews_text_cleaned"].values.astype(str))
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            temp = filtered_data[['id', 'predicted_sentiment']]
            temp_grouped = temp.groupby('id', as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(lambda x: temp[(
                temp.id == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped["total_review_count"] = temp_grouped['predicted_sentiment']
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100, 2)
            sorted_products = temp_grouped.sort_values(
                'pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])

        else:
            print(f"User name {user} doesn't exist")
            return None
