import pandas as pd

def clean_data(data: pd.DataFrame):
	data = data.drop(["Index"], axis=1)


	

	return data


if __name__ == "__main__":
	data = pd.read_csv("./data/sentiment_tweets3.csv")
	print(clean_data(data))