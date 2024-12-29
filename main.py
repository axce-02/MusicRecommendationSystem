# main.py
import pandas as pd
from recommendation_songs import search_box_ui

# Load the necessary data (make sure these CSV files exist in the data folder)
data = pd.read_csv("data/data.csv")
genre_data = pd.read_csv('data/data_by_genres.csv')
year_data = pd.read_csv('data/data_by_year.csv')

# Define the list of songs for recommendation
song_list = [
    {'name': 'Come As You Are', 'year': 1991},
    {'name': 'Smells Like Teen Spirit', 'year': 1991},
    {'name': 'Lithium', 'year': 1992},
    {'name': 'All Apologies', 'year': 1993}
]

# Call the recommend_songs function from the business logic file
# display_recommendations(song_list, data)
search_box_ui()
# # Print the recommended songs
# print("Recommended Songs:")
# for song in recommended_songs:
#     print(f"Name: {song['name']}, Year: {song['year']}, Artists: {song['artists']}")
