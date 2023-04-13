import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import difflib
from fuzzywuzzy import process

# Load the dataset
df = pd.read_csv('song_data.csv')

# Remove duplicates based on song_name
df = df.drop_duplicates(subset='song_name')

# Define the input features
X = df[['acousticness', 'danceability', 'energy', 'liveness', 'loudness', 'tempo']]

# Initialize the scaler and fit_transform the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a new DataFrame with the scaled data and the original index and column names
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Reset the index of both DataFrames
X_scaled_df.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Update y to match the updated data DataFrame
y = df['song_name']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Create the k-NN model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='brute')

# Train the model
knn.fit(X_train)

# Get the nearest neighbors for each point in the test set
neighbors = knn.kneighbors(X_test)
# Prints the song list
print(y)


def recommend_songs(songs_name, data=df, X_data=X, knn_model=knn, n_recommendations=5):
    # Find the index of the input song in the dataset
    song_index = data.index[data['song_name'] == songs_name].tolist()[0]

    # Get the song features from X_data
    song_features = X_data.loc[song_index]

    # Convert the song_features to a DataFrame with feature names
    song_features_df = pd.DataFrame([song_features], columns=X.columns)

    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors(song_features_df, n_neighbors=n_recommendations + 1)

    # Get the indices of the nearest neighbors in the original dataset
    neighbor_indices = data.iloc[y_train.iloc[indices.flatten()].index]['song_name'].index

    # Drop the input song from the neighbor indices only if it's present
    if song_index in neighbor_indices:
        neighbor_indices = neighbor_indices.drop(song_index)

    # Get the song names of the nearest neighbors and return them
    recommended_songs = data.loc[neighbor_indices]['song_name']
    return recommended_songs


def find_closest_song(query, data=df, threshold=70):
    song_names = data['song_name'].tolist()
    best_match, best_match_score = process.extractOne(query, song_names)

    if best_match_score >= threshold:
        return best_match
    else:
        return None


# song_name = input("Please enter a song name: ")
# print(recommend_songs(song_name))
# Get the user's input
user_query = input("Please enter a song name: ")

# Find the closest matching song name in the dataset
closest_song = find_closest_song(user_query)

# Check if a match was found
if closest_song is not None:
    print(f"Found a matching song: {closest_song}")
    # Call the recommend_songs() function with the closest matching song
    recommendations = recommend_songs(closest_song)
    print("Recommended songs:")
    print(recommendations)
else:
    print("No matching song found.")
