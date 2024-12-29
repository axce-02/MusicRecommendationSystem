
# **Song Recommendation System**

## **Overview**
This project is a Song Recommendation System that uses Spotify's API and machine learning techniques to provide personalized song recommendations based on user input. The system includes interactive visualizations and clustering methods to analyze musical features and suggest songs.

---

## **Features**
- **Interactive Search**: Enter a song name and year to receive personalized song recommendations.
- **Visual Analysis**: Insights into sound features, genres, and popularity trends over time.
- **Clustering**: Group songs into clusters using KMeans and visualize them using PCA and t-SNE.
- **Spotify Integration**: Fetch song metadata and audio features from Spotify's API.

---

## **Technologies Used**
- **Python Libraries**: 
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `seaborn`, `matplotlib`, `plotly`
  - Machine Learning: `scikit-learn`, `yellowbrick`, `scipy`
  - API Integration: `spotipy`
  - Interactive UI: `streamlit`
- **Spotify API**: Fetch metadata and audio features for songs.

---

## **Setup Instructions**

### **1. Prerequisites**
Ensure the following are installed on your system:
- Python (>= 3.7)
- Pip package manager
- Spotify API credentials (Client ID and Client Secret)

### **2. Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

### **3. Install Dependencies**
Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### **4. Set Up Spotify API**
- Create a Spotify Developer Account and generate your API credentials (Client ID and Secret).
- Add the credentials in your code:
  ```python
  sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="<your_client_id>", client_secret="<your_client_secret>"))
  ```

### **5. Run the Application**
Run the Streamlit application:

```bash
streamlit run main.py
```

---

## **Usage**
1. Open the application in your browser (Streamlit provides a local URL).
2. Enter a song name and year in the input fields.
3. View the recommended songs along with their album art and metadata.

---

## **Project Structure**
```plaintext
.
├── data/                         # Contains input data files
│   ├── data.csv
│   ├── data_by_genres.csv
│   └── data_by_year.csv
├── main.py                       # Entry point for the application
├── recommendation_songs.py       # Business logic for song recommendation
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
```

---

## **Data Files**
Ensure the following CSV files are present in the `data/` directory:
1. `data.csv`: Contains metadata for individual songs.
2. `data_by_genres.csv`: Aggregate data grouped by genres.
3. `data_by_year.csv`: Aggregate data grouped by years.

---

## **Key Functions**
### **`find_song`**
Fetch song details and features from Spotify based on name and year.

### **`recommend_songs`**
Generate personalized song recommendations using cosine similarity.

### **`display_recommendations`**
Streamlit-based UI for displaying recommended songs.

### **`search_box_ui`**
Interactive search box for user input.

---

## **License**
This project is open-source and available under the MIT License.

---
