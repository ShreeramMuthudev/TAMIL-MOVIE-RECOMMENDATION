import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (replace 'data.csv' with your file)
data = pd.read_csv('Tamil_movies_dataset.csv')

# Combine selected columns into a single feature column
features = ['Genre', 'Rating', 'Director', 'Actor', 'PeopleVote', 'Year', 'Hero_Rating', 'movie_rating', 'content_rating']
data['combined_features'] = data[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Create a CountVectorizer and fit_transform the combined features
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)
def get_recommendations(movie_name, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = data[data['MovieName'] == movie_name].index[0]

    # Pairwise similarity scores with other movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies
    top_similar_movies = sim_scores[1:11]

    # Get movie indices
    movie_indices = [i[0] for i in top_similar_movies]

    # Return the top 10 most similar movies
    return data.iloc[movie_indices]['MovieName']
import streamlit as st

st.set_page_config(page_title="SRMD")
st.title("Tamil Movie Recommendation System")


import base64
# Function to convert file to base64
def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your image
image_path = r"ajith.jpg"


# Convert the image to base64
image_base64 = get_image_as_base64(image_path)

# Function to add background from local
def add_bg_from_base64(base64_string):
    # The corrected CSS string with the base64 image
    css_string = f'''
.stApp {{
    background-image: url("data:image/jpg;base64,{base64_string}");
    background-size: 100%;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;

}}
.stApp h1 {{
    color: #ffffff; /* Title color */
}}
'''
    # Using st.markdown to inject the CSS string with the base64 image
    st.markdown(f'<style>{css_string}</style>', unsafe_allow_html=True)

    #     f"""
    #     <style>
        # .stApp {{
        #     background-image: url("data:image/jpg;base64,{base64_string}");
        #     background-size: cover;
        #     background-repeat: no-repeat;
        #     background-attachment: fixed;
    #     }}

    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

# Call the function to add the background
add_bg_from_base64(image_base64)



# Select box for movie selection
selected_movie = st.selectbox('Select a movie:', data['MovieName'])

# Display selected movie details
if st.button('Show Movie Details'):
    movie_details = data[data['MovieName'] == selected_movie].squeeze()
    st.subheader('Movie Details:')
    
    st.write(f"**Name:** {movie_details['MovieName']}")
    st.write(f"**Director:** {movie_details['Director']}")
    st.write(f"**Actor:** {movie_details['Actor']}")
    st.write(f"**Genre:** {movie_details['Genre']}")
    st.write(f"**Rating:** {movie_details['Rating']}")
    st.write(f"**PeopleVote:** {movie_details['PeopleVote']}")
    st.write(f"**Year:** {movie_details['Year']}")
    st.write(f"**Hero_Rating:** {movie_details['Hero_Rating']}")
    st.write(f"**movie_rating:** {movie_details['movie_rating']}")
    st.write(f"**content_rating:** {movie_details['content_rating']}")

# Recommendations section
if st.button('Get Recommendations'):
    recommendations = get_recommendations(selected_movie)
    st.subheader('Top 10 Recommendations:')
    for movie in recommendations:
        st.write(movie)



# Contact me button
if st.button("Contact Me"):
    st.empty()  # Clear the current content
    st.title("Contact Page")  # Display new content for contact page
    st.markdown("---")
    st.markdown("**Contact Information:**")
st.markdown("- Email: srmdguru@gmail.com", unsafe_allow_html=True)
st.markdown("- Twitter: [My Twitter](https://twitter.com/SRMuthuDev)", unsafe_allow_html=True)
st.markdown("- Instagram: [My Instagram](https://www.instagram.com/sr_md_10/)", unsafe_allow_html=True)
st.write("THIS WEBSITE MIGHT BE USEFUL FOR U THANKS.", unsafe_allow_html=True)
