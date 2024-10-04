import streamlit as st
from pymongo import MongoClient
import urllib, io, json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from huggingface_hub import login
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# Ambil token dari secrets
token = st.secrets["HUGGINGFACE_TOKEN"]  # Ganti dengan nama secret Anda

# Login menggunakan token
login(token)

# Set your OpenAI API Key
api_key = st.secrets["OPENAI_API_KEY"]  # Pastikan secret ini sudah ada di Hugging Face

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# MongoDB Atlas setup
username = "notbeekay"
pwd = "brenda123"
client = MongoClient("mongodb+srv://" + urllib.parse.quote(username) + ":" + urllib.parse.quote(pwd) + "@cluster1.esrw4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1")
db = client["final-project"]
collection = db["locations"]

st.title("SpotSnap: Southeast Asia Travel Recommender Chatbot")
st.image("spotsnap.png", caption="Ask anything about your travel destination and we'll provide you the information!")

# Initialize conversation state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Membuat deskripsi
st.write(
    '''SpotSnap is an AI-based chatbot designed to simplify the process of finding travel destination 
    recommendations according to users' preferences. This chatbot is capable of providing relevant 
    suggestions for tourist spots with complete information, such as the destination's name, place 
    type, rating, price, opening hours, and other related details. SpotSnap can also offer detailed 
    descriptions of tourist attractions and exciting activities available at each recommended location. 
    With SpotSnap, users can receive personalized travel recommendations that help them plan their trips 
    efficiently, making it easier to explore new destinations.'''
    )

# Membuat garis lurus
st.markdown('---')


visualizations , ask_me = st.tabs(['Visualizations','Ask Me'])


with visualizations:
        df1 = pd.read_csv("data.csv")
        st.subheader('Rating Distribution')
        st.write('This chart shows the distribution of ratings for different places.')

        # Function to convert rating to integer (1-5)
        def convert_rating(rating):
            if 1 <= rating < 1.5:
                return 1
            elif 1.6 <= rating < 2.5:
                return 2
            elif 2.6 <= rating < 3.5:
                return 3
            elif 3.6 <= rating < 4.5:
                return 4
            else:
                return 5

        # Apply the function to the 'rating' column
        df1['converted_rating'] = df1['rating'].apply(convert_rating)

        # Count the occurrences of each rating
        rating_counts = df1['converted_rating'].value_counts().sort_index()

        # Create a Plotly pie chart
        fig = px.pie(values=rating_counts.values, names=rating_counts.index, 
                    # title='Rating Distribution',
                    color_discrete_sequence=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

        st.subheader('Top 10 Cities Distribution')
        st.write('This chart illustrates the distribution of the top 10 cities with the highest number of tourist attractions, offering insights into popular travel destinations.')

        # Menghitung jumlah occurrences setiap kota
        city_counts = df1['city'].value_counts().nlargest(10)

        # Membuat skala warna berdasarkan jumlah (count)
        norm = plt.Normalize(city_counts.min(), city_counts.max())
        colors = plt.cm.viridis(norm(city_counts.values))  # Menggunakan skema warna 'viridis'

        # Membuat Bubble Chart
        fig, ax = plt.subplots(figsize=(20, 10))
        scatter = ax.scatter(city_counts.index, city_counts.values, 
                            s=city_counts.values * 100, 
                            c=colors, alpha=0.6, edgecolors="w", linewidth=2)

        # Menambahkan label di setiap bubble
        for i, count in enumerate(city_counts.values):
            ax.text(city_counts.index[i], count, str(count), fontsize=14, ha='center', va='center', color='black')

        # Menambahkan colorbar untuk menunjukkan skala warna berdasarkan jumlah
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Count')

        ax.set_title('Top 10 Cities Distribution', fontsize=14)
        ax.set_xlabel('City',fontsize=14)
        ax.set_ylabel('Count',fontsize=14)

        # Tampilkan chart di Streamlit
        st.pyplot(fig)

        st.subheader("Top 10 Place Types")
        st.write('This chart displays the distribution of the top 10 types of places, highlighting the most common categories of tourist destinations based on the number of occurrences.')
        place_type_count = df1.groupby("place_type").size().reset_index(name="count")
        top_10_place_types = place_type_count.sort_values(by="count", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(top_10_place_types['place_type'], top_10_place_types['count'], color=plt.cm.Paired.colors)

        # Add data labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

        ax.set_title('Top 10 Place Types')
        ax.set_xlabel('Place Type')
        ax.set_ylabel('Count')
        ax.set_xticklabels(top_10_place_types['place_type'], rotation=45, ha='right')
        st.pyplot(fig)

        # Create main dropdown for Place type
        viz_type = st.selectbox("Choose a Place type", ['Amusement Park', 'Aquarium', 'Art Gallery', 'Library', 
                                                        'Movie Theater', 'Museum', 'Park', 'Restaurant', 'Shopping Mall', 
                                                        'Stadium', 'Zoo'])
        if viz_type == "Amusement Park":
            # Bar chart untuk Top 10 Cities dengan Amusement Park
            st.subheader("Top 10 Cities with Most Amusement Parks")
            df_amusement_park = df1[df1['place_type'] == 'amusement_park']
            city_count = df_amusement_park.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            # Add numbers to the bars
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)

        elif viz_type == "Aquarium":
            # Bar chart untuk Top 10 Cities dengan Aquarium
            st.subheader("Top 10 Cities with Most Aquarium")
            df_aquarium = df1[df1['place_type'] == 'aquarium']
            city_count = df_aquarium.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            # Add numbers to the bars
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)


        elif viz_type == "Art Gallery":
            # Bar chart untuk Top 10 Cities dengan Art Galleries
            st.subheader("Top 10 Cities with Most Art Galleries")
            df_art_gallery = df1[df1['place_type'] == 'art_gallery']
            city_count = df_art_gallery.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            # Add numbers to the bars
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)


        elif viz_type == "Library":
            st.subheader("Top 10 Cities with Most Library")
            df_library = df1[df1['place_type'] == 'library']
            city_count = df_library.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            # Add numbers to the bars
            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)



        elif viz_type == "Movie Theater":
            st.subheader("Top 10 Cities with Most Movie Theater")
            df_movie_theater = df1[df1['place_type'] == 'movie_theater']
            city_count = df_movie_theater.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)


        elif viz_type == "Museum":
            st.subheader("Top 10 Cities with Most Museum")
            df_museum = df1[df1['place_type'] == 'museum']
            city_count = df_museum.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)


        elif viz_type == "Park":
            st.subheader("Top 10 Cities with Most Park")
            df_park = df1[df1['place_type'] == 'park']
            city_count = df_park.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)


        elif viz_type == "Restaurant":
            st.subheader("Top 10 Cities with Most Restaurant")
            df_restaurant = df1[df1['place_type'] == 'restaurant']
            city_count = df_restaurant.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)



        elif viz_type == "Shopping Mall":
            st.subheader("Top 10 Cities with Most Shopping Mall")
            df_shopping_mall = df1[df1['place_type'] == 'shopping_mall']
            city_count = df_shopping_mall.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)

        elif viz_type == "Stadium":
            st.subheader("Top 10 Cities with Most Stadium")
            df_stadium = df1[df1['place_type'] == 'stadium']
            city_count = df_stadium.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)

        elif viz_type == "Zoo":
            st.subheader("Top 10 Cities with Most Zoo")
            df_zoo = df1[df1['place_type'] == 'zoo']
            city_count = df_zoo.groupby("city").size().reset_index(name="count")
            top_10_cities = city_count.sort_values(by="count", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_10_cities['city'], top_10_cities['count'], color='c')

            for bar in bars:
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', color='black')

            ax.set_xlabel("Number of Tourist Attractions")
            ax.set_ylabel("City")
            ax.set_title("Top 10 Cities with Most Tourist Attractions")
            ax.invert_yaxis()  # Invert y-axis agar kota dengan jumlah terbanyak berada di atas
            st.pyplot(fig)
        
with ask_me:
    input_question = st.text_area("Enter your question here")
    with io.open("sample.txt", "r", encoding="utf-8") as f1:
        sample = f1.read()
    
    prompt = """
    You are a highly intelligent AI assistant, an expert in identifying and recommending travel destinations in Southeast Asia based on user input. Your task is to provide detailed responses in complete sentences, including recommendations, descriptions, and relevant information about various destinations.
    Please use the schema below to inform your responses, but focus on crafting engaging narratives rather than returning MongoDB aggregation pipeline queries. Make sure the website links can be accessed.
    Make sure your responses are in the form of sentences, not json.
    Ensure that you are only focused on Southeast Asia, not other parts of the world. If the user mentions countries or cities in other parts of the world display an error message.
    Ensure you replied with the same language as the user inputs. If user input the question with Bahasa Indonesia, you also replied with Bahasa Indonesia, if user input the question with english, make sure to answer it with english. The same also applies with other languages
    Make sure that when the user asks for budget/price-related questions give the prices in the country's currency.
    
    Here’s a breakdown of the schema with descriptions for each field:
    1. *_id*: Unique identifier for the place.
    2. *name*: Name of the place.
    3. *address*: Address of the place.
    4. *location*: The longitude (lng) and latitude (lat) of the place.
    5. *rating*: Measure of how good the place is out of 5.
    6. *types*: The categories the place is classified as, such as spa, restaurant, or amusement park.
    7. *city*: The city where the place is located.
    8. *place_type*: The main purpose of the place (e.g., amusement park).
    9. *business_status*: Indicates if the place is temporarily closed, permanently closed, or operating.
    10. *current_opening_hours*: The current opening hours of the place.
    11. *formatted_phone_number*: Contact phone number for the place.
    12. *photos*: Pictures of the place.
    13. *price_level*: Price category ranging from 0 (Free) to 4 (Very Expensive).
    14. *website*: Official website URL for more information.
    This schema provides a comprehensive view of the data structure for travel destinations, adding depth to your recommendations. Use the following sample questions to guide your responses.
    Sample question: {sample}
    Please provide clear, informative recommendations based on the user’s question: {question}
    """
    
    query_with_prompt = PromptTemplate(
        template=prompt,
        input_variables=["question", "sample"]
    )
    
    llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)
    
    # Button for submission
    button = st.button("Submit")
    if button and input_question:
        response = llmchain.invoke({
            "question": input_question,
            "sample": sample
        })

        # Check if the response is valid and display it
        if response and "text" in response:
            # Save the question and response to the conversation history
            st.session_state.conversation.append({"question": input_question, "response": response["text"]})

            # Display the conversation history
            for entry in st.session_state.conversation:
                st.write(f"**You:** {entry['question']}")
                st.write(f"**Bot:** {entry['response']}")
        else:
            st.error("No response from the model.")
