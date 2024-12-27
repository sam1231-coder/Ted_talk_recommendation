from flask import Flask, render_template, request
import pickle
import re
import pandas as pd

# Read the TED talks dataset
df = pd.read_csv('Data/TED.csv')

# Ensure the title is in lowercase
df['title'] = df['title'].str.lower()

# Load the cosine similarity matrix
cosine_sim = pickle.load(open('cs.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_ted_talk = request.form['ted-talk'].lower().strip()  # Strip extra spaces
        user_ted_talk_no_tags = remove_tags(user_ted_talk)
        recommendation = recommend_talks(user_ted_talk_no_tags)
        
        if recommendation:  # Check if recommendation list is not empty
            rec = ''.join([f"<li><a href='{talk[1]}' target='_blank'>{talk[0]}</a></li>" for talk in recommendation])  # Bullet list with links
        else:
            rec = "No recommendations found. Please check the input."
        
        return render_template('home.html', prediction_text=rec)

def recommend_talks(name):
    indices = pd.Series(df['title'])
    talks = []
    
    # Check if the name exists in the indices
    if name in indices.values:
        # Get the index of the TED talk that matches the user input
        idx = indices[indices == name].index[0]
        sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
        
        # Get top 10 recommended talks, starting from the second one
        top_10 = sort_index.iloc[1:10]  # Exclude the first one since it matches the user input
        
        for i in top_10.index:
            # Append the title and URL of the recommended talks
            talks.append((indices[i], df.loc[i, 'url']))
    
    return talks

def remove_tags(string):
    # Updated regex to remove HTML tags
    result = re.sub('<.*?>', '', string)
    return result

if __name__ == '__main__':
    app.run(debug=True)
