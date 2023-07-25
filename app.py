from flask import Flask, render_template, jsonify
import pandas as pd
from tool_box import *



df = pd.read_csv('./data/merged.csv')
df.fillna("-",inplace=True)
links=pd.read_csv('./data/links_poster.csv')
movies =pd.read_csv('./data/movies.csv')
similarity_matrix = pd.read_csv('./data/similarityDF.csv', index_col="movieId").values


# Read the CSV file using pandas

app = Flask(__name__)



@app.route('/')
def user():
    return render_template('user.html')

@app.route('/item')
def item():
    
    return render_template('item.html')

@app.route('/user_history/<user_id>')
def user_history(user_id):
    # Read user history data from merged.csv file
    data = df[df['userId'] == int(user_id)]
    

    
    # Modify the 'poster' field in the DataFrame
    # data['poster'] = data['imdbId'].apply(add_poster)
    
    # Convert DataFrame to dictionary
    user_history_data = data.to_dict('records')

    return jsonify({'user_history': user_history_data})

@app.route('/user_recomand/<user_id>')
def user_recomand(user_id):
    # Read user history data from merged.csv file
    data = recommend_top_k_movies(int(user_id))
    data = pd.merge(data, links[["movieId","poster"]], on=['movieId'])

    
    # Modify the 'poster' field in the DataFrame
    # data['poster'] = data['imdbId'].apply(add_poster)
    
    # Convert DataFrame to dictionary
    user_recomand_data = data.to_dict('records')

    return jsonify({'user_recomand': user_recomand_data})

@app.route('/item_sim/<item_name>')
def item_sim(item_name):
    # Read user history data from merged.csv file
    data = movies[movies['title'] == item_name]
    mov_id=data["movieId"].iloc[0]
    sim=recommend_similar_movies(similarity_matrix,mov_id)
    sim = pd.merge(sim, links[["movieId","poster"]], on=['movieId'])
    sim = pd.merge(sim, movies[["movieId","title","genres"]], on=['movieId'])


    
    # Modify the 'poster' field in the DataFrame
    # data['poster'] = data['imdbId'].apply(add_poster)
    
    # Convert DataFrame to dictionary
    user_history_data = sim.to_dict('records')

    return jsonify({'user_history': user_history_data})

if __name__ == '__main__':
    app.run(debug=True)
