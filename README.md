



# A taste of the Milvus search engine

This project is intended to explore the vector similarity search engine Milvus by integrating it with a movie recommendation system. The main idea is that, after training the neural network for recommender, we extract all the movie embeddings into a Milvus collection. Then we, as some special users (cold starters for the recommendation system), can input our personal information and movie preferences into the neural network and update the parameters, after which we can extract our specific user embeddings.  Now we can query the Milvus collection to search for the movie embeddings most similar to our user embeddings, and the corresponding result movies would highly likely be great recommendations for us.

We would use the public [Movielens](https://grouplens.org/datasets/movielens/) dataset as source of data and PaddlePaddle as the deep learning framework. Since this project is mainly about Milvus, we won't go into great details about the neural network. Now, let's say, we already trained our model, and I would like the system to make some personal movie recommendations for me based on my preferences. Then I need to manually input my data into the system because obviously I'm not a user in the Movielens dataset. 

Personally I enjoy action movies a lot, so I pick the following movies and give them 5.0 (maximum) as the rating score.

Shanghai Noon (featured by Jackie Chan, movie id 3624), Romeo Must Die (featured by Jet Li, movie id 3452), Mission: Impossible 2 (featured by Tom Cruise, movie id 3623) and X-Men (featured by Hugh Jackman, movie id 3793).

```python
movie_titles = paddle.dataset.movielens.get_movie_title_dict()
movie_categories = paddle.dataset.movielens.movie_categories()
favorite_movies = [3624, 3452, 3793, 3623]

movie_data = []
for movie_id in favorite_movies:
    movie_info = collections.defaultdict(list)
    entry = paddle.dataset.movielens.movie_info()[movie_id]
    movie_info["movie_id"] = movie_id
    movie_info["rating"] = [5.0]
    for category in entry.categories:
        movie_info["category"].append(movie_categories[category])
    for word in entry.title.lower().split():
        movie_info["title"].append(movie_titles[word])
    movie_data.append([movie_info[key] for key in ["movie_id", "category", "title", "rating"]])
    print(entry)
    
personal_data = [0, 0, 1, 4]  # {"user_id": 0, "gender": "male", "age": "18-24", "job": "college/grad student"}
new_data = [personal_data + movie_data[i] for i in range(len(movie_data))]    
```

Here is what the result looks like:

```
<MovieInfo id(3624), title(Shanghai Noon ), categories(['Action'])>
<MovieInfo id(3452), title(Romeo Must Die ), categories(['Action', 'Romance'])>
<MovieInfo id(3793), title(X-Men ), categories(['Action', 'Sci-Fi'])>
<MovieInfo id(3623), title(Mission: Impossible 2 ), categories(['Action', 'Thriller'])>
```

Now let's push all the movie embeddings into the Milvus collections. 

```python
from milvus import Milvus, IndexType, MetricType
_HOST = '127.0.0.1'
_PORT = '19530'  # default value
table_name = 'recommender_demo'
milvus = Milvus(_HOST, _PORT)

param = {
    'collection_name': table_name,
    'dimension': 200,
    'index_file_size': 1024,  # optional
    'metric_type': MetricType.IP  # optional
}

milvus.create_collection(param)
```

After that, we can query from the collection based on our user embeddings

```python
Status(code=0, message='Create table successfully!')
rows in table recommender_demo: 3883
Top      Ids     Title   Score
0        3030    Yojimbo         2.9444923996925354
1        3871    Shane           2.8583481907844543
2        3467    Hud     2.849525213241577
3        1809    Hana-bi         2.826111316680908
4        3184    Montana         2.8119677305221558
```

