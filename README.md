



# A taste of the Milvus search engine

|          | Config                                                       |
| :------: | :----------------------------------------------------------- |
|   CPU    | Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz                     |
|   GPU    | GeForce GTX 1050 Ti 4GB                                      |
|  Memory  | 32GB                                                         |
|    OS    | Ubuntu 18.04                                                 |
| Software | Milvus 0.10.6 <br />pymilvus 0.4.0  <br />PaddlePaddle 2.0.1  <br />Docker 20.10.2 <br> |

This project is intended to explore the vector similarity search engine [Milvus](https://milvus.io/) by integrating it with a movie recommendation system. The main idea is that, after training the neural network for recommender, we extract all the movie embeddings into a Milvus collection. Then we, as some special users (cold starters for the recommendation system), can input our personal information and movie preferences into the neural network and update the parameters, after which we can extract our specific user embeddings.  Now we can query the Milvus collection to search for the movie embeddings most similar to our user embeddings, and the corresponding result movies would highly likely be great recommendations for us.

We would use the public [Movielens](https://grouplens.org/datasets/movielens/) dataset as source of data and PaddlePaddle as the deep learning framework. Since this project is mainly about Milvus, we won't go into great details about the neural network. The network architecture basically looks like the following, and we mainly interested in making use of the user embeddings and movie embeddings. 

<img src="https://upload-images.jianshu.io/upload_images/10870953-4bb7caf9311cfa65.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" alt="architecture" style="zoom:60%;" />

Now that we already trained our model, I would like the system to make some personal movie recommendations for me based on my preferences. Then I need to manually input my data into the system because obviously I'm not a user in the Movielens dataset. Since personally I enjoy action movies a lot,  I pick the following movies and give them 5.0 (the maximum) as the rating score:  Shanghai Noon (featured by Jackie Chan, movie id 3624), Romeo Must Die (featured by Jet Li, movie id 3452), Mission: Impossible 2 (featured by Tom Cruise, movie id 3623) and X-Men (featured by Hugh Jackman, movie id 3793).

```python
import paddle
movie_titles = paddle.dataset.movielens.get_movie_title_dict()
movie_categories = paddle.dataset.movielens.movie_categories()
my_favorite_movies = [3624, 3452, 3793, 3623]

movie_data = []
for movie_id in my_favorite_movies:
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

Here is what each movie record originally looks like:

```
<MovieInfo id(3624), title(Shanghai Noon ), categories(['Action'])>
<MovieInfo id(3452), title(Romeo Must Die ), categories(['Action', 'Romance'])>
<MovieInfo id(3793), title(X-Men ), categories(['Action', 'Sci-Fi'])>
<MovieInfo id(3623), title(Mission: Impossible 2 ), categories(['Action', 'Thriller'])>
```

Now I can extract my unique user embedding from the neural network model:

```python
import paddle.fluid as fluid
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
inferencer, _, _ = fluid.io.load_inference_model(model_path, exe)
user_embedding = exe.run(inferencer, feed=new_data, fetch_list=fetch_targets)[1]
```

Next we need to extract movie embeddings for all the movies in the dataset and push them into a Milvus collection.

```python
from milvus import Milvus, MetricType
_HOST = '127.0.0.1'
_PORT = '19530' 
table_name = 'recommendation_system'
milvus = Milvus(_HOST, _PORT)

param = {
    'collection_name': table_name,
    'dimension': 200,
    'index_file_size': 1024,  
    'metric_type': MetricType.IP
}

milvus.create_collection(param)
status, ids = milvus.insert(collection_name=table_name, records=movie_embeddings)
```

At this point, we have obtained the user embedding and movie embeddings. In order for the system to make exciting  recommendations for me, we simply query the Milvus collection for the movie embeddings most similar to my user embedding. Note that we need to filter the 4 action movies I manually input to the system earlier , obviously you don't want to recommend movies to users that they have already watched.

```python
param = {
    'collection_name': table_name,
    'query_records': user_embedding,
    'top_k': 10,
    'params': {'nprobe': 16}
}
status, results = milvus.search(**param)
print(status)

print("ID\t", "Title\t", "Similarity")
for result in enumerate(results[0]):
    if result.id not in my_favorite_movies:
        title = paddle.dataset.movielens.movie_info()[int(result.id)].title
        print(result.id, "\t", title, "\t", round(float(result.distance) * 5, 3))
```

Hurray! We successfully obtain the result instantaneously and these are all great action movies! Thanks to the powerfulness of Milvus, we can easily make recommendations to users based on their preferences.

```bash
Status(code=0, message='Search vectors successfully!')
ID      Title                        Similarity
3184    Rush Hour                    2.981
2628    Star Wars: Episode I         2.924
2571    The Matrix                   2.858
3864    Godzilla 2000                2.839 	
3467    The Mask of Zorro            2.816		 
1562    Batman & Robin               2.783
```



## Thoughts

It has been super easy and painlessly to set up and use Milvus. It is very user-friendly to support both Numpy ndarray and Python list as the data type for search vectors. And from my hands on experience with Milvus, I believe it can readily scale to much larger datasets. 

### Use case

I notice that the blogs and documents from both Zilliz and Milvus have covered a lot of use cases in the industry, including image search, text search, molecular structural similarity analysis, audio data processing and so on. **I would like to point out one more potential use case, that is, DNA sequences analysis**. Evaluating the similarity between DNA sequences is crucial for the sequence analysis and its applications include discovering the evolutionary relationship between species and searching similar sequences in huge databases.  Researchers generally apply sequence alignment algorithm to compute the similarity between two sequences, which, unfortunately, depends on the orderings of the nucleotides and may be computationally prohibitive. Recently, many alignment-free algorithms have been proposed, for example, construct representative vectors based on frequency patterns and entropy. I believe Milvus could benefit from these algorithms, because they provide effective ways to represent DNA sequence as vectors, and then we can take advantage of the powerful Milvus engine to search for similar DNA sequences.

### Implementation suggestion

- It would be great if Milvus can support customized distance metric. Currently the choice for distance metrics is limited. Of course the Euclidean distance, Inner product and Jaccard similarity would be enough for most situations, but it's possible that users might want to use his own definition of distance, for example, weighted Euclidean distance where some dimension might be more important than the other dimensions. I understand that in the embeddings generated from deep learning models, all dimensions have the same scale and no any dimension is more important than the other, so a weighted distance is unnecessary. However, I think Milvus should not always be bundled with neural network embeddings.
- Currently the records to be searched must have the same dimension, otherwise the Milvus client would throw a `ParamError`. However, in some cases, the vectors might have different sizes. For example, in the DNA sequences analysis I mentioned above, the two DNA sequence vectors would not necessarily have the same size. It would be great if Milvus can support variable input dimension. Of course the standard distance metrics like Euclidean distance and Inner product would not work for vectors of different sizes, that's why I suggest supporting customized distance metric above.



Overall, Milvus is a remarkable and powerful open-source engine, helping so many companies and individuals to gain actionable insights from the vast amount of unstructured data (I believe this will be the main focus of data science in the next five years) instead of letting such data go wasted and ignored. Milvus indeed contributes a lot to the society as a whole. And I'd be honored if I had a chance to join this great community.











