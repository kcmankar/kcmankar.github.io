---
layout: post
title:  "Deep Learning Movie Recommendations based on plot similarity"
date:   2021-11-14
categories: [Deep Learning, NLP, Movie Recommendation]
---
## Deep Learning Movie Recommendations based on plot similarity using Setence Transformer

---

1. Introduction:

In this article we are going to build a movie recommendation system, but not based on score we are going to recommend movies based on the similarity of  their plot or summaries.

We are using this [dataset](https://www.kaggle.com/jrobischon/wikipedia-movie-plots) for getting our summary of movie or plot of the movie.

2. Process:

    1.  We somehow need to convert this plot of the movie into a vector representation, then we can find similarity between these vectors.
    2. For this we are using sentence-transformer library [sbert](https://www.sbert.net/)
    3. This library uses Siamese networks to find the similarity between two similar sentences.
    4. We will then convert all the plots of movies in the data-set into a vector embedding.
    5. Then we will find similar embedding to a given movie by using knn. [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

3. Why vectors:
    1. Once our plot summaries are converted to vectors, the plots whose summaries are semantically similar, their vector representations will be closer to each other.
    2. We can then use this property to train classifiers like knn or find the cosine similarity.

4. Sentence Bert (sbert)

    1. Sentence Bert or Sbert was introduced in this paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
    2. Here they train pairwise sentences and feed them into a [Siamese network](https://en.wikipedia.org/wiki/Siamese_neural_network). Siamese Networks take two distinct input but their weights are tied together.

    ![sbert](https://imgur.com/m7nXRwA.png)

   3. The above is the diagram of sbert is as mentioned in their paper  where we are giving two inputs and then their similarity is calculated. [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). When two vectors are close their similarity is one and when they are apart their cosine similarity is zero.

  4. The loss is calculated by [Triplet Loss](https://en.wikipedia.org/wiki/Triplet_loss) which is used in siamese networks.

  > max( \|\| s <sub>a</sub> − s <sub>p</sub> \|\| − \|\| s <sub>a</sub>  − s <sub>n</sub> \|\| + ep, 0)

  5. Given an anchor sentence a, a positive sentence p, and a negative sentence n, triplet loss tunes the network such that the distance between a and p is smaller than the distance between a and n. With s<sub>x</sub> the sentence embedding for a/n/p, || · || a distance metric and margin ep. Margin ep ensures
  that s<sub>p</sub> is at least ep closer to s<sub>a</sub> than s<sub>n</sub> . As metric the authors of paper used Euclidean distance and we set ep = 1.




Here is a very very simple website I made for movie reccomendation based on plot [site](https://kcmankar.github.io/website_movie_recommendations_sbert.github.io/).
Jupyter notebook: [notebook](https://github.com/kcmankar/MovieRecomendationsDL)

---

#### Code

Lets install the sentence-transformer package


```python
!pip install -U sentence-transformers
```



```python
path = "__path__to_data__"
```

For movie recommendations


```python
import pandas as pd #import pandas
dfM = pd.read_csv(path)
```

Lets take a look at our data



```python
dfM.head()
```



<div>
  <!--
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Title</th>
      <th>Plot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Kansas Saloon Smashers</td>
      <td>A bartender is working at a saloon, serving dr...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Love by the Light of the Moon</td>
      <td>The moon, painted with a smiling face hangs ov...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>The Martyred Presidents</td>
      <td>The film, just over a minute long, is composed...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Terrible Teddy, the Grizzly King</td>
      <td>Lasting just 61 seconds and consisting of two ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Jack and the Beanstalk</td>
      <td>The earliest known adaptation of the classic f...</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfM.describe()
```




<div>
  <!--
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
    .dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>
-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>34886.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17442.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10070.865082</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8721.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17442.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26163.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>34885.000000</td>
    </tr>
  </tbody>
</table>
</div>



There are total 34,886 movie entries


```python
dfM['Plot'].iloc[0]
```




    "A bartender is working at a saloon, serving drinks to customers. After he fills a stereotypically Irish man's bucket with beer, Carrie Nation and her followers burst inside. They assault the Irish man, pulling his hat over his eyes and then dumping the beer over his head. The group then begin wrecking the bar, smashing the fixtures, mirrors, and breaking the cash register. The bartender then sprays seltzer water in Nation's face before a group of policemen appear and order everybody to leave.[1]"



Looks like we need some data cleaning, there are reference numbers like [1] [2] from wikipedia, we want to remove that.


```python
import re #for data cleaning
```

Writing a function that takes a row of dataframe dfM as argument and then cleans the data


```python
def cleanRow(row):
  cleanData = re.sub('\[\d+\]','',row['Plot']) #this finds all the [number] in string and replaces it with empty string ''
  return cleanData
```

Lets test in on first entry of our dataframe


```python
cleanRow(dfM.iloc[0])
```




    "A bartender is working at a saloon, serving drinks to customers. After he fills a stereotypically Irish man's bucket with beer, Carrie Nation and her followers burst inside. They assault the Irish man, pulling his hat over his eyes and then dumping the beer over his head. The group then begin wrecking the bar, smashing the fixtures, mirrors, and breaking the cash register. The bartender then sprays seltzer water in Nation's face before a group of policemen appear and order everybody to leave."



Applying the cleanRow() function to all the rows in our dataframe


```python
dfM['cleanPlot'] = dfM.apply(lambda row: cleanRow(row),axis = 1)
```

As you can see all the reference numbers are gone. Now the rest is very simple, we just need to pass these plots into our sentence transformer model


```python
dfM.head()
```



<div>
<!--
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Title</th>
      <th>Plot</th>
      <th>cleanPlot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Kansas Saloon Smashers</td>
      <td>A bartender is working at a saloon, serving dr...</td>
      <td>A bartender is working at a saloon, serving dr...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Love by the Light of the Moon</td>
      <td>The moon, painted with a smiling face hangs ov...</td>
      <td>The moon, painted with a smiling face hangs ov...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>The Martyred Presidents</td>
      <td>The film, just over a minute long, is composed...</td>
      <td>The film, just over a minute long, is composed...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Terrible Teddy, the Grizzly King</td>
      <td>Lasting just 61 seconds and consisting of two ...</td>
      <td>Lasting just 61 seconds and consisting of two ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Jack and the Beanstalk</td>
      <td>The earliest known adaptation of the classic f...</td>
      <td>The earliest known adaptation of the classic f...</td>
    </tr>
  </tbody>
</table>
</div>



Importing our model


```python
from sentence_transformers import SentenceTransformer
```

We are using a pre-trained model.
Here is the list of all pre-trained models available in the library
[models](https://www.sbert.net/docs/pretrained_models.html)


```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```

Using GPU


```python
model = model.cuda()
```

This is the step where we get our embeddings in return, The sentence-transformer package made it very easy to compute all these vector embeddings

The model can take some time to process all 34,886 embeddings


```python
embeds = model.encode(dfM['cleanPlot'])
```

As you can see all 34,886 movie plots are converted to embeddings of 384 shape.


```python
embeds.shape
```




    (34886, 384)



Example


```python
embeds[0]
```




    array([-2.12161932e-02,  3.00669353e-02, -2.20030937e-02, -4.86579239e-02,
            4.65949513e-02,  1.39347359e-03,  1.11619830e-01, -1.05095394e-01,
            1.64841451e-02, -6.58066571e-02,  4.28134426e-02, -8.68948922e-02,
           -6.84939772e-02,  4.01585288e-02, -4.27194089e-02, -2.80906223e-02,
           -5.49537353e-02, -1.82290319e-02,  1.69992838e-02, -1.23173064e-02,
           -7.43851960e-02,  1.77508891e-02, -2.53297910e-02,  4.20343839e-02,
           -1.91297680e-02, -5.33761717e-02,  7.73109645e-02,  3.12473997e-02,
           -1.56767934e-03, -1.35527225e-02,  7.67379850e-02, -8.90569668e-03,
            1.67426597e-02,  5.27440123e-02, -5.66607080e-02, -3.37757282e-02,
            8.18887725e-02,  4.01530862e-02,  7.09116757e-02,  1.03832453e-01,
            1.21762659e-02, -3.67102958e-02, -5.90380980e-03,  2.12801900e-02,
            8.85269195e-02,  6.31055385e-02,  2.16334965e-03,  5.64815775e-02,
            3.59476879e-02, -1.78708490e-02, -1.78310033e-02, -3.49012762e-02,
            3.46690491e-02,  9.72064678e-03,  7.44531751e-02, -1.16908140e-01,
            3.73819843e-02, -5.11367582e-02,  4.22328897e-02,  3.78267169e-02,
            1.63187031e-02,  5.04687522e-03,  2.46317796e-02,  3.04010902e-02,
            1.50619701e-01, -3.56495380e-02, -5.36321253e-02,  8.55351016e-02,
           -3.40821519e-02,  5.93329929e-02,  8.72676633e-03, -7.47592598e-02,
           -4.54675360e-03, -3.00062858e-02, -6.03364520e-02, -1.04080379e-01,
           -3.90730537e-02, -5.85835893e-03, -2.94597242e-02,  3.58620211e-02,
           -1.06213533e-03, -9.85332131e-02,  4.84071067e-03,  2.71317679e-02,
           -4.01531942e-02,  2.14641970e-02, -2.77724639e-02, -3.90547998e-02,
           -2.76160473e-03,  2.16669478e-02, -8.57357532e-02,  2.96338787e-03,
            6.09617680e-02, -5.41299097e-02,  6.71687871e-02,  2.81013753e-02,
            5.57972118e-03, -5.57486415e-02, -1.56321712e-02,  1.21677339e-01,
           -4.05702787e-03,  5.93564734e-02, -6.29780367e-02, -1.05312131e-01,
            4.67817038e-02, -4.07967484e-03, -3.97506095e-02,  5.87533079e-02,
            1.52943190e-02,  7.57013028e-03,  4.88090850e-02,  1.25311660e-02,
           -4.22780551e-02, -1.42411357e-02,  2.57545151e-02, -8.54073616e-04,
           -1.66485645e-02, -5.17073972e-03, -3.01484037e-02, -1.58574600e-02,
            5.78604825e-02,  1.02385513e-01, -8.68215635e-02,  1.02301970e-01,
           -3.84437330e-02,  5.46138026e-02,  7.97055941e-03,  7.63060632e-34,
           -2.24114433e-02, -9.86156464e-02,  1.81630906e-02,  1.85835790e-02,
            1.41446248e-01, -3.30856144e-02, -4.57095578e-02, -4.95120557e-03,
           -7.19193444e-02, -2.60208230e-02, -3.07247434e-02, -1.18188195e-01,
           -8.46335888e-02,  2.52517289e-03, -1.09469739e-03, -2.53156642e-03,
           -3.50408666e-02, -6.56299759e-03, -8.39504506e-03, -3.71534228e-02,
           -1.75888669e-02,  3.65182981e-02, -6.39306009e-02,  7.10733980e-02,
           -7.85688162e-02,  3.61557642e-04,  4.50520813e-02,  1.12867346e-02,
            1.08463988e-01, -1.37030007e-02, -3.18820998e-02,  1.30245566e-01,
            1.42632015e-02,  7.63415964e-03,  6.20989203e-02, -2.29522195e-02,
           -5.94718673e-04,  2.48584021e-02, -3.66474204e-02,  3.65234772e-03,
           -1.44974768e-01,  5.28973863e-02,  4.52945791e-02,  6.01990940e-03,
           -7.22688287e-02, -6.30701473e-03, -7.26218373e-02,  1.22512756e-02,
            1.18219871e-02,  9.79589950e-03, -3.75862699e-03,  2.97069009e-02,
            7.04168975e-02,  3.98568250e-02,  4.75154631e-02, -1.91271100e-02,
           -7.73595832e-03,  1.64352413e-02,  2.35282648e-02, -2.18547750e-02,
            3.24454792e-02,  1.68896634e-02, -6.22175913e-03,  2.84431167e-02,
            3.44861448e-02,  2.11522710e-02, -8.96190293e-03,  5.72916027e-03,
            5.96740954e-02, -9.62257907e-02, -5.94271943e-02,  9.25972834e-02,
            1.03072040e-02, -1.71594545e-02, -4.75794226e-02,  5.50725050e-02,
            6.05137199e-02,  1.04070073e-02, -6.05522767e-02, -1.14610931e-02,
           -1.43278120e-02, -6.87917247e-02,  7.82089531e-02, -2.73606163e-02,
           -2.98696309e-02, -6.65436313e-02,  5.31267785e-02, -1.70934767e-01,
           -2.89114658e-03,  7.08695799e-02, -7.69388378e-02, -4.90841046e-02,
            3.19597349e-02,  2.79412735e-02,  2.70547923e-02, -2.49645172e-33,
            5.78023717e-02, -7.34835789e-02, -1.85252670e-02,  1.71814058e-02,
            6.99770972e-02, -6.29613027e-02,  9.09064338e-03,  2.55967733e-02,
            4.26787511e-02,  1.70449633e-02, -5.08790165e-02,  2.95289103e-02,
            1.85025502e-02,  5.47559336e-02,  5.58700599e-02, -2.27008015e-02,
            8.00882801e-02,  7.22413138e-02, -2.39725057e-02, -1.75498109e-02,
            5.31134158e-02,  3.13796895e-03,  6.41072467e-02, -3.85736785e-04,
           -1.35111762e-02,  4.10360955e-02,  3.48888487e-02,  1.92015972e-02,
           -5.18765077e-02, -6.70807734e-02,  1.68664183e-03, -5.37328683e-02,
            2.93370076e-02, -3.32006142e-02, -1.11329913e-01,  9.34464335e-02,
            5.63973561e-02,  2.22563073e-02, -2.59201266e-02, -3.33883148e-03,
            2.87980195e-02, -2.21632607e-02, -7.26132765e-02,  6.44899765e-03,
           -5.15565695e-03, -1.16213514e-02, -4.05008271e-02, -6.99214311e-03,
           -5.04501648e-02, -6.63406253e-02,  8.11421964e-03, -2.93816184e-03,
           -4.67047468e-02,  3.32037807e-02, -2.27790102e-02, -4.15245183e-02,
            9.77905467e-02, -1.03601329e-02, -8.40062872e-02, -2.80099679e-02,
           -6.11843169e-02,  1.86135340e-02, -5.19332625e-02,  3.63485143e-02,
           -1.71496104e-02, -1.43257439e-01, -4.53802124e-02,  6.28079921e-02,
            6.16132244e-02, -5.45926504e-02,  5.32125607e-02, -7.56241530e-02,
           -3.51599343e-02,  2.52744406e-02,  1.98267363e-02, -8.18747189e-03,
            4.13149782e-03, -2.47631762e-02, -1.34273255e-02, -1.88834481e-02,
           -8.36074501e-02, -5.29852472e-02,  9.59480554e-03,  1.48206130e-01,
            3.39602418e-02,  3.69826169e-03,  1.82431370e-01,  4.77819033e-02,
            1.27401119e-02,  6.84745833e-02,  4.29175794e-02,  5.38143422e-03,
           -2.99710850e-03, -1.81393530e-02,  5.16400635e-02, -4.52007143e-08,
           -7.68637732e-02, -1.69737265e-02, -4.30625565e-02,  2.08788402e-02,
           -1.04042813e-02, -2.27654558e-02,  2.11840980e-02,  1.58817414e-03,
           -4.82034646e-02, -2.78107990e-02, -5.23242541e-02,  4.29890677e-02,
            2.07168283e-03, -2.70264223e-02, -4.23216596e-02, -1.93599407e-02,
            2.19480824e-02, -2.70004431e-03, -5.46910837e-02, -3.04673836e-02,
           -1.29334349e-02, -4.10066888e-04,  3.01454589e-02,  3.05256043e-02,
           -9.82224643e-02, -1.65465742e-03, -4.35108654e-02,  4.69615050e-02,
           -2.35710703e-02,  9.44132283e-02,  7.22733443e-04,  2.42961440e-02,
           -6.73146844e-02, -3.48870456e-02, -4.11168300e-02, -4.61550150e-03,
           -6.66172653e-02, -1.88828725e-02,  3.81046534e-02, -5.35625927e-02,
           -4.48297784e-02, -7.92523474e-02, -2.86041247e-03, -2.63251476e-02,
            1.77027714e-02,  4.87428345e-03,  2.56778002e-02,  5.99750429e-02,
            1.48795592e-02,  3.10689136e-02,  7.98457190e-02, -9.92197264e-03,
            4.69778990e-03,  3.75730954e-02,  4.95770089e-02, -4.56519350e-02,
           -3.13913710e-02,  1.40139274e-02, -1.23726502e-02, -4.06946428e-02,
            1.14586137e-01,  4.19088086e-04,  4.95863743e-02,  2.86626667e-02],
          dtype=float32)



If you wish you can store the embeddings in our dataframe


```python
dfM['embeddings'] = embeds.tolist()
```

Now lets predict some movie recommendations, we are going to train our knn


```python
from sklearn.neighbors import NearestNeighbors
```


```python
import numpy as np
```


```python
neighbors = NearestNeighbors(n_neighbors=15)
```


```python
neighbors.fit(dfM['embeddings'].tolist())#fitting our data
```




    NearestNeighbors(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=15, p=2,
                     radius=1.0)



KNN return the distance and the index of the next 15 closest embeddings. So, lets make a dict which takes the index and returns the name of movie


```python
idx_dict = dfM['Title'].to_dict()#making an index dictionary  - index-> title of movie
```


```python
idx_dict[0]
```




    'Kansas Saloon Smashers'



We need the knn to predict all the closest 15 recommendations to a given movie embeddings, we will pass the embeddings one by one and then save the 15 recommended movies index

Here is an example


```python
neighbors.kneighbors(np.array(embeds[0]).reshape(1,-1))
```




    (array([[0.        , 0.8927627 , 0.97209753, 0.98154652, 0.98469113,
             0.99231732, 0.99920433, 1.00403609, 1.01327464, 1.01370713,
             1.0170882 , 1.01793011, 1.01885963, 1.02645933, 1.02752993]]),
     array([[    0,  9231, 22037,    98,   172, 10478,  8426,  1269, 25738,
               431, 21299, 34853, 26449,   231, 28327]]))



Here the second array [ 0,  9231, 22037,    98,   172, 10478,  8426,  1269, 25738, 431, 21299, 34853, 26449,   231, 28327] are the index of similar movies

So the 15 movies similar to movie present at index 0 i.e "Kansas Saloon Smashers" are present at index 0, 9231, 22037, 98, 172, 10478, 8426, 1269, 25738, 431, 21299, 34853, 26449, 231, 28327

Lets print their names


```python
recList = [    0,  9231, 22037,  98,   172, 10478,  8426,  1269, 25738, 431, 21299, 34853, 26449,   231, 28327]
print(f"movies similar to {idx_dict[0]} are following: ")
for i in recList:
  print(idx_dict[i])
```

    movies similar to Kansas Saloon Smashers are following:
    Kansas Saloon Smashers
    Dixie Dynamite
    FUBAR: The Movie
    The Rounders
    In Again, Out Again
    8 Million Ways to Die
    Childish Things
    Broadway to Cheyenne
    Amanaat
    The Frozen North
    U.F.O.
    Black and White
    Johnny Gaddar
    Out West
    Idiots


You can see the most similar to the movie is the movie itself, that's why their embeddings are close to each other and knn return it's index. The rest are in increasing order of their distances

Lets write a function to apply to all embeddings saved in our dataframe


```python
def recFun(row):#return similar movie indexes
    embeds = row['embeddings']
    dist,idx = neighbors.kneighbors(np.array(embeds).reshape(1,-1))
    return idx[0][1:] #since the 0th will always be the same movie
```

Apply to the dataframe, this also can take a lot of time (upto 20 mins in my case) since we are doing it for 34,886 movies and the knn returns 15 recommendation for each.


```python
dfM['reccomendations'] = dfM.apply(lambda row: recFun(row),axis=1)
```

Here are our reccomendations, they are index of that movie name in reccomendation column


```python
dfM.head()
```




<div>
  <!--
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Title</th>
      <th>Plot</th>
      <th>cleanPlot</th>
      <th>embeddings</th>
      <th>reccomendations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Kansas Saloon Smashers</td>
      <td>A bartender is working at a saloon, serving dr...</td>
      <td>A bartender is working at a saloon, serving dr...</td>
      <td>[-0.021216193214058876, 0.03006693534553051, -...</td>
      <td>[9231, 22037, 98, 172, 10478, 8426, 1269, 2573...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Love by the Light of the Moon</td>
      <td>The moon, painted with a smiling face hangs ov...</td>
      <td>The moon, painted with a smiling face hangs ov...</td>
      <td>[0.011549428105354309, 0.07491016387939453, 0....</td>
      <td>[14371, 28968, 28967, 23560, 27556, 7518, 8604...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>The Martyred Presidents</td>
      <td>The film, just over a minute long, is composed...</td>
      <td>The film, just over a minute long, is composed...</td>
      <td>[-0.019372664391994476, 0.04275871440768242, -...</td>
      <td>[15550, 32883, 3, 39, 14897, 17522, 9139, 1038...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Terrible Teddy, the Grizzly King</td>
      <td>Lasting just 61 seconds and consisting of two ...</td>
      <td>Lasting just 61 seconds and consisting of two ...</td>
      <td>[0.018185125663876534, 0.022814400494098663, 0...</td>
      <td>[2, 19433, 14148, 2702, 9297, 12015, 431, 1261...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Jack and the Beanstalk</td>
      <td>The earliest known adaptation of the classic f...</td>
      <td>The earliest known adaptation of the classic f...</td>
      <td>[-0.03626694902777672, -0.011878693476319313, ...</td>
      <td>[16572, 5651, 6246, 16573, 4370, 33672, 598, 1...</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have recommendation for every movie in our dataset, we can implement a simple search function and return the movie recommendations


```python
def recs(dfM,movie,idx_dict):
  reccoms = dfM[dfM['Title'].str.contains(movie, na=False, case=False)].iloc[0]['reccomendations'] #finds the movie passes in dataframe
  #print(reccoms)
  if (len(reccoms)>0):
    for i in reccoms:
      print(idx_dict[i])#convert index to movie name
  else:
    print("movie not in database")

```

Movies whose plot is similar to "The godfather" are


```python
recs(dfM,"The godfather",idx_dict)
```

    A Bronx Tale
    The Godfather Part II
    The Sicilian
    The Freshman
    Donnie Brasco
    Jane Austen's Mafia!
    Jersey Boys
    Avenging Angelo Baby Beethoven Baby Newton
    King of New York
    Black Hand
    Carlito's Way
    Miller's Crossing
    The Brothers Rico
    Family Business


Movies whose plot is similar to "ast five" are


```python
recs(dfM,"fast five",idx_dict)
```

     The Fast and the Furious
    The Fate of the Furious
    2 Fast 2 Furious
    Restraint
    Fast & Furious
    Collateral
    Gunmen
    Drive
    Beverly Hills Cop III
    The Fast and the Furious
    Dawn of the Dead
    Getaway
    Pulp Fiction
    The Courtship of Andy Hardy



```python
id_toRecs = dfM['reccomendations'].to_dict() #dictionary from index to reccomendations
```

### Here is a very very simple website I made for movie reccomendation based on plot [site](https://kcmankar.github.io/website_movie_recommendations_sbert.github.io/).

### The code for website is [here](https://github.com/kcmankar/website_movie_recommendations_sbert.github.io)

#### Jupyter notebook: [notebook](https://github.com/kcmankar/MovieRecomendationsDL)
