---
layout: post
title:  "Deep Learning Multilingual Sentence Similarity"
date:   2021-11-14
categories: [Deep Learning, NLP, Sentence Similarity]
---

### Understanding Multilingual Sentence Similarity and Knowledge Distillation with Sentence-Transformers
***

Sentence Similarity is one of the most important tasks in Natural Language Processing. Here we will look at what exactly is semantic similarity, how a method called Multilingual Knowledge Distillation [paper](https://arxiv.org/abs/2004.09813 "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation") is used to create sentence embedding where sentences which "mean" the same things are closer to each other in vector space.
#### 1. What is Semantic Similarity?<br>
[Semantic similarity](https://en.wikipedia.org/wiki/Semantic_similarity "Wikipedia") is a metric defined over a set of documents or terms, where the idea of distance between items is based on the likeness of their meaning or semantic content as opposed to lexicographical similarity.This means that sentences which have the same meaning should be more closer to each other in vector space.
#### 2. Existing Sentence Embeddings<br>
There are many multilingual language models which can be used to generate sentence embeddings but they don't necessarily put the sentence which mean the same things across multiple languages closer to each other in vector space. They sometimes put sentences in the same language closer to each other than sentences than carry the same meaning. We will see later with [mBERT](https://huggingface.co/bert-base-multilingual-cased "huggingface mbert") or multilingual BERT which was trained in 104 languages, it can produce embeddings but not necessarily put sentences with same meaning closer to each other across languages.
#### 3. Knowledge Distillation<br>
![Knowledge distillation implements two options for creating the student mode](https://dair.ai/images/summary-making-monolingual-senence-embeddings-multilingual-using-knowledge-distillation/training-process-schematic.png "Given parallel data e.g. English and German, train the student model such that the produced vectors for
the English and German sentences are close to the teacher English sentence vector")

In this method we take two models one is a Teacher model which is trained in any one language and a Student model with a multilingual vocabulary. The Student model is trained on parallel data and the task is to minimize the distance between the embeddings produced by the Teacher Model and the Student model.

Here as you can see in the diagram the Teacher model is given "Hello World" as an input, it outputs some vector embeddings. The Student model is given two inputs one is the English sentence "Hello World" and with that it's parallel translation "Hallo Welt". The student model has multilingual vocabulary so it will output different embeddings for English sentence "Hello World" and for German "Hallo Welt". The Student model is then trained to minimize the distance between it's output of English sentence and the Teacher's output embeddings of English Sentence. It is also trained to minimize the distance between it's output embeddings for German sentence "Hallo Welt" and Teachers output of English sentence.

To go more in detail I will quote some portion mentioned in original [paper](https://arxiv.org/abs/2004.09813 "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation")

> We require a teacher model M , that maps sentences in one or more source languages s to a dense vector space. Further, we need parallel (translated) sentences ((s<sub>1</sub> , t<sub>1</sub> ), ..., (s<sub>n</sub> , t<sub>1</sub>)) with s<sub>i</sub> a sentence in
one of the source languages and t<sub>i</sub> a sentence in
one of the target languages.
We train a student model M<sup>^</sup> such that
M<sup>^</sup>(s<sub>i</sub> ) ≈ M (s<sub>i</sub> ) and  M<sup>^</sup>
(t<sub>i</sub> ) ≈ M (s<sub>i</sub> ).<br>
For a given minibatch B, we minimize the mean-squared loss:
![loss](https://i.imgur.com/XfUyFmo.png)

What the above passage means that the distance between the embeddings output by Teacher model in source language and the distance between embedding output by Student model in both target and source language should be reduced. They are using [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) to do it.

The student model learns a multilingual sentence embedding space with two important properties:

1) Vector spaces are aligned across languages,i.e., identical sentences in different languages are closer to each other<br>

2) Vector space properties in the original source language from the teacher model are adopted and transferred to other languages.

#### 4. Checking similarity with Sentence-Transformer

The authors of the original paper created a library to make this model easily accessible [library](https://www.sbert.net/)

To install sentence-transformers

```
pip install -U sentence-transformers
```
##### Dataset
Here I created some sentences in three languages(English,Hindi,French) which have the same meaning.
*Note: I heavily relied on Google Translate to translate these sentences*

|English|Hindi|French|
|-|-|-|
|Muiriel is 20 now.|म्यूरियल अब बीस साल की हो गई है।|Muiriel a vingt ans maintenant.|
|Education in this world disappoints me.|मैं इस दुनिया में शिक्षा पर बहुत निराश हूँ।|L'éducation dans ce monde me déçoit.|
|That won't happen.|वैसा नहीं होगा।|Cela n'arrivera pas.|
|I miss you.|मुझें तुम्हारी याद आ रही है।|Tu me manques.|
|You should sleep.|तुम्हें सोना चाहिए।|Tu devrais dormir.|
|I never liked biology.|मुझे जीव विज्ञान कभी भी पसंद नहीं था।|Je n'ai jamais aimé la biologie.|
|No I'm not; you are!|मैं नहीं हूँ, तुम हो!|Non, je ne suis pas; tu es!|
|That's MY line!|वह तो मेरी लाईन है!|C'est ma ligne!|
|Are you sure?|पक्का?|Es-tu sûr?|
|Hurry up.|जल्दी करो!|Dépêche-toi.|

#### 5. Implementaion



``` python
import pandas as pd
```



Reading our data. The data is in tsv (tab seperated value) format


``` python
fd = pd.read_csv("/content/test_data.tsv",sep='\t')
```

Displaying our data


``` python
fd.head()
```

```
                                  English  ...                                French
0                       Muiriel is 20 now.  ...       Muiriel a vingt ans maintenant.
1  Education in this world disappoints me.  ...  L'éducation dans ce monde me déçoit.
2                       That won't happen.  ...                  Cela n'arrivera pas.
3                              I miss you.  ...                        Tu me manques.
4                        You should sleep.  ...                    Tu devrais dormir.

[5 rows x 3 columns]
```



``` python
fd.tail()
```



```
                 English  ...                            French
5  I never liked biology.  ...  Je n'ai jamais aimé la biologie.
6    No I'm not; you are!  ...       Non, je ne suis pas; tu es!
7         That's MY line!  ...                   C'est ma ligne!
8           Are you sure?  ...                        Es-tu sûr?
9               Hurry up.  ...                      Dépêche-toi.

[5 rows x 3 columns]
```


We need to input all these sentences into our transformer model. So we
will rearrange all these sentences in one columns and their respective
language in another column. This will also help us plot the data later
with [plotly](https://plotly.com/)

1.  We will create a copy of that dataframe i.e fd
2.  Then we will pivot or rearrange the dataframe using pd.melt [pandas melt](https://pandas.pydata.org/docs/reference/api/pandas.melt.html)
3.  We will input all these sentences into our SentenceTransformer model


``` python
df = fd
df["id"] = df.index
```


Created an ID column it will later help us rearrange the dataframe


``` python
df.head()
```

```
                                  English  ... id
0                       Muiriel is 20 now.  ...  0
1  Education in this world disappoints me.  ...  1
2                       That won't happen.  ...  2
3                              I miss you.  ...  3
4                        You should sleep.  ...  4

[5 rows x 4 columns]
```


Here we are rearranging our dataframe so that the \["id"\] is is our
unique id and the values will be the sentences in English,Hindi,French.
The variable will then be our column names


``` python
df = pd.melt(df, id_vars=["id"],value_vars=["English","Hindi","French"])
```

Here this example data will make it clear



``` python
df.head()
```



```
  id variable                                    value
0   0  English                       Muiriel is 20 now.
1   1  English  Education in this world disappoints me.
2   2  English                       That won't happen.
3   3  English                              I miss you.
4   4  English                        You should sleep.
```



Dropping "id" column, we don't need it anymore and also renaming our
columns to more suitable names


``` python
df.drop(columns=["id"],inplace=True)
```



``` python
df.columns=["Language","Sentence"]
```


``` python
df.head()
```



```
 Language                                 Sentence
0  English                       Muiriel is 20 now.
1  English  Education in this world disappoints me.
2  English                       That won't happen.
3  English                              I miss you.
4  English                        You should sleep.
```


``` python
from sentence_transformers import SentenceTransformer
```


We are using "paraphrase-multilingual-mpnet-base-v2" pre-trained model
for our embeddings. In this case the Teacher model was
paraphrase-mpnet-base-v2 and Student model was xlm-roberta-base

[mpnet](https://arxiv.org/abs/2004.09297 "MPNet: Masked and Permuted Pre-training for Language Understanding")

[xlm
roberta](https://arxiv.org/abs/1911.02116 "xlm-roberta paper: Unsupervised Cross-lingual Representation Learning at Scale")

[pre-trained model
list](https://www.sbert.net/docs/pretrained_models.html)


Downloading the model might take some time



``` python
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
```


Passing our sentences to model


``` python
embeddings = model.encode(df['Sentence'])
```



To plot this data in 2D we need to convert the output of model which is
in (768,) into a two dimensional space. Principal component analysis
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis "PCA wikipedia")
is used to do it


``` python
from sklearn.decomposition import PCA
pca = PCA(2)
#Transform the data
coordinates = pca.fit_transform(embeddings)
```



To plot with plotly we will need to append our x and y co-ordinates into
our dataframe df



``` python
x_list = []
y_list = []
for x,y in coordinates:
 x_list.append(x)
 y_list.append(y)
```



``` python
df['x_value'] = x_list
df['y_value'] = y_list
```



``` python
df.head()
```


```
 Language                                 Sentence   x_value   y_value
0  English                       Muiriel is 20 now. -0.901190 -0.654188
1  English  Education in this world disappoints me.  2.039421 -0.455700
2  English                       That won't happen.  0.187077 -0.644140
3  English                              I miss you. -0.088403  2.498857
4  English                        You should sleep. -0.079141  1.364437
```



Plot the data



``` python
import plotly.express as px
fig = px.scatter(df, x="x_value", y="y_value", color="Language", hover_data=['Sentence'])
fig.show()
```

![graph](https://i.imgur.com/6xGgAnv.png)

To view full interactive graph visit [link](https://plotly.com/~kcmplotly/1/ "plotly interactive graph")

As you can see with the above graph sentences which are closer together semantically i.e sentences that mean the same thing are more closer to each other even though they are in different languages

Now to compare how effective this Knowledge distillation strategy is lets check with mBERT which is multi-lingual BERT which is not trained on parallel data.

First lets import mBERT from transformer library by huggingface
[transformer](https://huggingface.co/transformers/model_doc/bert.html)

``` python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model1 = BertModel.from_pretrained("bert-base-multilingual-cased")
```


Lets give input to our BERT model


``` python
text = df['Sentence'].tolist()
```



First we tokenize the text and then give it as an input to model



``` python
encoded_input = tokenizer(text, return_tensors='pt' , padding=True)
output = model1(**encoded_input)
```


BERT outputs a dict with two keys we need the pooled output as our
embeddings


``` python
output.keys()
```


Convert the tensor into numpy array


``` python
outp_mbert = output['pooler_output'].detach().numpy()
```

Again using PCA for dimensionality reduction


``` python
from sklearn.decomposition import PCA
pca = PCA(2)
#Transform the data
coordinatesbert = pca.fit_transform(outp_mbert)
```

Adding the co-ordinates to our Dataframe


``` python
x_listbert = []
y_listbert = []
for x,y in coordinatesbert:
 x_listbert.append(x)
 y_listbert.append(y)
```


``` python
df["x_value_mbert"] = x_listbert
df["y_value_mbert"] = y_listbert
```


Plotting data


``` python
import plotly.express as px
fig = px.scatter(df, x="y_value_mbert", y="x_value_mbert", color="Language", hover_data=['Sentence'])
fig.show()
```

![mbert graph](https://i.imgur.com/tjUIN02.png)

To view full interactive graph visit [link](https://plotly.com/~kcmplotly/3/ "plotly interactive graph")


You can clearly see that the Sentences with same meanining are not
closer to each other but sentences with same languages are more closer
to each other
