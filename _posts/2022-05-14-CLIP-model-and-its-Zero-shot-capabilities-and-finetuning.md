---
layout: post
title:  "CLIP model and its Zero shot capabilities"
date:   2022-05-14
categories: [Deep Learning,CLIP]
---
## CLIP model and its Zero shot capabilities

---
### 1. Introduction
  CLIP which stands for Contrastive Language-Image Pre-training is a model made by OpenAI which can match the given image with a suitable text description of that image. CLIP is trained such that it takes input images and a textual descriptions of those images and tries to find which description matches with given images.
  In this article we will use the huggingface implementation of [clip](https://huggingface.co/docs/transformers/model_doc/clip) and find it's zero shot capabilities and try to finetune clip by adding and training some last layers.

### 2. Understanding CLIP model
  CLIP was developed as a way to identify weather an image matches it's text description or not. CLIP was trained with N (image,text) pairs as input. There are two parts of CLIP one is the image encoder and the other is text encoder.

  ![training](https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-a.svg)

  The output of image encoder (1 x N) and text encoder (1 x N) are multiplied with each other to produce (N x N) possible image text pairings. Hopefully the diagonal of the resultant matrix should have highest values, where each image corresponds to it's text description. During the training the loss is then calculated on both horizontally and vertically axis.

  ![axis](https://imgur.com/IGuhkAH)

  The loss is then added. In their paper they give some dummy code

```
  extract feature representations of each modality
  I_f = image_encoder(I) #[n, d_i]
  T_f = text_encoder(T) #[n, d_t]

  joint multimodal embedding [n, d_e]
  I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
  T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

  scaled pairwise cosine similarities [n, n]
  logits = np.dot(I_e, T_e.T) * np.exp(t)

  symmetric loss function
  labels = np.arange(n)
  loss_i = cross_entropy_loss(logits, labels, axis=0)
  loss_t = cross_entropy_loss(logits, labels, axis=1)
  loss = (loss_i + loss_t)/2
```
This teacher the model two things which images and text pair belong together and more importantly which pairs don't belong together. The batch size should be very high to do this, in paper they use a very large minibatch size of 32,768.

To use clip in zero shot setting as an image recognition model we just need to give the image and all the text description of the classes with a little bit of prompt. For eg. an image of dog with the text saying ["this is an image of dog","this is an image of cat"]. Then get the maximum value out of the output array.

![zero shot](https://openaiassets.blob.core.windows.net/$web/clip/draft/20210104b/overview-b.svg)

#### 3. Resources to learn more about CLIP
  Here are some resources that I found that can be used to understand more about how clip works.

  1. [OpenAI blog](https://openai.com/blog/clip/)
  2. [AI Coffee Break with Letitia](https://www.youtube.com/watch?v=dh8Rxhf7cLU)
  3. [OpenAI CLIP: ConnectingText and Images Paper Explained](https://www.youtube.com/watch?v=T9XSU0pKX2E)

### 4. Code using huggingface transformer library
#### Dataset
We are going to fine-tune our CLIP model on [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) dataset.
It contains images cars with their names.  
We are going to download and extract the dataset and look at how it's structured. This dataset contains 196 different classes of cars and their images

![dataset](https://ai.stanford.edu/~jkrause/cars/class_montage_flop.jpg)


```python
!wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
```


```python
!wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```


```python
!tar -xvzf /content/cars_train.tgz
```


```python
!tar -xvzf /content/car_devkit.tgz
```

The Stanford dataset has two folders one /cars_train where all the images of the cars are stored. The other one is "car_devkit", where the images path are linked with their labels. It contains metadata about those images.

Lets look at the car_devkit folder.

![Folder Structure](https://imgur.com/a/E08Dctj)

Here we can see that file 'cars_train_annos.mat' contains the annotations of the file such as
bbox_x1,bbox_x2,bbox_y1,bbox_y2 for bounding box and also label,file name. We don't need bounding box co-ordinates for this project.

The 'cars_meta.mat' contains the label name. The actual images are stored in 'cars_trains' folder but without any labels


#### Model

We will be using huggingfaces transformer library. Let's install it quickly and run a trial of some random images.


```python
!pip install transformers
```


```python
from transformers import CLIPProcessor, CLIPModel
```

Import all libraries


```python
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch import nn
import torch
import torch.optim as optim
import scipy.io
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm.notebook import tqdm as tq
```

### Zero-Shot CLIP

Here we are using [pre-trained model](https://huggingface.co/openai/clip-vit-base-patch32) of CLIP, it will take some time to download.


```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
```

This [processor](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPProcessor) is used to process the input data of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel). We don't need to write any custom data manipulation function. CLIP takes two inputs one is the image and the another is text. This processor converts image to pixel values and the text to tokens and return a dictionary containing both and also attension mask.


```python
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

Let's try giving some input

Lets give this images as input

![cats](http://images.cocodataset.org/val2017/000000039769.jpg)


```python
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
#Here we need to give CLIP a texual descriptions of image
#CLIP will find the one which matches with the image the most out of the list
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=[image], return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```


```python
probs
#The first has highest probability and it is correct answer
```




    tensor([[0.9949, 0.0051]], grad_fn=<SoftmaxBackward0>)



#### Input to CLIP model

The inputs variable is a dictionary.
```
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=[image,image], return_tensors="pt", padding=True
)
```
Since our clip model takes two inputs one is the image and another is the text for it's images and text encoder.

The input dictionary contains keys such as
> 'input_ids', 'attention_mask', 'pixel_values'

input_ids is the text that is converted to vector and the pixel_values contain the image pixel values. This three values are our input to model.



```python
inputs.keys()
```




    dict_keys(['input_ids', 'attention_mask', 'pixel_values'])



Lets make a [dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) class and try to test CLIP on our Stanford cars dataset

One more observation can be made here. Since our text input is going to be same for all images. We just need to process this input only once. The images need to be processed every time.


```python
class StanfordCars(Dataset):
  def __init__(self,metaPath,imgDir,labelMeta,model_name="openai/clip-vit-base-patch32",cuda=False):
    """
    mataPath: path to the annotation file

    imgDir: Where images are stored

    labelMeta: File where label data is stored

    model_name: Name of model we need to store. It is needed because we need to use the
    processor of the particular model to process inputs.

    cuda : To enable gpu acceleration    

    text: to store text like "This is image of {image} car"

    textInput: Input_ids of the text which needs to be passed to CLIP model

    """
    super(StanfordCars,self).__init__()
    self.metaPath = metaPath
    self.labelMeta = labelMeta
    self.path = imgDir
    train_data = scipy.io.loadmat(self.metaPath)
    class_data = scipy.io.loadmat(self.labelMeta)
    #class names
    self.classes = class_data['class_names'][0]
    # This is our data i.e filenames and their labels
    self.data = train_data['annotations'][0]
    # To process inputs
    self.processor = CLIPProcessor.from_pretrained(model_name)
    self.text = []
    self.textInput = None
    self.cuda = cuda

  def processLabels(self):
    """
    Only needs to process text once since every image will belong to at least one class in labels.
    We just process labels one time and then add these 'input_ids' to our images. We will append these later
    to our image pixel_values and pass the whole dict to CLIP model.
    """
    for i in self.classes:
      # Adding text prompt to help clip
      self.text.append(f'This is photo of {i[0]} car')
    #processing this text
    self.textInput = self.processor(text=self.text,return_tensors="pt", padding=True)

    if(self.cuda):
      for k in self.textInput.keys():
        self.textInput[k] = self.textInput[k].cuda()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    #just to check of processLable method is run or not.
    assert self.textInput!=None,'run the processLabels method'

    bbox_x1,bbox_x2,bbox_y1,bbox_y2,label,fname = self.data[idx]

    label = label.item() - 1 # because labeling starts from 1 in metadata file
    pth = self.path+'/'+fname.item()
    img = Image.open(pth)
    img = img.convert('RGB')
    #using CLIP processor to apply image pre-processing
    img = self.processor(images=img,return_tensors="pt")
    img['pixel_values'] = img['pixel_values'].squeeze() # by default batch size is one

    if(self.cuda):
      img['pixel_values'] = img['pixel_values'].cuda()

    return (img,label)
```


```python
dataset = StanfordCars(metaPath='/content/devkit/cars_train_annos.mat',imgDir='/content/cars_train',labelMeta='/content/devkit/cars_meta.mat',cuda=True)
```


```python
dataset.processLabels()
```

Lets train,eval split the dataset


```python
def train_eval_split(dataset,per,seed):
  """
  dataset: Full dataset object

  per: How much train test split

  seed: Random seed

  Splitting dataset.data which contains file name and labels into two parts.
  and then creating two different dataset for train and eval
  """
  train_data,test_data = train_test_split(dataset.data,test_size = per,random_state=seed)
  dataset.data = train_data
  evalDataset = StanfordCars(metaPath='/content/devkit/cars_train_annos.mat',imgDir='/content/cars_train',labelMeta='/content/devkit/cars_meta.mat',cuda=True)
  evalDataset.processLabels()
  evalDataset.data = test_data
  return (dataset,evalDataset)
```


```python
trainData,evalData = train_eval_split(dataset,0.05,3)
```


```python
len(trainData)
```




    7349




```python
len(evalData)
```




    387




```python
trainLoader = DataLoader(trainData,batch_size=64,shuffle=True)
evalLoader = DataLoader(evalData,batch_size=8,shuffle=True)
```

#### Check the zero-shot capacities of CLIP on eval Dataset.

This default CLIP model is not trained of any specific Stanford cars data. It is just given input for the first time without prior training.


```python
predictions = []
truth = []
#we defined eariler
model.cuda()
model.eval()
for inputs,label in tq(evalLoader):
  #add the attention mask and input_ids to input image pixel values
  for k in evalData.textInput.keys():
    inputs[k] = evalData.textInput[k]
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs = logits_per_image.softmax(dim=1)
  preds =  torch.argmax(probs, dim=1)
  preds=preds.cpu()
  for i in preds:
    predictions.append(i.item())
  for j in label:
    truth.append(j.item())
```

```python
acc = accuracy_score(truth,predictions)
print(acc)
```

    0.6020671834625323



```python
score = f1_score(truth,predictions,average='weighted')
print(score)
```

    0.5738274215018401


It is a pretty good accuracy score for the zero-shot setting. The model was able to score 60.20 percentage and f1 score of 0.573 without any specific prior training. This shows how powerful training on huge data can be.

### Fine Tune CLIP model

Let's add some layers to the end of CLIP model. We will keep the model weights frozen just add some extra layers at the end and train those layers for only few epochs.


```python
class FineTuneCLIP(nn.Module):
  def __init__(self,out_shape=196,model_name="openai/clip-vit-base-patch32",freeze=True):
    super(FineTuneCLIP,self).__init__()
    self.CLIP = CLIPModel.from_pretrained(model_name)
    # Freezing the CLIP model
    if(freeze):
      for parameter in self.CLIP.parameters():
        parameter.requires_grad=False
    # Adding extra last layers
    self.fc1 = nn.Sequential(
        nn.Linear(out_shape,out_shape*5),
        nn.BatchNorm1d(out_shape*5),
        nn.ReLU(),
        nn.Dropout(0.25)
    )

    self.fc2 =  nn.Sequential(
        nn.Linear(out_shape*5,out_shape*5),
        nn.BatchNorm1d(out_shape*5),
        nn.ReLU(),
        nn.Linear(out_shape*5,out_shape*5),
        nn.BatchNorm1d(out_shape*5),
        nn.ReLU(),
        nn.Dropout(0.3)
    )


    self.fc3 = nn.Sequential(
        nn.Linear(out_shape*5,out_shape),
        nn.BatchNorm1d(out_shape),
    )

  def forward(self,x):
    out = self.CLIP(**x)
    out = out.logits_per_image
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out
```

#### Training NN

We are writing a pytorch [training](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) loop, it is a basic training loop just we have added [tqdm](https://github.com/tqdm/tqdm) for checking our progress with time.




```python
def train(model,train_loader,eval_loader,epochs,criterion,optimizer):
  """
  This function trains our model.

  model: Our model we need to train

  train_loader: contains training data

  eval_loader: Contains validation data

  epochs: No. of epochs

  criterions: Loss function

  optimizer: Optimizer for learning

  """
  model = model.cuda()
  loss_list=[]
  accuracy_list=[]
  size = len(train_loader)
  eval_size = len(eval_loader)
  #val_steps = size//2
  for epoch in range(epochs):
    model.train()
    steps = 1
    #initilizing our tqdm progress bar for checking progress
    train_tq = tq(train_loader)
    for inputs,labels in train_tq:
      steps+=1
      """
      add text input info to dict,
      Here we are adding our 'input_ids' and 'attention_masks'
      which we have already calculated by calling processLabels() function in dataset
      to our 'pixel_values' i.e inputs which are from train_loader

      dataset.textInput = {
        'input_ids' : [tensor]
        'attention_mask': [tensor]
      }

      inputs = {
        'pixel_values' : [tensor] of shape (3,224,224)
      }

      we are adding the 'input_ids' and 'attention_masks'  values so the final input should be

      inputs = {
        'input_ids' : [tensor]
        'attention_mask': [tensor]
        'pixel_values' : [tensor] of shape (3,224,224)
      }

      This is the input to our CLIP model

      """
      for k in dataset.textInput.keys():
        inputs[k] = dataset.textInput[k]
      optimizer.zero_grad()
      outputs = model(inputs)
      #predictions
      preds =  torch.argmax(outputs, dim=1)
      #loss
      loss = criterion(outputs, labels.cuda())
      #accuracy
      acc = torch.sum(preds.cpu() == labels.cpu().data).item()
      acc = acc/len(preds)
      accuracy_list.append(acc)
      loss_list.append(loss.item())
      #backprop
      loss.backward()
      optimizer.step()
      #setting the values of our progress bar
      train_tq.set_description(f'TRAIN :: steps: {steps}/{size+1} accuray : {acc*100:.3f} loss: {loss.item():.4f} preds:{preds[0].item()} label:{labels[0].item()}')
    #calling evaluate method to check validation accuracy
    accuracy,val_loss_list = evaluate(model,eval_loader,criterion)


  return {
      "accuracy":accuracy,
      "train_loss":loss_list,
      "train_accuracy":accuracy_list,
      "val_loss":val_loss_list,
  }
```

Evaluate function to evaluate the eval dataset


```python
def evaluate(model,eval_loader,criterion):
  #calculates validation accuracy
  eval_size = len(eval_loader)
  val_acc_list = []
  val_loss_list = []
  eval_tq = tq(eval_loader)
  esteps = 0
  model.eval()
  for inputs,labels in eval_tq:
    esteps+=1
    #add text info to dict
    for k in dataset.textInput.keys():
      inputs[k] = dataset.textInput[k]

    outputs = model(inputs)
    preds =  torch.argmax(outputs, dim=1)
    val_loss = criterion(outputs, labels.cuda())
    val_acc = torch.sum(preds.cpu() == labels.cpu().squeeze().data).item()
    val_acc = val_acc/len(preds)
    val_loss_list.append(val_loss.item())
    val_acc_list.append(val_acc)

    eval_tq.set_description(f'EVAL :=: steps: {esteps}/{eval_size} accuray : {val_acc*100:.3f} loss: {val_loss.item():.4f}')

  accuracy = sum(val_acc_list)/len(val_acc_list)
  return (accuracy,val_loss_list)
```


```python
fineCLIP = FineTuneCLIP()
```

Since we frooze weight of CLIP model we only need to update parameter where required_grad property is true. Below code does that it checks which parameter has requires_grad = "True" and adds them to a list.

This list will be then given to optimizer for updation of parameters


```python
feature_extract = True
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in fineCLIP.named_parameters():
        if param.requires_grad == True:#
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in fineCLIP.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
```

    Params to learn:
    	 fc1.0.weight
    	 fc1.0.bias
    	 fc1.1.weight
    	 fc1.1.bias
    	 fc2.0.weight
    	 fc2.0.bias
    	 fc2.1.weight
    	 fc2.1.bias
    	 fc2.3.weight
    	 fc2.3.bias
    	 fc2.4.weight
    	 fc2.4.bias
    	 fc3.0.weight
    	 fc3.0.bias
    	 fc3.1.weight
    	 fc3.1.bias



```python
optimizer = optim.Adam(params_to_update,lr=0.0002)
criterion=nn.CrossEntropyLoss()
```


```python
kwargs = {"model":fineCLIP,
          "train_loader":trainLoader,
          "eval_loader":evalLoader,
          "epochs":6,
          "criterion":criterion,
          "optimizer":optimizer,
}
```

#### Training model


```python
res=train(**kwargs)
```


##### Checking validation accuray of Finetune CLIP


```python
predictions = []
truth = []
fineCLIP.eval()
for inputs,label in tq(evalLoader):
  #add the attention mask and input_ids to input image pixel values
  for k in dataset.textInput.keys():
    inputs[k] = dataset.textInput[k]
  outputs = fineCLIP(inputs)
  probs = outputs.softmax(dim=1)
  preds =  torch.argmax(probs, dim=1)
  preds=preds.cpu()
  for i in preds:
    predictions.append(i.item())
  for j in label:
    truth.append(j.item())
```


```python
acc = accuracy_score(truth,predictions)
print(acc)
```

    0.7441860465116279



```python
score = f1_score(truth,predictions,average='weighted')
print(score)
```

    0.7289987359754804


As you can see here our current accuracy is 74.41% with just few epochs, which is pretty good compared to zero-shot CLIP by only adding last layers. Remember we have keept the parameters of CLIP completly frozen. This can further be improved by doing some hyperparameter tuning.



### Conclusion

CLIP is a very powerful model with great capability. This shows that transformers models with huge dataset can learn very effectively they are also good a Zero-Shot tasks.

More detailed results are given in CLIP's original [paper](https://arxiv.org/abs/2103.00020) like how CLIP is way more robust that traditional CNN type models.
