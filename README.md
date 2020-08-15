# Translator between German and English
 Building a Seq2Seq mode, that takes as an input a german sentence or word (encoder) outputs its english translation (decoder)
 
 ## Dataset
 
 The dataset is a English-German words/sentence pairs from : http://www.manythings.org/anki/
 
 ## Results
 <p float="left">
 <img src="https://user-images.githubusercontent.com/36988046/90320315-144d6300-df38-11ea-9aa7-c34c69e27c4d.png" width="400" /> 
  <img src="https://user-images.githubusercontent.com/36988046/90320455-85414a80-df39-11ea-91e3-b52d3eb0d227.png" width="400" />
 
 After ~ 25 epochs our validation loss stop decreasing. 
  
  
</p>
 <p float="left">
 <img src="https://user-images.githubusercontent.com/36988046/90320362-99387c80-df38-11ea-8ed1-87a010e0fe2e.png" width="400" />
 </p>
 
 The model obtained translate correctly the german sentences, however it misses out between past and present tense. For example, the model translate
 "ate" to "caught" or "sang" to "likes'. 
 
 ## Setup 
 
 * Python 3.7
 * Numpy,Pandas
 * Keras


