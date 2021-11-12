---
layout: post
title: movie_review의 corpus data 를 sklearn을 통해 로드
categories: dev
class: post-template
comments: true
---
{% include python-table-of-contents.html %}

# movie_review의 corpus data 를 sklearn을 통해 로드

    

```python
import sklearn
from sklearn.datasets import load_files
import os
```

# 사용할 데이터는 구글 드라이브에 올려 났습니다.

[https://drive.google.com/drive/folders/1xKFDDtB8nU6D9PWO7LrP-H0G-19iPP5G?usp=sharing](https://drive.google.com/drive/folders/1xKFDDtB8nU6D9PWO7LrP-H0G-19iPP5G?usp=sharing)


```python
movietxt = "C:/Users/MyCom/Downloads/movie_reviews/movie_reviews"
```


```python
movie_train = load_files(movietxt, shuffle=True)
```


```python
# 데이터 수
len(movie_train.data)
```




    2000




```python
# moviedir 아래에는 neg, pos 폴더가 있다.
movie_train.target_names
```




    ['neg', 'pos']




```python
#  첫 번째 파일은 아놀드 슈왈제네거의 영화에 관한 것으로 보인다.
movie_train.data[0][:500]
```




    b"arnold schwarzenegger has been an icon for action enthusiasts , since the late 80's , but lately his films have been very sloppy and the one-liners are getting worse . \nit's hard seeing arnold as mr . freeze in batman and robin , especially when he says tons of ice jokes , but hey he got 15 million , what's it matter to him ? \nonce again arnold has signed to do another expensive blockbuster , that can't compare with the likes of the terminator series , true lies and even eraser . \nin this so cal"




```python
# 첫번째 파일명을 가져온다. neg 폴더에 있는 파일이다.
movie_train.filenames[0]
```




    'C:/Users/MyCom/Downloads/movie_reviews/movie_reviews\\neg\\cv405_21868.txt'




```python
# 첫번째 파일은 부정적인 리뷰이며 0번째 색인이다.
movie_train.target[0]
```




    0



# CountVectorizer & TF-IDF 를 시도해 본다.


```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
import nltk
```


```python
sents = ['A rose is a rose is a rose is a rose.',
         'Oh, what a fine day it is.',
        "It ain't over till it's over, I tell you!!"]
```


```python
# CountVectorizer의 tokenizer를 nltk의 word_tokenize를 사용하도록 변경한다.
# 디폴트는 구두점과 불용어를 무시한다.
# 그리고 최소 문서 빈도를 1로 설정한다.
foovec = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
```


```python
# 단어의 빈도수를 벡터로 바꾼다.
sents_counts = foovec.fit_transform(sents)
# foovec은 단어장에 고유 단어로 색인 된다.
foovec.vocabulary_
```




    {'a': 4,
     'rose': 14,
     'is': 9,
     '.': 3,
     'oh': 12,
     ',': 2,
     'what': 17,
     'fine': 7,
     'day': 6,
     'it': 10,
     'ai': 5,
     "n't": 11,
     'over': 13,
     'till': 16,
     "'s": 1,
     'i': 8,
     'tell': 15,
     'you': 18,
     '!': 0}




```python
# sents_counts는 3개의 문서 수와 19개의 고유 한 단어를 가진다.
# sents는 센텐스를 의미하는 듯
sents_counts.shape
```




    (3, 19)




```python
# 이 벡터는 작아서 볼 수 있다.
sents_counts.toarray()
```




    array([[0, 0, 0, 1, 4, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0],
           [2, 1, 1, 0, 0, 1, 0, 0, 1, 0, 2, 1, 0, 2, 0, 1, 1, 0, 1]],
          dtype=int64)




```python

# 빈도수를 카운트에 대한 값을 문서 빈도에 대한 반전 값으로 변환한다. 
# TF-IDF (Term Frequency -- Inverse Document Frequency)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
sents_tfidf = tfidf_transformer.fit_transform(sents_counts)
```

# TF-IDF
TF(단어 빈도, term frequency)는 특정한 단어가 문서 내에 얼마나 자주 등장하는지를 나타내는 값으로, 이 값이 높을수록 문서에서 중요하다고 생각할 수 있다. 하지만 단어 자체가 문서군 내에서 자주 사용되는 경우, 이것은 그 단어가 흔하게 등장한다는 것을 의미한다. 이것을 DF(문서 빈도, document frequency)라고 하며, 이 값의 역수를 IDF(역문서 빈도, inverse document frequency)라고 한다. TF-IDF는 TF와 IDF를 곱한 값이다.

...

특정 문서 내에서 단어 빈도가 높을 수록, 그리고 전체 문서들 중 그 단어를 포함한 문서가 적을 수록 TF-IDF값이 높아진다. 따라서 이 값을 이용하면 모든 문서에 흔하게 나타나는 단어를 걸러내는 효과를 얻을 수 있다. IDF의 로그 함수값은 항상 1 이상이므로, IDF값과 TF-IDF값은 항상 0 이상이 된다. 특정 단어를 포함하는 문서들이 많을 수록 로그 함수 안의 값이 1에 가까워지게 되고, 이 경우 IDF값과 TF-IDF값은 0에 가까워지게 된다.

출처 : TF-IDF - 위키백과, 우리 모두의 백과사전

자주 사용 되는 단어는 오히려 별로 중요하지 않은 단어일 수도 있다. 예를 들어 조사나 this, that, is, a, the 같은 단어들, 그래서 이 값의 역수를 곱해 단어의 가중치를 주게 된다.


```python
# TF-IDF values
# 문서 길이에 반해 raw counts가 정규화(normalized)되어 있다.
# 많은 문서에서 발견 되는 단어에는 가중치가 적용되어 있다.
sents_tfidf.toarray()
```




    array([[0.        , 0.        , 0.        , 0.13650997, 0.54603988,
            0.        , 0.        , 0.        , 0.        , 0.40952991,
            0.        , 0.        , 0.        , 0.        , 0.71797683,
            0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 0.28969526, 0.28969526, 0.28969526,
            0.        , 0.38091445, 0.38091445, 0.        , 0.28969526,
            0.28969526, 0.        , 0.38091445, 0.        , 0.        ,
            0.        , 0.        , 0.38091445, 0.        ],
           [0.47282517, 0.23641258, 0.17979786, 0.        , 0.        ,
            0.23641258, 0.        , 0.        , 0.23641258, 0.        ,
            0.35959573, 0.23641258, 0.        , 0.47282517, 0.        ,
            0.23641258, 0.23641258, 0.        , 0.23641258]])



# 실제 데이터로 돌아가기 : 영화 리뷰 변환



```python
# movie_vector 객체를 초기화 한 다음 영화 학습(train) 데이터를 벡터로 변환한다.
# 모든 단어를 사용한다. 82%의 정확도를 가진다.
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
# 상위 3000개의 단어만 사용하면78.5%의 정확도를 가진다.
# movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_feature=3000)
movie_counts = movie_vec.fit_transform(movie_train.data)
```


```python
# corpus에서 'screen'이라는 단어를 찾으면 19637번의 인덱스에 매핑되어있다.
movie_vec.vocabulary_.get('screen')
```




    19604




```python
# 비슷하게 스티븐 시걸을 찾아본다.
movie_vec.vocabulary_.get('seagal')
```




    19657




```python
# 중복 되면 어떤 인덱스를 반환할까? 맨 앞? 맨 뒤? 랜덤?
movie_vec.vocabulary_.get('good')
```




    9622




```python
# 엄청 크다. 2,000개의 문서와 2만5천개의 유니크한 단어들이 있다.
movie_counts.shape
```




    (2000, 25280)




```python
# 원래의 빈도수를 TF-IDF 값으로 바꾼다.
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)
```


```python
# 변환 전과 같은 크기를 가진다. 원래의 발생 빈도 대신 tf-idf 값으로 변경 되었다.
movie_tfidf.shape
```




    (2000, 25280)



# 나이브-베이즈 분류기로 트레이닝과 테스팅



```python
# 분류기를 빌드한다.
# Multinomial 나이브 베이즈 를 우리 모델에 사용한다.
# 나이브 베이즈 분류로 주로 사용 되는 예제는 영화 리뷰 감성 분석, 스팸메일 필터링 등이 있다.
from sklearn.naive_bayes import MultinomialNB
```


```python
# 트레이닝과 테스트 데이터로 나눈다.
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(
    movie_tfidf, movie_train.target, test_size = 0.20, random_state=12)
```


```python
# Multimoda 나이브 베이즈 분류기로 학습시킨다.
clf = MultinomialNB().fit(docs_train, y_train)
```


```python
# 테스트 셋에 대한 결과의 정확도를 예측한다.
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)
```




    0.82




```python
# Confusion Matrix를 만든다.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
```




    array([[175,  31],
           [ 41, 153]], dtype=int64)



# 엉터리 영화리뷰에 나이브 베이즈 분류기를 사용해 보기


```python
# 매우 짧거나 가짜 엉터리 리뷰
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shined through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']
```


```python
# 위에 있는 엉터리 리뷰를 벡터화 한다.
reviews_new_counts = movie_vec.transform(reviews_new)
reviews_new_counts.shape
```




    (13, 25280)




```python
# tf-idf로 가중치를 주어 다시 변환한다.
reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)
reviews_new_tfidf.shape
```




    (13, 25280)




```python
# MultinomialNB Classifier Multinomial 나이브 베이즈 분류기로 예측
# clf = MultinomialNB().fit(docs_train, y_train) 
pred = clf.predict(reviews_new_tfidf)
pred
```




    array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])




```python
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie_train.target_names[category]))
```

    'This movie was excellent' => pos
    'Absolute joy ride' => pos
    'Steven Seagal was terrible' => neg
    'Steven Seagal shined through.' => neg
    'This was certainly a movie' => neg
    'Two thumbs up' => neg
    'I fell asleep halfway through' => neg
    "We can't wait for the sequel!!" => neg
    '!' => neg
    '?' => neg
    'I cannot recommend this highly enough' => pos
    'instant classic.' => pos
    'Steven Seagal was amazing. His performance was Oscar-worthy.' => neg
    
