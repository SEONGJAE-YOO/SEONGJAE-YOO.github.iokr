---
layout: post
title: Explore The Taylor Swift Lyrics - NLP 캐글하기 위해 알아야 하는 것!
categories: dev
class: post-template
comments: true
---
{% include ExploreTheTaylorSwiftLyrics.html %}

# Explore The Taylor Swift Lyrics - NLP 캐글하기 위해 알아야 하는 것!

# 1. 멀티 라인 출력



```python
#예제1
```


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
import numpy as np

np.arange(5)
np.arange(5,10)
np.arange(1,11,5)
```




    array([0, 1, 2, 3, 4])






    array([5, 6, 7, 8, 9])






    array([1, 6])



- 위에 소스를 보면 한셀에 복수개가 출력 된다


```python
def testf(p1):
    return p1**2
```


```python
testf(4)
testf(3)
```




    16






    9



# 2. nltk 불용어 제거

갖고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업이 필요합니다. 여기서 큰 의미가 없다라는 것은 자주 등장하지만 분석을 하는 것에 있어서는 큰 도움이 되지 않는 단어들을 말합니다. 예를 들면, I, my, me, over, 조사, 접미사 같은 단어들은 문장에서는 자주 등장하지만 실제 의미 분석을 하는데는 거의 기여하는 바가 없는 경우가 있습니다. 이러한 단어들을 불용어(stopword)라고 하며, NLTK에서는 위와 같은 100여개 이상의 영어 단어들을 불용어로 패키지 내에서 미리 정의하고 있습니다.

물론 불용어는 개발자가 직접 정의할 수도 있습니다. 이번 챕터에서는 영어 문장에서 NLTK가 정의한 영어 불용어를 제거하는 실습을 하고, 한국어 문장에서 직접 정의한 불용어를 제거해보겠습니다.

NLTK 실습에서는 1챕터에서 언급했듯이 NLTK Data가 필요합니다. 만약, 데이터가 없다는 에러가 발생 시에는 nltk.download(필요한 데이터)라는 커맨드를 통해 다운로드 할 수 있습니다. 해당 커맨드 또한 에러가 발생할 경우 1챕터의 NLTK Data 가이드를 참고 바랍니다.


```python
# 예제1
from nltk.corpus import stopwords
stopwords.words('english')[:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]



stopwords.words("english")는 NLTK가 정의한 영어 불용어 리스트를 리턴합니다. 출력 결과가 100개 이상이기 때문에 여기서는 간단히 10개만 확인해보았습니다. 'i', 'me', 'my'와 같은 단어들을 NLTK에서 불용어로 정의하고 있음을 확인할 수 있습니다.


```python
# 에제2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))


word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
        
print(word_tokens)
print(result)
```

    ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
    ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']


# 한국어에서 불용어 제거하기

한국어에서 불용어 제거하기.불용어 제거하기위해서는 토큰화 후에 조사,접속사 등을 제거하는 방법이 있습니다.


```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

word = "불용어란 자주 등장하지만 데이터를 분석하는데 있어 큰 의미를 갖지 않는 단어들을 뜻합니다.불용어는 임의로 설정할 수 도 있고,영문의 불용어 리스트의 경우 NLTK 라이브러리에서 정의한 불용어 리스트를 사용할 다만 한국어의 경우 조사와 접속사의 사용이 다양하며,언어의 변형이 많기 때문에 직접 정의하는게 좋습니다."

stop ="자주,종종,가끔,많이"

stop_list = stop.split(',')
tok = word_tokenize(word)

def stopword(word_tokenize):
    result = []
    
    for w in word_tokenize:
        if w not in stop_list:
            result.append(w)
            
    return result

print('토큰화한 문장은'+ str(tok) + '입니다.',end='\n')
print('\n')
print('불용어를 제거하면' + str(stopword(tok)) + '입니다')
```

    토큰화한 문장은['불용어란', '자주', '등장하지만', '데이터를', '분석하는데', '있어', '큰', '의미를', '갖지', '않는', '단어들을', '뜻합니다.불용어는', '임의로', '설정할', '수', '도', '있고', ',', '영문의', '불용어', '리스트의', '경우', 'NLTK', '라이브러리에서', '정의한', '불용어', '리스트를', '사용할', '다만', '한국어의', '경우', '조사와', '접속사의', '사용이', '다양하며', ',', '언어의', '변형이', '많기', '때문에', '직접', '정의하는게', '좋습니다', '.']입니다.
    
    
    불용어를 제거하면['불용어란', '등장하지만', '데이터를', '분석하는데', '있어', '큰', '의미를', '갖지', '않는', '단어들을', '뜻합니다.불용어는', '임의로', '설정할', '수', '도', '있고', ',', '영문의', '불용어', '리스트의', '경우', 'NLTK', '라이브러리에서', '정의한', '불용어', '리스트를', '사용할', '다만', '한국어의', '경우', '조사와', '접속사의', '사용이', '다양하며', ',', '언어의', '변형이', '많기', '때문에', '직접', '정의하는게', '좋습니다', '.']입니다


# 3.defaultdict()는 딕셔너리를 만드는 dict클래스의 서브클래스

- defaultdict를 활용해 다음과 같이 기본값을 'int'로 선언해주고, 기존에 없던 key를 호출하면 다음과 같이 해당 key가 0으로 자동 초기화된다.


```python
from collections import defaultdict
d_dict = defaultdict(float)
d_dict["a"]
```




    0.0




```python
from collections import defaultdict
d_dict = defaultdict(int)
d_dict["a"]
```




    0



# 4. 문자열 연산


```python
import string

s = 'spam and ham'

s.upper()
s.capitalize() # 첫문자를 대문자로 한다
```




    'SPAM AND HAM'






    'Spam and ham'




```python
# 검색관련 때 사용가능
s.count('a') # 문자열 a가 몇번 나왔는지 갯수 확인한다,
```




    3




```python
s.count('sp')
```




    1




```python
s.find('a')
```




    2




```python
s.find('and') #위치
```




    5




```python
s.index('a')
```




    2




```python
#문자열 편집 및 치환
t = ' spam '
t.lstrip() #공백 제거
```




    'spam '




```python
t.strip() #좌,우 공백 제거
```




    'spam'




```python
t.rstrip()
```




    ' spam'




```python
s.replace('and','or')
```




    'spam or ham'




```python
s
```




    'spam and ham'




```python
#분리와 결합
s.split()
```




    ['spam', 'and', 'ham']




```python
s.split('and')
```




    ['spam ', ' ham']




```python
t = s.split()
```


```python
t
```




    ['spam', 'and', 'ham']




```python
''.join(t)
```




    'spamandham'




```python
'\n'.join(t)
```




    'spam\nand\nham'



# 5. from sklearn.preprocessing import StandardScaler

# 데이터 전처리와 스케일 조정


- standardScaler

  1.각 특성의 평균을 0,분산을 1로 변경

  2.최솟값과 최댓값의 크기를 제한 하지 않는다.

- RobustScaler

  1.평균과 분산 대신에 중간 값과 사분위 값을 사용함

- MinMaxScaler

  1.모든 특성이 0과 1 사이에 위치하도록 데이터를 변경함



```python
#예제1
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

```


```python
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

```

- 이 데이터셋에는 569개의 데이터 포인트에 있고 각 데이터 포인트는 30개의 측정값으로 이뤄져있습니다. 이 데이터셋에서 샘플 425개를 훈련, 143개를 테스트 세트로 나눴습니다.




```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```


```python
scaler.fit(X_train) #fit()에 훈련 데이터를 적용시킴
```




    MinMaxScaler()




```python
#데이터 변환 
#fit() 다음 transform() 사용함
X_train_scaled = scaler.transform(X_train)
# 스케일이 조정된 후 데이터셋의 속성을 출력합니다
print("변환된 후 크기: {}".format(X_train_scaled.shape))
print("스케일 조정 전 특성별 최소값:\n {}".format(X_train.min(axis=0)))
print("스케일 조정 전 특성별 최대값:\n {}".format(X_train.max(axis=0)))
print("스케일 조정 후 특성별 최소값:\n {}".format(X_train_scaled.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n {}".format(X_train_scaled.max(axis=0)))
```

    변환된 후 크기: (426, 30)
    스케일 조정 전 특성별 최소값:
     [6.981e+00 9.710e+00 4.379e+01 1.435e+02 5.263e-02 1.938e-02 0.000e+00
     0.000e+00 1.060e-01 5.024e-02 1.153e-01 3.602e-01 7.570e-01 6.802e+00
     1.713e-03 2.252e-03 0.000e+00 0.000e+00 9.539e-03 8.948e-04 7.930e+00
     1.202e+01 5.041e+01 1.852e+02 7.117e-02 2.729e-02 0.000e+00 0.000e+00
     1.566e-01 5.521e-02]
    스케일 조정 전 특성별 최대값:
     [2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.634e-01 2.867e-01 4.268e-01
     2.012e-01 3.040e-01 9.575e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
     3.113e-02 1.354e-01 3.960e-01 5.279e-02 6.146e-02 2.984e-02 3.604e+01
     4.954e+01 2.512e+02 4.254e+03 2.226e-01 9.379e-01 1.170e+00 2.910e-01
     5.774e-01 1.486e-01]
    스케일 조정 후 특성별 최소값:
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.]
    스케일 조정 후 특성별 최대값:
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1.]



```python
# 테스트 데이터 변환
X_test_scaled = scaler.transform(X_test)
# 스케일이 조정된 후 테스트 데이터의 속성을 출력합니다
print("스케일 조정 후 특성별 최소값:\n{}".format(X_test_scaled.min(axis=0)))
print("스케일 조정 후 특성별 최대값:\n{}".format(X_test_scaled.max(axis=0)))
```

    스케일 조정 후 특성별 최소값:
    [ 0.0336031   0.0226581   0.03144219  0.01141039  0.14128374  0.04406704
      0.          0.          0.1540404  -0.00615249 -0.00137796  0.00594501
      0.00430665  0.00079567  0.03919502  0.0112206   0.          0.
     -0.03191387  0.00664013  0.02660975  0.05810235  0.02031974  0.00943767
      0.1094235   0.02637792  0.          0.         -0.00023764 -0.00182032]
    스케일 조정 후 특성별 최대값:
    [0.9578778  0.81501522 0.95577362 0.89353128 0.81132075 1.21958701
     0.87956888 0.9333996  0.93232323 1.0371347  0.42669616 0.49765736
     0.44117231 0.28371044 0.48703131 0.73863671 0.76717172 0.62928585
     1.33685792 0.39057253 0.89612238 0.79317697 0.84859804 0.74488793
     0.9154725  1.13188961 1.07008547 0.92371134 1.20532319 1.63068851]


- MinMaxScaler의 변환은 데이터를 0과 1사이로 조정하는 것이 포인트입니다. 하지만 X_train의 데이터는 잘 됬지만, X_test는 그렇지 않은 것을 확인할 수 있습니다.

이는 scaler가 X_train의 데이터를 학습하여 식 자체가 xtest−xtrainmin/ xtrainmax−xtrainmin로 적용되었기 때문입니다.

스케일은 X_train을 기준으로 맞추어야 하므로 결론적으로는 맞는 변환 방식입니다. 만약 스케일 조정 후 특성별 최댓값이 0과 1보다 많이 벗어난다면 훈련 데이터와 테스트 데이터의 경향성이 다른 이상치가 있다는 것을 알 수 있습니다.


```python
# 예제2 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 메소드 체이닝(chaining)을 사용하여 fit과 transform을 연달아 호출합니다
X_scaled = scaler.fit(X_train).transform(X_train)
# 위와 동일하지만 더 효율적입니다
X_scaled_d = scaler.fit_transform(X_train)
```


```python
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=0)

svm = SVC(gamma='auto', C=100)
svm.fit(X_train, y_train)
print("테스트 세트 정확도: {:.2f}".format(svm.score(X_test, y_test)))

# 0~1 사이로 스케일 조정
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 조정된 데이터로 SVM 학습
svm.fit(X_train_scaled, y_train)

# 스케일 조정된 테스트 세트의 정확도
print("스케일 조정된 테스트 세트의 정확도: {:.2f}".format(svm.score(X_test_scaled, y_test)))
```




    SVC(C=100, gamma='auto')



    테스트 세트 정확도: 0.63
    




    MinMaxScaler()






    SVC(C=100, gamma='auto')



    스케일 조정된 테스트 세트의 정확도: 0.97


# from datetime import date, timedelta
# datetime 내장 모듈의 timedelta 클래스는 기간을 표현하기 위해 사용함




```python
#예제1 
import datetime
today = datetime.date.today()
today
```




    datetime.date(2021, 10, 12)


