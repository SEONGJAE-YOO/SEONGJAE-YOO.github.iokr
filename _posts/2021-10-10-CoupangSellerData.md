---
layout: post
title: 쿠팡 셀러 데이터 분석
subtitle: 개인사업 데이터 EXCEL 분석
tags: [dev]
categories: dev
tags: competition list
class: post-template
comments: true
---

```python
import matplotlib.pyplot as plt  
from matplotlib import style
from matplotlib import font_manager, rc

import numpy as np
import pandas as pd
#seaborn
import seaborn as sns
COLORS = sns.color_palette()

%matplotlib inline
```


```python
print(plt.rcParams["font.family"])
```

    ['sans-serif']



```python
# matplotlib 한글 폰트 출력코드
# 출처 : 데이터공방( https://kiddwannabe.blog.me)

import matplotlib 
from matplotlib import font_manager, rc
import platform

try : 
    if platform.system() == 'Windows':
    # 윈도우인 경우
        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)
    else:    
    # Mac 인 경우
        rc('font', family='AppleGothic')
except : 
    pass
matplotlib.rcParams['axes.unicode_minus'] = False   
```


```python
print(plt.rcParams["font.family"])
```

    ['Malgun Gothic']



```python
coupang = pd.read_excel('coupangdata.xlsx') 
coupang = pd.DataFrame(coupang)
coupang.head()
```




<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>묶음배송번호</th>
      <th>주문번호</th>
      <th>택배사</th>
      <th>운송장번호</th>
      <th>분리배송 Y/N</th>
      <th>분리배송 출고예정일</th>
      <th>주문시 출고예정일</th>
      <th>출고일(발송일)</th>
      <th>주문일</th>
      <th>등록상품명</th>
      <th>...</th>
      <th>최초등록옵션명</th>
      <th>업체상품코드</th>
      <th>바코드</th>
      <th>결제액</th>
      <th>배송비구분</th>
      <th>배송비</th>
      <th>도서산간 추가배송비</th>
      <th>구매수(수량)</th>
      <th>옵션판매가(판매단가)</th>
      <th>구매자</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1863148812</td>
      <td>5000087866634</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>배송중</td>
      <td>2020-12-02</td>
      <td>NaN</td>
      <td>2020-12-01 09:03:21</td>
      <td>마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음[무료배송]</td>
      <td>...</td>
      <td>001. 파티하는산타[무료배송]</td>
      <td>11172108||fed78e0d6d</td>
      <td>NaN</td>
      <td>3900</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3900</td>
      <td>최정미</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1864045159</td>
      <td>22000087613050</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>배송중</td>
      <td>2020-12-04</td>
      <td>NaN</td>
      <td>2020-12-01 15:12:35</td>
      <td>다이얼식 비밀번호식 열쇠[무료배송]</td>
      <td>...</td>
      <td>003단-소[무료배송]</td>
      <td>11568306||7dd580b226</td>
      <td>NaN</td>
      <td>5440</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5440</td>
      <td>조경만</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1864089590</td>
      <td>32000087295636</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>배송중</td>
      <td>2020-12-03</td>
      <td>NaN</td>
      <td>2020-12-01 15:29:09</td>
      <td>하오츠 백탕 훠궈소스 마라탕 샹궈 재료 중국식품[무료배송]</td>
      <td>...</td>
      <td>하오츠 백탕 훠궈소스 마라탕 샹궈 재료 중국식품[무료배송]</td>
      <td>9407748||00</td>
      <td>NaN</td>
      <td>23160</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>5790</td>
      <td>박순화</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1864204459</td>
      <td>6000088215603</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>배송중</td>
      <td>2020-12-03</td>
      <td>NaN</td>
      <td>2020-12-01 16:18:17</td>
      <td>bob 갤럭시핏2 fit2 3D 곡면엣지 풀커버 PET 보호필름[무료배송]</td>
      <td>...</td>
      <td>3D풀커버필름_Fit2/블랙[무료배송]</td>
      <td>11201053||317c53cc4f</td>
      <td>NaN</td>
      <td>9680</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4840</td>
      <td>김승용</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1864399502</td>
      <td>12000087552145</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>배송중</td>
      <td>2020-12-02</td>
      <td>NaN</td>
      <td>2020-12-01 17:35:39</td>
      <td>아동 성인 남녀공용 어른 겨울 뜨개실 귀마개[무료배송]</td>
      <td>...</td>
      <td>00랜덤컬러[무료배송]</td>
      <td>11291960||662b48f7d3</td>
      <td>NaN</td>
      <td>12560</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6280</td>
      <td>박현기</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
print(coupang.size)
print(len(coupang))
```

    72648
    3027



```python
# dataframe으로 만들고 '노출상품명(옵션명)' 컬럼 string으로 타입변경
#  https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html 
# 위 사이트 참고
coupang['노출상품명(옵션명)'] = coupang['노출상품명(옵션명)'].astype("string")
```


```python
coupang.dtypes
```




    묶음배송번호         object
    주문번호           object
    택배사            object
    운송장번호          object
    분리배송 Y/N       object
    분리배송 출고예정일     object
    주문시 출고예정일      object
    출고일(발송일)       object
    주문일            object
    등록상품명          object
    등록옵션명          object
    노출상품명(옵션명)     string
    노출상품ID         object
    옵션ID           object
    최초등록옵션명        object
    업체상품코드         object
    바코드            object
    결제액            object
    배송비구분          object
    배송비            object
    도서산간 추가배송비     object
    구매수(수량)        object
    옵션판매가(판매단가)    object
    구매자            object
    dtype: object



# 노출상품명(옵션명) 기준 group


```python
grouped = coupang.groupby('노출상품명(옵션명)')
```


```python
grouped
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x00000192FDACA860>




```python
grouped.count()
```




<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>묶음배송번호</th>
      <th>주문번호</th>
      <th>택배사</th>
      <th>운송장번호</th>
      <th>분리배송 Y/N</th>
      <th>분리배송 출고예정일</th>
      <th>주문시 출고예정일</th>
      <th>출고일(발송일)</th>
      <th>주문일</th>
      <th>등록상품명</th>
      <th>...</th>
      <th>최초등록옵션명</th>
      <th>업체상품코드</th>
      <th>바코드</th>
      <th>결제액</th>
      <th>배송비구분</th>
      <th>배송비</th>
      <th>도서산간 추가배송비</th>
      <th>구매수(수량)</th>
      <th>옵션판매가(판매단가)</th>
      <th>구매자</th>
    </tr>
    <tr>
      <th>노출상품명(옵션명)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(사이즈3종) 자세교정 허리 /자가발열 /허리보호대 굽은 [생활플러스] 허리복대 벨트, 00_02사이즈/ XXL</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>(스타몰)도루코 커터날 대형 10개x10팩, 상세페이지 참조</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>(오팔스토아)주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>(큐트캣) 앞치마 어깨끈 요리 주방 H형 투포켓, 상세페이지 참조</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>+코코사라+[GTrend]버핏 패턴 지퍼케이스 전기종모음 추가금X, S20+ G986/핑크</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>화이트우드사각휴지케이스[무료배송], 상세페이지 참조</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>화장실 압축기 주름 변기 뚫어펑 뚫어뻥 (15cm 레드), 상세페이지 참조</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>후드방역모자 모자페이스쉴드 후디방역모자 모자투명막 방역모자 방역후드모자 투명막후드모자[무료배송], 02그레이</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>휴대용 접이식방석 카키 등산 낚시 야외방석, 단품</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>희귀씨앗SY 씨앗10립 메버릭 제라늄 혼합, 상세페이지 참조</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2101 rows × 23 columns</p>
</div>




```python
df_group=grouped.sum()
df_group
```




<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>묶음배송번호</th>
      <th>주문일</th>
      <th>등록상품명</th>
      <th>등록옵션명</th>
      <th>노출상품ID</th>
      <th>옵션ID</th>
      <th>최초등록옵션명</th>
      <th>업체상품코드</th>
      <th>바코드</th>
      <th>배송비구분</th>
      <th>배송비</th>
      <th>도서산간 추가배송비</th>
      <th>구매수(수량)</th>
      <th>구매자</th>
    </tr>
    <tr>
      <th>노출상품명(옵션명)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(사이즈3종) 자세교정 허리 /자가발열 /허리보호대 굽은 [생활플러스] 허리복대 벨트, 00_02사이즈/ XXL</th>
      <td>1909801673</td>
      <td>2020-12-17 14:22:16</td>
      <td>[생활플러스] 굽은 허리 자세교정 벨트 /허리보호대 /자가발열 허리복대 (사이즈3종...</td>
      <td>00_02사이즈/ XXL[무료배송]</td>
      <td>4545101585</td>
      <td>72813731763</td>
      <td>00_02사이즈/ XXL[무료배송]</td>
      <td>11820400||d41294195d</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>이수정</td>
    </tr>
    <tr>
      <th>(스타몰)도루코 커터날 대형 10개x10팩, 상세페이지 참조</th>
      <td>188810073920288228142277905990</td>
      <td>2020-12-10 08:16:422021-01-29 16:59:392021-04-...</td>
      <td>도루코 커터날 소형 10개x10팩[무료배송][제프파이썬] 도루코 커터날 소형 10개...</td>
      <td>도루코 커터날 소형 10개x10팩[무료배송][제프파이썬] 도루코 커터날 소형 10...</td>
      <td>438728853543872885354387288535</td>
      <td>727510581877275105818772751058187</td>
      <td>도루코 커터날 소형 10개x10팩[무료배송][제프파이썬] 도루코 커터날 소형 10...</td>
      <td>9719011||009719011||009719011||00</td>
      <td>0</td>
      <td>무료무료무료</td>
      <td>000</td>
      <td>000</td>
      <td>4</td>
      <td>송진주박정식최미영</td>
    </tr>
    <tr>
      <th>(오팔스토아)주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445</th>
      <td>1916296697</td>
      <td>2020-12-19 19:23:45</td>
      <td>주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445[무료배송]</td>
      <td>주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445[무료배송]</td>
      <td>2092582935</td>
      <td>72331080824</td>
      <td>주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445[무료배송]</td>
      <td>10758488||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>정다운</td>
    </tr>
    <tr>
      <th>(큐트캣) 앞치마 어깨끈 요리 주방 H형 투포켓, 상세페이지 참조</th>
      <td>2339047465</td>
      <td>2021-05-16 21:42:34</td>
      <td>[jepython]@[세일상품](큐트캣) 앞치마 어깨끈 H형 요리 투포켓 주방jff...</td>
      <td>[jepython]@[세일상품](큐트캣) 앞치마 어깨끈 H형 요리 투포켓 주방jff...</td>
      <td>5267381501</td>
      <td>75689000301</td>
      <td>[jepython]@[세일상품](큐트캣) 앞치마 어깨끈 H형 요리 투포켓 주방jff...</td>
      <td>13412054||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>한지연</td>
    </tr>
    <tr>
      <th>+코코사라+[GTrend]버핏 패턴 지퍼케이스 전기종모음 추가금X, S20+ G986/핑크</th>
      <td>2116178401</td>
      <td>2021-03-02 22:18:53</td>
      <td>[제프파이썬+할인점] @전기종모음 추가금X [GTrend]버핏 지퍼케이스 패턴무료배...</td>
      <td>[제프파이썬+할인점] @S20+ G986/핑크무료배송상품~!!</td>
      <td>4733752280</td>
      <td>74000668148</td>
      <td>[제프파이썬+할인점] @S20+ G986/핑크무료배송상품~!!</td>
      <td>9092143||fbd21ef7c2</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>이은정</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>화이트우드사각휴지케이스[무료배송], 상세페이지 참조</th>
      <td>1995359239</td>
      <td>2021-01-17 00:35:33</td>
      <td>화이트우드사각휴지케이스[무료배송]</td>
      <td>화이트우드사각휴지케이스[무료배송]</td>
      <td>2360203257</td>
      <td>72081048147</td>
      <td>화이트우드사각휴지케이스[무료배송]</td>
      <td>11190111||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>양희중</td>
    </tr>
    <tr>
      <th>화장실 압축기 주름 변기 뚫어펑 뚫어뻥 (15cm 레드), 상세페이지 참조</th>
      <td>1906632249</td>
      <td>2020-12-16 15:03:20</td>
      <td>화장실 압축기 주름 변기 뚫어펑 뚫어뻥 (15cm 레드)[무료배송]</td>
      <td>화장실 압축기 주름 변기 뚫어펑 뚫어뻥 (15cm 레드)[무료배송]</td>
      <td>4602750903</td>
      <td>72979440769</td>
      <td>화장실 압축기 주름 변기 뚫어펑 뚫어뻥 (15cm 레드)[무료배송]</td>
      <td>11987532||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>김예지</td>
    </tr>
    <tr>
      <th>후드방역모자 모자페이스쉴드 후디방역모자 모자투명막 방역모자 방역후드모자 투명막후드모자[무료배송], 02그레이</th>
      <td>1910822764</td>
      <td>2020-12-17 20:26:06</td>
      <td>후드방역모자 모자페이스쉴드 후디방역모자 모자투명막 방역모자 방역후드모자 투명막후드모...</td>
      <td>02그레이[무료배송]</td>
      <td>4550562528</td>
      <td>72837959103</td>
      <td>02그레이[무료배송]</td>
      <td>11826092||a244908d03</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>JIN LIANHONG</td>
    </tr>
    <tr>
      <th>휴대용 접이식방석 카키 등산 낚시 야외방석, 단품</th>
      <td>2044438968</td>
      <td>2021-02-04 07:57:40</td>
      <td>[제프파이썬] 휴대용 접이식방석 카키 등산 낚시 야외방석[무료배송]</td>
      <td>[제프파이썬] 휴대용 접이식방석 카키 등산 낚시 야외방석[무료배송]</td>
      <td>4542461867</td>
      <td>73186959795</td>
      <td>[제프파이썬] 휴대용 접이식방석 카키 등산 낚시 야외방석[무료배송]</td>
      <td>12189631||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>윤석선</td>
    </tr>
    <tr>
      <th>희귀씨앗SY 씨앗10립 메버릭 제라늄 혼합, 상세페이지 참조</th>
      <td>2296588774</td>
      <td>2021-05-03 09:42:44</td>
      <td>[제프파이썬+할인점] @ [jepython][제프파이썬]제라늄 희귀씨앗SY 메버릭 ...</td>
      <td>[제프파이썬+할인점] @ [jepython][제프파이썬]제라늄 희귀씨앗SY 메버릭 ...</td>
      <td>4924415731</td>
      <td>73784185490</td>
      <td>[제프파이썬+할인점] @ [jepython][제프파이썬]제라늄 희귀씨앗SY 메버릭 ...</td>
      <td>12796989||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>이영선</td>
    </tr>
  </tbody>
</table>
<p>2101 rows × 14 columns</p>
</div>




```python
df_group.head()
```




<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>묶음배송번호</th>
      <th>주문일</th>
      <th>등록상품명</th>
      <th>등록옵션명</th>
      <th>노출상품ID</th>
      <th>옵션ID</th>
      <th>최초등록옵션명</th>
      <th>업체상품코드</th>
      <th>바코드</th>
      <th>배송비구분</th>
      <th>배송비</th>
      <th>도서산간 추가배송비</th>
      <th>구매수(수량)</th>
      <th>구매자</th>
    </tr>
    <tr>
      <th>노출상품명(옵션명)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(사이즈3종) 자세교정 허리 /자가발열 /허리보호대 굽은 [생활플러스] 허리복대 벨트, 00_02사이즈/ XXL</th>
      <td>1909801673</td>
      <td>2020-12-17 14:22:16</td>
      <td>[생활플러스] 굽은 허리 자세교정 벨트 /허리보호대 /자가발열 허리복대 (사이즈3종...</td>
      <td>00_02사이즈/ XXL[무료배송]</td>
      <td>4545101585</td>
      <td>72813731763</td>
      <td>00_02사이즈/ XXL[무료배송]</td>
      <td>11820400||d41294195d</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>이수정</td>
    </tr>
    <tr>
      <th>(스타몰)도루코 커터날 대형 10개x10팩, 상세페이지 참조</th>
      <td>188810073920288228142277905990</td>
      <td>2020-12-10 08:16:422021-01-29 16:59:392021-04-...</td>
      <td>도루코 커터날 소형 10개x10팩[무료배송][제프파이썬] 도루코 커터날 소형 10개...</td>
      <td>도루코 커터날 소형 10개x10팩[무료배송][제프파이썬] 도루코 커터날 소형 10...</td>
      <td>438728853543872885354387288535</td>
      <td>727510581877275105818772751058187</td>
      <td>도루코 커터날 소형 10개x10팩[무료배송][제프파이썬] 도루코 커터날 소형 10...</td>
      <td>9719011||009719011||009719011||00</td>
      <td>0</td>
      <td>무료무료무료</td>
      <td>000</td>
      <td>000</td>
      <td>4</td>
      <td>송진주박정식최미영</td>
    </tr>
    <tr>
      <th>(오팔스토아)주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445</th>
      <td>1916296697</td>
      <td>2020-12-19 19:23:45</td>
      <td>주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445[무료배송]</td>
      <td>주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445[무료배송]</td>
      <td>2092582935</td>
      <td>72331080824</td>
      <td>주방 스텐 요리 업소용 파스타 ms 뉴 얼음집게 445[무료배송]</td>
      <td>10758488||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>정다운</td>
    </tr>
    <tr>
      <th>(큐트캣) 앞치마 어깨끈 요리 주방 H형 투포켓, 상세페이지 참조</th>
      <td>2339047465</td>
      <td>2021-05-16 21:42:34</td>
      <td>[jepython]@[세일상품](큐트캣) 앞치마 어깨끈 H형 요리 투포켓 주방jff...</td>
      <td>[jepython]@[세일상품](큐트캣) 앞치마 어깨끈 H형 요리 투포켓 주방jff...</td>
      <td>5267381501</td>
      <td>75689000301</td>
      <td>[jepython]@[세일상품](큐트캣) 앞치마 어깨끈 H형 요리 투포켓 주방jff...</td>
      <td>13412054||00</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>한지연</td>
    </tr>
    <tr>
      <th>+코코사라+[GTrend]버핏 패턴 지퍼케이스 전기종모음 추가금X, S20+ G986/핑크</th>
      <td>2116178401</td>
      <td>2021-03-02 22:18:53</td>
      <td>[제프파이썬+할인점] @전기종모음 추가금X [GTrend]버핏 지퍼케이스 패턴무료배...</td>
      <td>[제프파이썬+할인점] @S20+ G986/핑크무료배송상품~!!</td>
      <td>4733752280</td>
      <td>74000668148</td>
      <td>[제프파이썬+할인점] @S20+ G986/핑크무료배송상품~!!</td>
      <td>9092143||fbd21ef7c2</td>
      <td>0</td>
      <td>무료</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>이은정</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_group.columns
```




    Index(['묶음배송번호', '주문일', '등록상품명', '등록옵션명', '노출상품ID', '옵션ID', '최초등록옵션명',
           '업체상품코드', '바코드', '배송비구분', '배송비', '도서산간 추가배송비', '구매수(수량)', '구매자'],
          dtype='object')



# 구매수(수량) 데이터 변환


```python
df_group.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2101 entries, (사이즈3종) 자세교정 허리 /자가발열 /허리보호대 굽은 [생활플러스] 허리복대 벨트, 00_02사이즈/ XXL to 희귀씨앗SY 씨앗10립 메버릭 제라늄 혼합, 상세페이지 참조
    Data columns (total 14 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   묶음배송번호      2101 non-null   object
     1   주문일         2101 non-null   object
     2   등록상품명       2101 non-null   object
     3   등록옵션명       2101 non-null   object
     4   노출상품ID      2101 non-null   object
     5   옵션ID        2101 non-null   object
     6   최초등록옵션명     2101 non-null   object
     7   업체상품코드      2101 non-null   object
     8   바코드         2101 non-null   object
     9   배송비구분       2101 non-null   object
     10  배송비         2101 non-null   object
     11  도서산간 추가배송비  2101 non-null   object
     12  구매수(수량)     2101 non-null   object
     13  구매자         2101 non-null   object
    dtypes: object(14)
    memory usage: 246.2+ KB


-  12  구매수(수량)     2101 non-null   object 을 int타입으로 변환하기


```python
df_group.dtypes
```




    묶음배송번호        object
    주문일           object
    등록상품명         object
    등록옵션명         object
    노출상품ID        object
    옵션ID          object
    최초등록옵션명       object
    업체상품코드        object
    바코드           object
    배송비구분         object
    배송비           object
    도서산간 추가배송비    object
    구매수(수량)       object
    구매자           object
    dtype: object




```python
df_group['구매수(수량)'] = pd.to_numeric(df_group['구매수(수량)'],errors = 'coerce')
print(df_group.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2101 entries, (사이즈3종) 자세교정 허리 /자가발열 /허리보호대 굽은 [생활플러스] 허리복대 벨트, 00_02사이즈/ XXL to 희귀씨앗SY 씨앗10립 메버릭 제라늄 혼합, 상세페이지 참조
    Data columns (total 14 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   묶음배송번호      2101 non-null   object 
     1   주문일         2101 non-null   object 
     2   등록상품명       2101 non-null   object 
     3   등록옵션명       2101 non-null   object 
     4   노출상품ID      2101 non-null   object 
     5   옵션ID        2101 non-null   object 
     6   최초등록옵션명     2101 non-null   object 
     7   업체상품코드      2101 non-null   object 
     8   바코드         2101 non-null   object 
     9   배송비구분       2101 non-null   object 
     10  배송비         2101 non-null   object 
     11  도서산간 추가배송비  2101 non-null   object 
     12  구매수(수량)     2100 non-null   float64
     13  구매자         2101 non-null   object 
    dtypes: float64(1), object(13)
    memory usage: 246.2+ KB
    None


- https://stackoverflow.com/questions/48094854/python-convert-object-to-float


12  구매수(수량)     2100 non-null   float64 으로 타입 변경 할 때 위 사이트 참고함


#  노출상품명(옵션명)별 구매수(수량)


```python
qua_by_items = df_group.groupby('노출상품명(옵션명)').sum()['구매수(수량)'].sort_values(ascending=False)
qua_by_items
```




    노출상품명(옵션명)
    레고 저금통(색상랜덤), 센스넷 1                                                                     70.0
    마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 023. 루돌프와산타                                  46.0
    크리스마스 모음 스티커 마음담아 9종 핸드메이드 산타 귀여운 루돌프 눈사람, 001. 파티하는산타                                  42.0
    마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 001. 파티하는산타                                  38.0
    마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 012. 크리스마스리스                                 37.0
                                                                                            ... 
    내추럴발란스 전연령 LID 감자 오리고기 포뮬라 드라이 반려견 사료 Large Bite, 11.8kg, 1개                             0.0
    [제프파이썬+할인점] @리어스텝 범퍼몰딩 엑스원 올뉴투싼 트렁크 발판 메탈무료배송상품~!!, 상세페이지 참조                             0.0
    [제프파이썬+할인점] @도구 필기 학생 정리 패스트푸드 메모 0.5mm 핑크풋 샤프무료배송상품~!!, 제품선택/감자튀김                       0.0
    [제프파이썬+할인점] @ [jepython][제프파이썬+할인점] @손가락 마디마사지기 친환경 핑거마사지기무료배송상품~!!jff2021, 상세페이지 참조     0.0
    [jepython]@[세일상품][제프파이썬+할인점] @어몽어스 소비 AMONG 용돈기입장 1000 습관 US무료배송상품~!!jff2021, 단품(랜덤)     0.0
    Name: 구매수(수량), Length: 2101, dtype: float64



# Top 10 판매 제품 확인


```python
top_selling = df_group.groupby('노출상품명(옵션명)').sum()['구매수(수량)'].sort_values(ascending=False)[:10]
top_selling
```




    노출상품명(옵션명)
    레고 저금통(색상랜덤), 센스넷 1                                                 70.0
    마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 023. 루돌프와산타              46.0
    크리스마스 모음 스티커 마음담아 9종 핸드메이드 산타 귀여운 루돌프 눈사람, 001. 파티하는산타              42.0
    마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 001. 파티하는산타              38.0
    마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 012. 크리스마스리스             37.0
    아파트방음문 문방풍 방문틈막이 월동준비 문틈바람[무료배송], 그레이                               35.0
    bob 갤럭시핏2 fit2 3D 곡면엣지 풀커버 PET 보호필름[무료배송], 단일상품, 3D풀커버필름_Fit2/블랙    32.0
    마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 045. 싱글벙글산타              27.0
    [제프파이썬]7cm책철 책철 2600책철 화신책철 50개입 725책철-소무료배송, 상세페이지 참조              21.0
    화분 자동관수 패트병 공용기 장거리 여행 색상랜덤                                         20.0
    Name: 구매수(수량), dtype: float64



# Top 10 판매 제품 데이터 시각화


```python
plot = top_selling.plot(kind='bar', color=COLORS[-1], figsize=(20,10))
plot.set_xlabel('노출상품명(옵션명)', fontsize=25)
plot.set_ylabel('구매수(수량)', fontsize=25)
plot.set_title('노출상품명(옵션명)별 구매수(수량)', fontsize=30)
plot.set_xticklabels(labels=top_selling.index, rotation=75, fontsize=13)

plt.show()

```

![coupang1](https://github.com/SEONGJAE-YOO/GitHubPageMaker/blob/master/img/coupang1.png)



# 노출상품명(옵션명) 중에 많이 나온 단어 빈도 조회


```python
# 필요한 모듈 실행
import konlpy
from konlpy.tag import *
#wordcloud 사용 
from wordcloud import wordcloud
from collections import Counter #단어에 포함된 각 글자 수 세어준다
import nltk #토큰 데이터를 살펴볼 수 있는 라이브러리
from subprocess import check_output
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator #wordcloud 라이브러리
import matplotlib as mpl


```


```python
okt = Okt() # Konlpy를 사용하여 한글 명사 추출 및 빈도 계산 
            # Okt(Twitter) 클래스를 사용하여 한글 명사 단어 빈도 계산 
            # 클래스 종류 - Hannanum, Kkma, Komoran, Mecab, Okt(Twitter)
kkma = Kkma() #

```


```python
print(okt.morphs("도움되셨다면, 공감을 꾸욱 눌러주세요~"))
# morphs는 형태소 단위로 구문 분석을 수행한다 
# okt 잘 실행되는지 확인했음
```

    ['도움', '되셨다면', ',', '공감', '을', '꾸욱', '눌러주세요', '~']


## 노출상품명(옵션명) 컬럼만 데이터 저장


```python
data1 = pd.read_excel('coupangdata.xlsx', usecols='L') # Usecols은 엑셀파일의 특정 행만을 데이터로 출력하고 싶을때 사용하는 파라미터

print(data1)

```

                                                 노출상품명(옵션명)
    0     마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 001...
    1                                 다이얼식 비밀번호식 열쇠, 003단-소
    2        현모양처 무료배송 하오츠 백탕 훠궈소스 마라탕 샹궈 재료 중국식품, 상세페이지 참조
    3     bob 갤럭시핏2 fit2 3D 곡면엣지 풀커버 PET 보호필름[무료배송], 단일상...
    4                      아동 성인 남녀공용 어른 겨울 뜨개실 귀마개, 00랜덤컬러
    ...                                                 ...
    3022  [제프파이썬+할인점] @P36766 튼튼플러스요거얌얌 이츠웰 가공식품 오렌지125g...
    3023     [제프파이썬+할인점] @그로비타 거북이사료 85g무료배송상품~!!, 상세페이지 참조
    3024              HOME 밥주걱 홀더 받침대 거치대 가정 식당 HOLDER, 화이트
    3025              HOME 밥주걱 홀더 받침대 거치대 가정 식당 HOLDER, 화이트
    3026  [jepython]@[세일상품]영대 은박지 알루미늄 테이프 (48mm7M)jff20...
    
    [3027 rows x 1 columns]



```python
# morphs = 형태소 , noun = 명사 , pos = 형태소 + 품사
#okt()함수를 이용하여 주요 키워드 추출

data1.loc[:,'노출상품명(옵션명)']
```




    0       마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음, 001...
    1                                   다이얼식 비밀번호식 열쇠, 003단-소
    2          현모양처 무료배송 하오츠 백탕 훠궈소스 마라탕 샹궈 재료 중국식품, 상세페이지 참조
    3       bob 갤럭시핏2 fit2 3D 곡면엣지 풀커버 PET 보호필름[무료배송], 단일상...
    4                        아동 성인 남녀공용 어른 겨울 뜨개실 귀마개, 00랜덤컬러
                                  ...                        
    3022    [제프파이썬+할인점] @P36766 튼튼플러스요거얌얌 이츠웰 가공식품 오렌지125g...
    3023       [제프파이썬+할인점] @그로비타 거북이사료 85g무료배송상품~!!, 상세페이지 참조
    3024                HOME 밥주걱 홀더 받침대 거치대 가정 식당 HOLDER, 화이트
    3025                HOME 밥주걱 홀더 받침대 거치대 가정 식당 HOLDER, 화이트
    3026    [jepython]@[세일상품]영대 은박지 알루미늄 테이프 (48mm7M)jff20...
    Name: 노출상품명(옵션명), Length: 3027, dtype: object




```python
## 정규식을 이용해 한글,숫자만 추출
import re
import io
k =0 

for i in data1['노출상품명(옵션명)']:
    text = re.compile('[ㄱ-ㅎ|\d\ㅏ-ㅣ|가-힣]+').findall(i)
    data1.loc[k,'노출상품명(옵션명)'] = ' '.join(text).strip()
    k+=1
```


```python
data2 = data1.loc[:'노출상품명(옵션명)'] #정규식 처리 결과 ,특수기호와 이모지가 없는 것을 확인할 수 있다.
data2
```




<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>노출상품명(옵션명)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>마음담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드메이드 스티커 9종 모음 001 ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>다이얼식 비밀번호식 열쇠 003단 소</td>
    </tr>
    <tr>
      <th>2</th>
      <td>현모양처 무료배송 하오츠 백탕 훠궈소스 마라탕 샹궈 재료 중국식품 상세페이지 참조</td>
    </tr>
    <tr>
      <th>3</th>
      <td>갤럭시핏2 2 3 곡면엣지 풀커버 보호필름 무료배송 단일상품 3 풀커버필름 2 블랙</td>
    </tr>
    <tr>
      <th>4</th>
      <td>아동 성인 남녀공용 어른 겨울 뜨개실 귀마개 00랜덤컬러</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>3022</th>
      <td>제프파이썬 할인점 36766 튼튼플러스요거얌얌 이츠웰 가공식품 오렌지125 무료배송...</td>
    </tr>
    <tr>
      <th>3023</th>
      <td>제프파이썬 할인점 그로비타 거북이사료 85 무료배송상품 상세페이지 참조</td>
    </tr>
    <tr>
      <th>3024</th>
      <td>밥주걱 홀더 받침대 거치대 가정 식당 화이트</td>
    </tr>
    <tr>
      <th>3025</th>
      <td>밥주걱 홀더 받침대 거치대 가정 식당 화이트</td>
    </tr>
    <tr>
      <th>3026</th>
      <td>세일상품 영대 은박지 알루미늄 테이프 48 7 2021 상세페이지 참조</td>
    </tr>
  </tbody>
</table>
<p>3027 rows × 1 columns</p>
</div>



- 특수기호가 제거된 것을 확인 할 수 있다


```python
#https://hwao-story.tistory.com/3
# 위 사이트 참고했음
def word_cloud_save(data2):
    mpl.rcParams['font.size']=12 
    mpl.rcParams['savefig.dpi']=100
    mpl.rcParams['figure.subplot.bottom']=.1
    
    
    stopwords = set(STOPWORDS)
    
    wordcloud = WordCloud(
                    font_path = 'C:\Windows\Fonts\H2GTRE.TTF',
                    background_color = 'white',
                    stopwords = stopwords,
                    max_words = 200,
                    max_font_size=40,
                    random_state=42
    ).generate(str(data2['노출상품명(옵션명)']))

```


```python
okt = Okt()
data2['token_item'] = data2['노출상품명(옵션명)'].apply(okt.morphs) # tokenize
data2['tagging'] = data2['노출상품명(옵션명)'].apply(okt.pos) #tokenize + 형태소 
data2['Noun'] = data2['노출상품명(옵션명)'].apply(okt.nouns)# only 명사

```

    C:\Users\MyCom\anaconda3\envs\text_analysis\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\MyCom\anaconda3\envs\text_analysis\lib\site-packages\ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    C:\Users\MyCom\anaconda3\envs\text_analysis\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.



```python
tagging = [j for i in data2['tagging'] for j in i] #리스트로 추출
tagging
```




    [('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('001', 'Number'),
     ('파티', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('다이얼', 'Noun'),
     ('식', 'Suffix'),
     ('비밀번호', 'Noun'),
     ('식', 'Suffix'),
     ('열쇠', 'Noun'),
     ('003', 'Number'),
     ('단', 'Noun'),
     ('소', 'Noun'),
     ('현모양처', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('하오츠', 'Noun'),
     ('백탕', 'Noun'),
     ('훠궈', 'Noun'),
     ('소스', 'Noun'),
     ('마라', 'Adjective'),
     ('탕', 'Noun'),
     ('샹궈', 'Noun'),
     ('재료', 'Noun'),
     ('중국', 'Noun'),
     ('식품', 'Noun'),
     ('상세', 'Noun'),
     ('페이지', 'Noun'),
     ('참조', 'Noun'),
     ('갤럭시', 'Noun'),
     ('핏', 'Noun'),
     ('2', 'Number'),
     ('2', 'Number'),
     ('3', 'Number'),
     ('곡면', 'Noun'),
     ('엣지', 'Noun'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('보호', 'Noun'),
     ('필름', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('3', 'Number'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('필름', 'Noun'),
     ('2', 'Number'),
     ('블랙', 'Noun'),
     ('아동', 'Noun'),
     ('성인', 'Noun'),
     ('남녀', 'Noun'),
     ('공용', 'Noun'),
     ('어른', 'Noun'),
     ('겨울', 'Noun'),
     ('뜨', 'Verb'),
     ('개실', 'Verb'),
     ('귀마개', 'Noun'),
     ('00', 'Number'),
     ('랜덤', 'Noun'),
     ('컬러', 'Noun'),
     ('멀티', 'Noun'),
     ('염료', 'Noun'),
     ('실크', 'Noun'),
     ('린넨', 'Noun'),
     ('울', 'Noun'),
     ('다', 'Adverb'),
     ('이론', 'Noun'),
     ('53', 'Number'),
     ('면', 'Noun'),
     ('섬유', 'Noun'),
     ('염색', 'Noun'),
     ('1321', 'Number'),
     ('엘리펀트', 'Noun'),
     ('그레이', 'Noun'),
     ('대', 'Modifier'),
     ('용량', 'Noun'),
     ('식', 'Noun'),
     ('자재', 'Noun'),
     ('반찬', 'Noun'),
     ('소시지', 'Noun'),
     ('사조', 'Noun'),
     ('1', 'Number'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('상세', 'Noun'),
     ('페이지', 'Noun'),
     ('참조', 'Noun'),
     ('범퍼', 'Noun'),
     ('케이스', 'Noun'),
     ('그립', 'Verb'),
     ('톡', 'Noun'),
     ('아이폰', 'Noun'),
     ('글라스', 'Noun'),
     ('12', 'Number'),
     ('미러', 'Noun'),
     ('하트', 'Noun'),
     ('스마트', 'Noun'),
     ('톡', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('실버', 'Noun'),
     ('갤럭시', 'Noun'),
     ('핏', 'Noun'),
     ('2', 'Number'),
     ('2', 'Number'),
     ('3', 'Number'),
     ('곡면', 'Noun'),
     ('엣지', 'Noun'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('보호', 'Noun'),
     ('필름', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('3', 'Number'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('필름', 'Noun'),
     ('2', 'Number'),
     ('블랙', 'Noun'),
     ('마향', 'Noun'),
     ('마라', 'Adjective'),
     ('탕용', 'Noun'),
     ('소스', 'Noun'),
     ('100', 'Number'),
     ('마라샹궈', 'Noun'),
     ('산초', 'Noun'),
     ('기름', 'Noun'),
     ('마', 'Noun'),
     ('조', 'Modifier'),
     ('유', 'Noun'),
     ('상세', 'Noun'),
     ('페이지', 'Noun'),
     ('참조', 'Noun'),
     ('헤어', 'Noun'),
     ('쁘띠', 'Noun'),
     ('명품', 'Noun'),
     ('머리핀', 'Noun'),
     ('동백', 'Noun'),
     ('이', 'Josa'),
     ('미니', 'Noun'),
     ('프랑스', 'Noun'),
     ('핀', 'Noun'),
     ('집게', 'Noun'),
     ('핀', 'Noun'),
     ('2', 'Number'),
     ('개', 'Noun'),
     ('1', 'Number'),
     ('쌍', 'Noun'),
     ('공효진', 'Noun'),
     ('동글이', 'Noun'),
     ('앞머리', 'Noun'),
     ('00', 'Number'),
     ('02', 'Number'),
     ('디자인', 'Noun'),
     ('에트', 'Noun'),
     ('로', 'Josa'),
     ('갤럭시', 'Noun'),
     ('핏', 'Noun'),
     ('2', 'Number'),
     ('2', 'Number'),
     ('3', 'Number'),
     ('곡면', 'Noun'),
     ('엣지', 'Noun'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('보호', 'Noun'),
     ('필름', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('3', 'Number'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('필름', 'Noun'),
     ('2', 'Number'),
     ('블랙', 'Noun'),
     ('간편한', 'Adjective'),
     ('시공', 'Noun'),
     ('주방', 'Noun'),
     ('욕실', 'Noun'),
     ('곰팡이', 'Noun'),
     ('오염', 'Noun'),
     ('방지', 'Noun'),
     ('방수', 'Noun'),
     ('테이프', 'Noun'),
     ('곰팡이', 'Noun'),
     ('방수', 'Noun'),
     ('테이프', 'Noun'),
     ('오염', 'Noun'),
     ('방지', 'Noun'),
     ('욕실', 'Noun'),
     ('틈새', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('00', 'Number'),
     ('단', 'Modifier'),
     ('품', 'Noun'),
     ('실버', 'Noun'),
     ('소품', 'Noun'),
     ('10', 'Number'),
     ('집게', 'Noun'),
     ('형', 'Suffix'),
     ('걸이', 'Noun'),
     ('고리', 'Noun'),
     ('걸이', 'Noun'),
     ('형', 'Suffix'),
     ('봉', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('상세', 'Noun'),
     ('페이지', 'Noun'),
     ('참조', 'Noun'),
     ('상세', 'Noun'),
     ('페이지', 'Noun'),
     ('참조', 'Noun'),
     ('92', 'Number'),
     ('핑코앤', 'Noun'),
     ('루리', 'Noun'),
     ('캐릭터', 'Noun'),
     ('케이스', 'Noun'),
     ('920', 'Number'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('03', 'Number'),
     ('하트', 'Noun'),
     ('패턴', 'Noun'),
     ('스마', 'Noun'),
     ('토', 'Noun'),
     ('별', 'Modifier'),
     ('비트', 'Noun'),
     ('소켓', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('073', 'Number'),
     ('8', 'Number'),
     ('45', 'Number'),
     ('설치', 'Noun'),
     ('가', 'Josa'),
     ('간단한', 'Adjective'),
     ('빨래', 'Noun'),
     ('줄', 'Noun'),
     ('15', 'Number'),
     ('갤럭시', 'Noun'),
     ('핏', 'Noun'),
     ('2', 'Number'),
     ('2', 'Number'),
     ('3', 'Number'),
     ('곡면', 'Noun'),
     ('엣지', 'Noun'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('보호', 'Noun'),
     ('필름', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('3', 'Number'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('필름', 'Noun'),
     ('2', 'Number'),
     ('블랙', 'Noun'),
     ('나비', 'Noun'),
     ('다이어리', 'Noun'),
     ('갤럭시', 'Noun'),
     ('31', 'Number'),
     ('315', 'Number'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('01', 'Number'),
     ('블랙', 'Noun'),
     ('갤럭시', 'Noun'),
     ('21', 'Number'),
     ('쉬움', 'Noun'),
     ('우레', 'Noun'),
     ('탄', 'Verb'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('필름', 'Noun'),
     ('우레', 'Noun'),
     ('탄', 'Verb'),
     ('2', 'Number'),
     ('매', 'Noun'),
     ('217', 'Number'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('00', 'Number'),
     ('통합', 'Noun'),
     ('범퍼', 'Noun'),
     ('케이스', 'Noun'),
     ('그립', 'Verb'),
     ('톡', 'Noun'),
     ('아이폰', 'Noun'),
     ('글라스', 'Noun'),
     ('12', 'Number'),
     ('미러', 'Noun'),
     ('하트', 'Noun'),
     ('스마트', 'Noun'),
     ('톡', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('실버', 'Noun'),
     ('갤럭시', 'Noun'),
     ('31', 'Number'),
     ('3', 'Number'),
     ('강화', 'Noun'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('컬러풀', 'Noun'),
     ('하드', 'Noun'),
     ('케이스', 'Noun'),
     ('001', 'Number'),
     ('1', 'Number'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('00', 'Number'),
     ('레드', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('023', 'Number'),
     ('루돌프', 'Noun'),
     ('와', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('034', 'Number'),
     ('연주회', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('045', 'Number'),
     ('싱글벙글', 'Noun'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('001', 'Number'),
     ('파티', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('023', 'Number'),
     ('루돌프', 'Noun'),
     ('와', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('034', 'Number'),
     ('연주회', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('045', 'Number'),
     ('싱글벙글', 'Noun'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('056', 'Number'),
     ('크리스마스트리', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('067', 'Number'),
     ('눈사람', 'Noun'),
     ('과', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('078', 'Number'),
     ('산타', 'Noun'),
     ('마을', 'Noun'),
     ('양말', 'Noun'),
     ('미끄럼', 'Noun'),
     ('방지', 'Noun'),
     ('융털', 'Noun'),
     ('양말', 'Noun'),
     ('수면', 'Noun'),
     ('양말', 'Noun'),
     ('방한', 'Noun'),
     ('양말', 'Noun'),
     ('검정', 'Noun'),
     ('갤럭시', 'Noun'),
     ('핏', 'Noun'),
     ('2', 'Number'),
     ('2', 'Number'),
     ('3', 'Number'),
     ('곡면', 'Noun'),
     ('엣지', 'Noun'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('보호', 'Noun'),
     ('필름', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('3', 'Number'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('필름', 'Noun'),
     ('2', 'Number'),
     ('블랙', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('012', 'Number'),
     ('크리스마스', 'Noun'),
     ('리스', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('023', 'Number'),
     ('루돌프', 'Noun'),
     ('와', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('056', 'Number'),
     ('크리스마스트리', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('078', 'Number'),
     ('산타', 'Noun'),
     ('마을', 'Noun'),
     ('크리스마스', 'Noun'),
     ('33', 'Number'),
     ('인테리어', 'Noun'),
     ('용', 'Noun'),
     ('루돌프', 'Noun'),
     ('사슴', 'Noun'),
     ('장식', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('상세', 'Noun'),
     ('페이지', 'Noun'),
     ('참조', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('012', 'Number'),
     ('크리스마스', 'Noun'),
     ('리스', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('023', 'Number'),
     ('루돌프', 'Noun'),
     ('와', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('067', 'Number'),
     ('눈사람', 'Noun'),
     ('과', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('034', 'Number'),
     ('연주회', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('001', 'Number'),
     ('파티', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('012', 'Number'),
     ('크리스마스', 'Noun'),
     ('리스', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('067', 'Number'),
     ('눈사람', 'Noun'),
     ('과', 'Josa'),
     ('산타', 'Noun'),
     ('꽃양배추', 'Noun'),
     ('임', 'Noun'),
     ('모란', 'Noun'),
     ('화분', 'Noun'),
     ('월동', 'Noun'),
     ('가능', 'Noun'),
     ('겨울', 'Noun'),
     ('화단', 'Noun'),
     ('가꾸기', 'Verb'),
     ('상세', 'Noun'),
     ('페이지', 'Noun'),
     ('참조', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('012', 'Number'),
     ('크리스마스', 'Noun'),
     ('리스', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('045', 'Number'),
     ('싱글벙글', 'Noun'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('056', 'Number'),
     ('크리스마스트리', 'Noun'),
     ('알뜰', 'Noun'),
     ('형', 'Suffix'),
     ('보풀', 'Noun'),
     ('먼지', 'Noun'),
     ('제거', 'Noun'),
     ('용', 'Noun'),
     ('롤링', 'Noun'),
     ('기', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('01', 'Number'),
     ('리필', 'Noun'),
     ('2', 'Number'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('001', 'Number'),
     ('파티', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('더', 'Noun'),
     ('예뻐진', 'Adjective'),
     ('너', 'Noun'),
     ('종이', 'Noun'),
     ('정밀', 'Noun'),
     ('아트', 'Noun'),
     ('공', 'Modifier'),
     ('예', 'Modifier'),
     ('칼', 'Noun'),
     ('컷터', 'Noun'),
     ('3', 'Number'),
     ('프린터', 'Noun'),
     ('후', 'Noun'),
     ('가공', 'Noun'),
     ('갤럭시', 'Noun'),
     ('핏', 'Noun'),
     ('2', 'Number'),
     ('2', 'Number'),
     ('3', 'Number'),
     ('곡면', 'Noun'),
     ('엣지', 'Noun'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('보호', 'Noun'),
     ('필름', 'Noun'),
     ('무료', 'Noun'),
     ('배송', 'Noun'),
     ('단일', 'Noun'),
     ('상품', 'Noun'),
     ('3', 'Number'),
     ('풀', 'Noun'),
     ('커버', 'Noun'),
     ('필름', 'Noun'),
     ('2', 'Number'),
     ('블랙', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('056', 'Number'),
     ('크리스마스트리', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('067', 'Number'),
     ('눈사람', 'Noun'),
     ('과', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('078', 'Number'),
     ('산타', 'Noun'),
     ('마을', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('012', 'Number'),
     ('크리스마스', 'Noun'),
     ('리스', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('023', 'Number'),
     ('루돌프', 'Noun'),
     ('와', 'Josa'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('034', 'Number'),
     ('연주회', 'Noun'),
     ('하는', 'Verb'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('045', 'Number'),
     ('싱글벙글', 'Noun'),
     ('산타', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ('078', 'Number'),
     ('산타', 'Noun'),
     ('마을', 'Noun'),
     ('마음', 'Noun'),
     ('담아', 'Verb'),
     ('크리스마스', 'Noun'),
     ('귀여운', 'Adjective'),
     ('산타', 'Noun'),
     ('루돌프', 'Noun'),
     ('눈사람', 'Noun'),
     ('핸드', 'Noun'),
     ('메이드', 'Noun'),
     ('스티커', 'Noun'),
     ('9', 'Number'),
     ('종', 'Noun'),
     ('모음', 'Noun'),
     ...]




```python
print(len(tagging)) #53633
```

    53633



```python
#빈도수 높은 탑 10단어 추출
#형태소 분석 및 품사 태깅
tokens = [ take2 for take1 in data2['token_item'] for take2 in take1]
text = nltk.Text(tokens, name='NMSC')
print(text.vocab().most_common(10)) #상위10
print(text.vocab().most_common()[:-20:-1]) #하위 10
```

    [('제프', 1888), ('배송', 1876), ('무료', 1863), ('파이썬', 1860), ('상세', 1714), ('페이지', 1714), ('참조', 1714), ('상품', 1578), ('할인점', 1352), ('2021', 570)]
    [('85', 1), ('거북이', 1), ('그로', 1), ('얌얌', 1), ('36766', 1), ('인디', 1), ('발톱깎이', 1), ('개시', 1), ('더펫', 1), ('마루', 1), ('6835', 1), ('색견출', 1), ('플래그', 1), ('먹물', 1), ('대명', 1), ('지속', 1), ('8시간', 1), ('자극', 1), ('쑥갓', 1)]



```python
tokens = [ take2 for take1 in data2['token_item'] for take2 in take1]
len(tokens)
```




    53633




```python
#입력한 키워드와 같이 나온 토큰을 보여줌 fL:연관 검색어
text = nltk.Text(tokens, name='NMSC')
text.concordance('무료')
```

    Displaying 25 of 1863 matches:
    파티 하는 산타 다이얼 식 비밀번호 식 열쇠 003 단 소 현모양처 무료 배송 하오츠 백탕 훠궈 소스 마라 탕 샹궈 재료 중국 식품 상세 페
    세 페이지 참조 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 아동 성인 남녀 공용 어
    색 1321 엘리펀트 그레이 대 용량 식 자재 반찬 소시지 사조 1 무료 배송 상세 페이지 참조 범퍼 케이스 그립 톡 아이폰 글라스 12 미
    참조 범퍼 케이스 그립 톡 아이폰 글라스 12 미러 하트 스마트 톡 무료 배송 단일 상품 실버 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호
    단일 상품 실버 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 마향 마라 탕용 소스 1
    디자인 에트 로 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 간편한 시공 주방 욕실 
    이 오염 방지 방수 테이프 곰팡이 방수 테이프 오염 방지 욕실 틈새 무료 배송 00 단 품 실버 소품 10 집게 형 걸이 고리 걸이 형 봉 
     배송 00 단 품 실버 소품 10 집게 형 걸이 고리 걸이 형 봉 무료 배송 상세 페이지 참조 상세 페이지 참조 92 핑코앤 루리 캐릭터 
    이지 참조 상세 페이지 참조 92 핑코앤 루리 캐릭터 케이스 920 무료 배송 03 하트 패턴 스마 토 별 비트 소켓 무료 배송 073 8 
    터 케이스 920 무료 배송 03 하트 패턴 스마 토 별 비트 소켓 무료 배송 073 8 45 설치 가 간단한 빨래 줄 15 갤럭시 핏 2 
     빨래 줄 15 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 나비 다이어리 갤럭시 3
     상품 3 풀 커버 필름 2 블랙 나비 다이어리 갤럭시 31 315 무료 배송 단일 상품 01 블랙 갤럭시 21 쉬움 우레 탄 풀 커버 필름
    랙 갤럭시 21 쉬움 우레 탄 풀 커버 필름 우레 탄 2 매 217 무료 배송 단일 상품 00 통합 범퍼 케이스 그립 톡 아이폰 글라스 12
    통합 범퍼 케이스 그립 톡 아이폰 글라스 12 미러 하트 스마트 톡 무료 배송 단일 상품 실버 갤럭시 31 3 강화 풀 커버 컬러풀 하드 케
    방한 양말 검정 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 마음 담아 크리스마스 귀
    음 078 산타 마을 크리스마스 33 인테리어 용 루돌프 사슴 장식 무료 배송 상세 페이지 참조 마음 담아 크리스마스 귀여운 산타 루돌프 눈
    종 모음 056 크리스마스트리 알뜰 형 보풀 먼지 제거 용 롤링 기 무료 배송 01 리필 2 마음 담아 크리스마스 귀여운 산타 루돌프 눈사람
    프린터 후 가공 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 마음 담아 크리스마스 귀
     산타 해피 바스 그린 콜라겐 바디 워시 200 자스민 베르가 못향 무료 배송 상세 페이지 참조 핸드 레일 부속 형 원형 35 골드 일제 무
    료 배송 상세 페이지 참조 핸드 레일 부속 형 원형 35 골드 일제 무료 배송 상세 페이지 참조 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보
    세 페이지 참조 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 갤럭시 핏 2 2 3 곡
     필름 2 블랙 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 마음 담아 크리스마스 귀
    스티커 9 종 모음 045 싱글벙글 산타 014 제주 삼다수 500 무료 배송 상세 페이지 참조 우레 탄 튜브 유공 압 호스 공 압 호스 우
     튜브 유공 압 호스 공 압 호스 우레 탄 호스 관 에어 호스 튜브 무료 배송 투명 1 마음 담아 크리스마스 귀여운 산타 루돌프 눈사람 핸드
     크리스마스트리 갤럭시 핏 2 2 3 곡면 엣지 풀 커버 보호 필름 무료 배송 단일 상품 3 풀 커버 필름 2 블랙 크리스마스 메리 산타모 



```python
# 라이브러리 다시 불러오기 
from collections import Counter
import matplotlib
from matplotlib import font_manager,rc

matplotlib.rcParams['axes.unicode_minus'] = False
#그래프에서 마이너스 기호가 표시되도록 하는 설정

def showGraph(wordInfo):
    
    font_location = "C:\Windows\Fonts\H2GTRE.TTF"
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    rc('font',family=font_name)
    
    plt.xlabel('주요 단어')
    plt.ylabel('빈도수')
    plt.grid(True)
    
    
    Sorted_Dict_Values = sorted(wordInfo.values(),reverse=True)
    Sorted_Dict_Keys = sorted(wordInfo,key=wordInfo.get,reverse=True)
    
    plt.bar(range(len(wordInfo)),Sorted_Dict_Values, align='center')
    plt.xticks(range(len(wordInfo)), list(Sorted_Dict_Keys), rotation='70')
        
    plt.show()
    
  
noun_text = [ take2 for take1 in data2['Noun'] for take2 in take1]
text = nltk.Text(noun_text, name='NMSC')
count = Counter(text.vocab())
wordInfo = dict()
for tags, counts in text.vocab().most_common(30):
    if (len(str(tags)) > 1):
        wordInfo[tags] = counts
        print ("%s : %d" % (tags, counts))
            
showGraph(wordInfo)
```

    제프 : 1888
    배송 : 1876
    무료 : 1863
    파이썬 : 1860
    상세 : 1714
    페이지 : 1714
    참조 : 1714
    상품 : 1578
    할인점 : 1352
    산타 : 440
    크리스마스 : 348
    루돌프 : 339
    스티커 : 280
    눈사람 : 258
    핸드 : 251
    모음 : 237
    마음 : 235
    메이드 : 230
    블랙 : 211
    세트 : 172
    밴드 : 159
    커버 : 154
    걸이 : 145
    갤럭시 : 143
    케이스 : 142
    필름 : 137
    보호 : 119

![coupang2](https://github.com/SEONGJAE-YOO/GitHubPageMaker/blob/master/img/coupang2.png)




```python
#'stopwords' is not defined 해결방안 - 밑에 2번줄 실행하기
from nltk.corpus import stopwords

data = text.vocab().most_common(100)

wordcloud = WordCloud(
    font_path= 'C:\Windows\Fonts\H2GTRE.TTF',
    relative_scaling=0.4,
    stopwords = stopwords,
    background_color='white',
).generate_from_frequencies( dict(data) )


plt.figure( figsize=(12, 6) )
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

```

![coupang3](./assets/img/coupang3.png)


