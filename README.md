# 비화재보
---
## 배경
코로나 19로 인한 소방인력 부족
비화재보로 인한 인력낭비 
비화재보 지속적 발생

## 1. 비화재보 데이터 분류 모델
비화재보 데이터와 실제 화재 데이터를 분류하여 이후 분석에 필요한 데이터 셋 구축
#### 기존의 분류 체계 :
신고 내용 중 **경보설비, 속보 설비**  단어가 포함된 데이터에서  
**테스트, 오인주의, 훈련, 점검 문의**를 제외하는 방식  
#### 기존의 분류 체계의 문제 : 
비화재보 데이터와 실제 화재 데이터가 제대로 구분되지 않음.
(실제화재 중 비화재보로 분류되는 경우 多)
예시:

사용 데이터 : 2016년_01월 ~ 2021년_12월 대구 소방청 신고데이터 1711724건

## 비화재보 예측 모델