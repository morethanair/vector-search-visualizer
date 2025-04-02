# Vector Search Visualizer

3D 벡터 공간에서 사용자 선호도 기반 상품 추천 시스템을 시각화하는 Streamlit 애플리케이션입니다.

## 주요 기능

- 3D 벡터 공간에서 상품과 사용자 벡터 시각화
- 실시간 사용자 선호도 업데이트
- 상품 추천 시스템
- 유사 사용자 표시
- 추천 상품 하이라이트 기능

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/morethanair/vector-search-visualizer.git
cd vector-search-visualizer

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

## 사용 방법

1. 왼쪽 사이드바에서 사용자 선택
2. 상품 ID를 입력하여 좋아요 표시
3. 벡터 공간에서 사용자와 상품의 위치 확인
4. 추천된 상품 확인

## 기술 스택

- Python
- Streamlit
- Plotly
- NumPy
- Pandas