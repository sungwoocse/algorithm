algorithm/
├── basic/         # 전통적인 6개 정렬 알고리즘
├── advanced/      # 현대적인 6개 정렬 알고리즘
├── data/          # 테스트 데이터 저장
├── tests/         # 테스트 코드
└── utils/         # 유틸리티 함수 (데이터 생성, 성능 측정 등)

## Project Structure

### Utils
- `data_generator.py`: 다양한 유형(정렬된, 무작위, 부분 정렬)과 크기(1K-1M)의 테스트 데이터셋을 생성하는 도구입니다.
- `performance_tester.py`: 정렬 알고리즘의 성능을 측정하고 비교하는 도구입니다. 실행 시간, 메모리 사용량을 측정하고 결과를 시각화합니다.

### How to use:
1. 데이터 생성: `python -m utils.data_generator`
2. 알고리즘 테스트: 알고리즘 구현 후 `performance_tester.py`를 사용하여 성능 측정