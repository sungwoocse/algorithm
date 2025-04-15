algorithm/
├── basic/         # 전통적인 6개 정렬 알고리즘
├── advanced/      # 현대적인 6개 정렬 알고리즘
├── data/          # 테스트 데이터 저장
└── utils/         # 유틸리티 함수 (데이터 생성, 성능 측정 등)
└── basic_results/       # 베이직 결과
└── advanced_results/ # 현대적인 6개 정렬 알고리즘 결과

## Project Structure

### Utils
- `data_generator.py`: 다양한 유형(정렬된, 무작위, 부분 정렬)과 크기(1K-1M)의 테스트 데이터셋을 생성하는 도구입니다.
- `performance_tester.py`: 기본 정렬 알고리즘의 성능을 측정하고 비교하는 도구입니다. 실행 시간, 메모리 사용량을 측정하고 결과를 시각화합니다.
- `advanced_performance_tester.py`: 현대 정렬 알고리즘의 성능을 측정하고 비교하는 도구입니다. 실행 시간, 메모리 사용량을 측정하고 결과를 시각화합니다.
- `combine_graph.py`: 기본 및 현대 정렬 알고리즘 결과 CSV를 결합해 실행 시간 및 이론적 vs 실제 성능 그래프를 생성하는 도구입니다.

### How to use:
1. 데이터 생성: `python -m utils.data_generator`
2. 알고리즘 테스트: 알고리즘 구현 후 `performance_tester.py`, `advanced_performance_tester.py`를 사용하여 성능 측정