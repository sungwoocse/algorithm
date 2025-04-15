import sys
import os

# 현재 파일의 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.performance_tester import PerformanceTester
from basic.bubble_sort import bubble_sort
from basic.selection_sort import selection_sort
from basic.insertion_sort import insertion_sort
from basic.merge_sort import merge_sort
from basic.quick_sort import quick_sort
from basic.heap_sort import heap_sort

# 성능 테스터 초기화
tester = PerformanceTester()

# 테스트할 알고리즘
algorithms = {
    'Bubble Sort': bubble_sort,
    'Selection Sort': selection_sort,
    'Insertion Sort': insertion_sort,
    'Merge Sort': merge_sort,
    'Quick Sort': quick_sort,
    'Heap Sort': heap_sort
}

# O(n²) 알고리즘과 O(n log n) 알고리즘 구분
o_n2_algorithms = ['Bubble Sort', 'Selection Sort', 'Insertion Sort']
o_nlogn_algorithms = ['Merge Sort', 'Quick Sort', 'Heap Sort']

# 테스트 데이터셋 정의
test_files = [
    # 1K 크기
    'sorted_asc_1000.txt', 'sorted_desc_1000.txt', 'random_1000.txt', 
    'partial_30_1000.txt', 'partial_70_1000.txt',
    
    # 10K 크기
    'sorted_asc_10000.txt', 'sorted_desc_10000.txt', 'random_10000.txt',
    'partial_30_10000.txt', 'partial_70_10000.txt',
    
    # 100K 크기
    'sorted_asc_100000.txt', 'sorted_desc_100000.txt', 'random_100000.txt',
    'partial_30_100000.txt', 'partial_70_100000.txt',
    
    # 1M 크기
    'sorted_asc_1000000.txt', 'sorted_desc_1000000.txt', 'random_1000000.txt',
    'partial_30_1000000.txt', 'partial_70_1000000.txt'
]

print("테스트 파일 목록:")
for file in test_files:
    print(f"- {file}")

# 테스트 실행
results = tester.compare_algorithms(algorithms, test_files)

# 결과 저장
tester.save_results(results)

# 결과 시각화
tester.plot_time_comparison(save_path='results/graphs')
tester.plot_memory_comparison(save_path='results/graphs')
tester.plot_line_comparison(save_path='results/graphs')
tester.plot_algorithm_scaling(save_path='results/graphs')
tester.plot_theoretical_vs_actual(save_path='results/graphs')