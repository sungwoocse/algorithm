import sys
import os

# 현재 파일의 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.advanced_performance_tester import AdvancedPerformanceTester
from advanced.cocktail_sort import cocktail_sort
from advanced.comb_sort import comb_sort
from advanced.intro_sort import introsort
from advanced.library_sort import library_sort
from advanced.tim_sort import tim_sort
from advanced.tournament_sort import tournament_sort

# 성능 테스터 초기화
tester = AdvancedPerformanceTester(results_dir='advanced_results')

# 테스트할 알고리즘
algorithms = {
    'Cocktail Sort': cocktail_sort,
    'Comb Sort': comb_sort,
    'Intro Sort': introsort,
    'Library Sort': library_sort,
    'Tim Sort': tim_sort,
    'Tournament Sort': tournament_sort
}

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
tester.save_results(results, filename='advanced_sorting_results.csv')

# 결과 시각화
tester.plot_time_comparison(save_path='graphs')
tester.plot_memory_comparison(save_path='graphs')
tester.plot_line_comparison(save_path='graphs')
tester.plot_algorithm_scaling(save_path='graphs')
tester.plot_theoretical_vs_actual(save_path='graphs')