import time
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, List, Dict, Any, Tuple
from pathlib import Path

class AdvancedPerformanceTester:
    """
    고급 정렬 알고리즘의 성능을 측정하고 비교하는 클래스
    """
    
    def __init__(self, data_dir='data', results_dir='advanced_results'):
        """
        초기화 함수
        
        Args:
            data_dir (str): 테스트 데이터가 저장된 디렉토리
            results_dir (str): 결과를 저장할 디렉토리
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {}
    
    def load_data(self, filename):
        """
        데이터 파일 로드
        
        Args:
            filename (str): 로드할 파일 이름
            
        Returns:
            list: 정수 리스트
        """
        filepath = os.path.join(self.data_dir, filename)
        try:
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    data.append(int(line.strip()))
            return data
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found. Generating data on the fly.")
            
            # 데이터 크기 추출
            size = int(filename.split('_')[-1].split('.')[0])
            
            # 데이터 유형 결정
            if 'sorted_asc' in filename:
                return list(range(size))
            elif 'sorted_desc' in filename:
                return list(range(size, 0, -1))
            elif 'random' in filename:
                return np.random.randint(0, size * 10, size).tolist()
            elif 'partial' in filename:
                # 부분 정렬 데이터 생성 (30% 무작위로 스왑)
                data = list(range(size))
                num_swaps = int(size * 0.3)
                for _ in range(num_swaps):
                    i, j = np.random.randint(0, size, 2)
                    data[i], data[j] = data[j], data[i]
                return data
            else:
                # 기본값: 무작위 데이터
                return np.random.randint(0, size * 10, size).tolist()
    
    def measure_time(self, sort_func: Callable, data: List[int]) -> float:
        """
        정렬 함수의 실행 시간 측정
        
        Args:
            sort_func (callable): 정렬 함수
            data (list): 정렬할 데이터
            
        Returns:
            float: 실행 시간 (초)
        """
        # 입력 데이터의 복사본 사용 (원본 데이터는 보존)
        data_copy = data.copy()
        
        # 시간 측정 시작
        start_time = time.time()
        
        # 정렬 함수 실행
        sorted_data = sort_func(data_copy)
        
        # 시간 측정 종료
        end_time = time.time()
        
        # 정렬 결과 확인 (정확성 검증)
        if sorted_data is None:  # 일부 정렬 함수는 원본 배열을 변경
            sorted_data = data_copy
            
        is_sorted = all(sorted_data[i] <= sorted_data[i+1] for i in range(len(sorted_data)-1))
        if not is_sorted:
            func_name = sort_func.__name__ if hasattr(sort_func, '__name__') else "정렬 알고리즘"
            print(f"경고: {func_name} 함수가 데이터를 올바르게 정렬하지 않았습니다!")
        
        return end_time - start_time
    
    def estimate_time(self, algo_name: str, data_size: int, sample_times: Dict[str, Dict[int, float]]) -> str:
        """
        대용량 데이터에 대한 예상 실행 시간 계산
        
        Args:
            algo_name (str): 알고리즘 이름
            data_size (int): 데이터 크기
            sample_times (dict): 알고리즘별, 크기별 실행 시간 샘플
            
        Returns:
            str: 예상 실행 시간 문자열
        """
        if algo_name not in sample_times:
            return "Estimated time: Unknown"
        
        algo_times = sample_times[algo_name]
        sizes = sorted(algo_times.keys())
        
        # 충분한 샘플이 없는 경우
        if len(sizes) < 2:
            return "Estimated time: Unknown"
        
        # O(n^2) 알고리즘에 대한 예상 (Cocktail Sort)
        if algo_name in ['Cocktail Sort']:
            # 가장 큰 샘플 크기와 실행 시간
            largest_size = sizes[-1]
            largest_time = algo_times[largest_size]
            
            # n^2 복잡도에 따른 예상 시간
            estimated_time = largest_time * ((data_size / largest_size) ** 2)
            
            # 시간 포맷팅
            if estimated_time < 60:
                return f"Estimated time: {estimated_time:.2f} seconds"
            elif estimated_time < 3600:
                return f"Estimated time: {estimated_time/60:.2f} minutes"
            elif estimated_time < 86400:
                return f"Estimated time: {estimated_time/3600:.2f} hours"
            else:
                return f"Estimated time: {estimated_time/86400:.2f} days"
        
        # O(n log n) 알고리즘에 대한 예상 (대부분의 고급 정렬 알고리즘)
        else:
            # 가장 큰 샘플 크기와 실행 시간
            largest_size = sizes[-1]
            largest_time = algo_times[largest_size]
            
            # n log n 복잡도에 따른 예상 시간
            ratio = (data_size / largest_size) * (np.log2(data_size) / np.log2(largest_size))
            estimated_time = largest_time * ratio
            
            # 시간 포맷팅
            if estimated_time < 60:
                return f"Estimated time: {estimated_time:.2f} seconds"
            elif estimated_time < 3600:
                return f"Estimated time: {estimated_time/60:.2f} minutes"
            else:
                return f"Estimated time: {estimated_time/3600:.2f} hours"
    
    def measure_memory(self, sort_func: Callable, data: List[int]) -> float:
        """
        정렬 함수의 메모리 사용량 측정
        
        Args:
            sort_func (callable): 정렬 함수
            data (list): 정렬할 데이터
            
        Returns:
            float: 사용된 최대 메모리 (MB)
        """
        data_copy = data.copy()
        
        # 현재 프로세스의 메모리 사용량 추적
        process = psutil.Process(os.getpid())
        
        # 초기 메모리 사용량
        start_memory = process.memory_info().rss / 1024 / 1024  # MB 단위로 변환
        
        # 정렬 함수 실행
        sort_func(data_copy)
        
        # 최종 메모리 사용량
        end_memory = process.memory_info().rss / 1024 / 1024  # MB 단위로 변환
        
        # 메모리 사용량이 음수인 경우 0으로 처리
        memory_usage = end_memory - start_memory
        return max(0, memory_usage)  # 음수 값은 0으로 대체
    
    def run_test(self, algorithm_name: str, sort_func: Callable, filename: str, num_runs: int = 10) -> Dict[str, Any]:
        """
        하나의 정렬 알고리즘에 대한 테스트 실행
        
        Args:
            algorithm_name (str): 알고리즘 이름
            sort_func (callable): 정렬 함수
            filename (str): 테스트 데이터 파일 이름
            num_runs (int): 반복 실행 횟수
            
        Returns:
            dict: 테스트 결과
        """
        # 데이터 로드
        data = self.load_data(filename)
        
        # 여러 번 실행하여 평균 시간 계산
        execution_times = []
        for _ in range(num_runs):
            time_taken = self.measure_time(sort_func, data)
            execution_times.append(time_taken)
        
        # 메모리 사용량 측정 (한 번만)
        memory_usage = self.measure_memory(sort_func, data)
        
        # 결과 반환
        result = {
            'algorithm': algorithm_name,
            'dataset': filename,
            'data_size': len(data),
            'avg_time': np.mean(execution_times),
            'min_time': np.min(execution_times),
            'max_time': np.max(execution_times),
            'std_time': np.std(execution_times),
            'memory_usage': memory_usage,
            'all_times': execution_times,
            'estimated': False
        }
        
        return result
    
    def compare_algorithms(self, algorithms: Dict[str, Callable], filenames: List[str], num_runs: int = 10) -> pd.DataFrame:
        """
        여러 정렬 알고리즘의 성능 비교
        
        Args:
            algorithms (dict): 알고리즘 이름과 함수의 딕셔너리
            filenames (list): 테스트할 데이터 파일 이름 목록
            num_runs (int): 각 테스트의 반복 실행 횟수
            
        Returns:
            pandas.DataFrame: 비교 결과
        """
        all_results = []
        sample_times = {}  # 알고리즘별 샘플 실행 시간 저장
        
        # 각 알고리즘과 데이터셋 조합에 대해 테스트 실행
        for algo_name, sort_func in algorithms.items():
            if algo_name not in sample_times:
                sample_times[algo_name] = {}
                
            for filename in filenames:
                # 올바른 데이터 크기 추출 방법
                if '_' in filename:
                    # 파일명에서 마지막 숫자 부분만 추출 (예: 'partial_30_1000.txt' -> 1000)
                    data_size = int(filename.split('_')[-1].split('.')[0])
                else:
                    # 숫자가 없는 경우 기본값
                    data_size = 0
                
                # O(n^2) 알고리즘은 크기가 100000을 초과하는 경우에만 예상값 사용
                if data_size > 10000 and algo_name in ['Cocktail Sort', 'Library Sort']:
                    print(f"Large dataset: {algo_name} on {filename} - using estimated time")
                    
                    # 예상 시간 계산
                    estimated_time_str = self.estimate_time(algo_name, data_size, sample_times)
                    
                    # 예상 결과 생성
                    result = {
                        'algorithm': algo_name,
                        'dataset': filename,
                        'data_size': data_size,
                        'avg_time': -1,  # 예상값 표시용
                        'min_time': -1,
                        'max_time': -1,
                        'std_time': 0,
                        'memory_usage': -1,
                        'all_times': [-1] * num_runs,
                        'estimated': True,
                        'estimated_time_str': estimated_time_str
                    }
                    
                    all_results.append(result)
                    
                    # 결과 저장 (알고리즘별 누적)
                    if algo_name not in self.results:
                        self.results[algo_name] = []
                    self.results[algo_name].append(result)
                    
                    continue
                
                print(f"테스트: {algo_name} on {filename}")
                result = self.run_test(algo_name, sort_func, filename, num_runs)
                all_results.append(result)
                
                # 샘플 시간 저장
                sample_times[algo_name][data_size] = result['avg_time']
                
                # 결과 저장 (알고리즘별 누적)
                if algo_name not in self.results:
                    self.results[algo_name] = []
                self.results[algo_name].append(result)
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(all_results)
        
        return results_df
    
    def save_results(self, df: pd.DataFrame, filename: str = 'sorting_results.csv'):
        """
        테스트 결과를 CSV 파일로 저장
        
        Args:
            df (pandas.DataFrame): 저장할 결과 데이터
            filename (str): 저장할 파일 이름
        """
        filepath = os.path.join(self.results_dir, filename)
        
        # 예상값 처리 - pandas 경고 해결
        df_save = df.copy()
        
        # 새로운 열 생성
        df_save['time_or_estimate'] = df_save['avg_time']
        
        # 예상값 행 처리
        for i, row in df_save.iterrows():
            if row.get('estimated', False):
                df_save.at[i, 'time_or_estimate'] = row.get('estimated_time_str', 'Unknown')
        
        # estimated 열 제거
        cols_to_drop = ['estimated', 'estimated_time_str', 'all_times']
        df_save = df_save.drop([col for col in cols_to_drop if col in df_save.columns], axis=1)
        
        # avg_time 열 제거하고 time_or_estimate를 사용
        if 'avg_time' in df_save.columns:
            df_save = df_save.drop('avg_time', axis=1)
        
        df_save.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def plot_time_comparison(self, df: pd.DataFrame = None, save_path: str = None):
        """
        알고리즘 실행 시간 비교 그래프 생성
        
        Args:
            df (pandas.DataFrame): 그래프화할 결과 데이터
            save_path (str): 그래프를 저장할 경로
        """
        if df is None:
            # 저장된 모든 결과에서 평균 시간 데이터 추출
            data = []
            for algo_name, results in self.results.items():
                for result in results:
                    if not result.get('estimated', False):  # 예상값 제외
                        data.append({
                            'algorithm': algo_name,
                            'dataset': result['dataset'],
                            'data_size': result['data_size'],
                            'avg_time': result['avg_time']
                        })
            df = pd.DataFrame(data)
        
        # 데이터 크기별로 그룹화
        sizes = sorted(df['data_size'].unique())
        
        # 각 데이터 크기별로 그래프 생성
        for size in sizes:
            size_df = df[df['data_size'] == size]
            
            plt.figure(figsize=(12, 8))
            ax = plt.subplot(111)
            
            # 데이터셋 유형별로 그래프 생성
            datasets = size_df['dataset'].unique()
            x = np.arange(len(datasets))
            width = 0.8 / len(size_df['algorithm'].unique())
            
            for i, algo in enumerate(sorted(size_df['algorithm'].unique())):
                algo_df = size_df[size_df['algorithm'] == algo]
                times = []
                
                for dataset in datasets:
                    dataset_df = algo_df[algo_df['dataset'] == dataset]
                    if not dataset_df.empty:
                        times.append(dataset_df['avg_time'].values[0])
                    else:
                        times.append(0)
                
                ax.bar(x + i*width, times, width, label=algo)
            
            # 축 레이블 설정
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Average Execution Time (seconds)')
            ax.set_title(f'Sorting Algorithms Comparison - Size: {size}')
            ax.set_xticks(x + width * (len(size_df['algorithm'].unique()) - 1) / 2)
            ax.set_xticklabels([os.path.basename(d) for d in datasets])
            ax.legend()
            
            plt.tight_layout()
            
            # 그래프 저장
            if save_path:
                save_dir = os.path.join(self.results_dir, save_path)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'time_comparison_{size}.png'))
            
            plt.close()
    
    def plot_memory_comparison(self, df: pd.DataFrame = None, save_path: str = None):
        """
        알고리즘 메모리 사용량 비교 그래프 생성
        
        Args:
            df (pandas.DataFrame): 그래프화할 결과 데이터
            save_path (str): 그래프를 저장할 경로
        """
        if df is None:
            # 저장된 모든 결과에서 메모리 사용량 데이터 추출
            data = []
            for algo_name, results in self.results.items():
                for result in results:
                    if not result.get('estimated', False):  # 예상값 제외
                        data.append({
                            'algorithm': algo_name,
                            'dataset': result['dataset'],
                            'data_size': result['data_size'],
                            'memory_usage': result['memory_usage']
                        })
            df = pd.DataFrame(data)
        
        # 데이터 크기별로 그룹화
        sizes = sorted(df['data_size'].unique())
        
        # 각 데이터 크기별로 그래프 생성
        for size in sizes:
            size_df = df[df['data_size'] == size]
            
            plt.figure(figsize=(12, 8))
            ax = plt.subplot(111)
            
            # 알고리즘별로 그래프 생성
            algorithms = sorted(size_df['algorithm'].unique())
            x = np.arange(len(algorithms))
            
            # 각 알고리즘의 메모리 사용량 평균 계산
            memory_usage = []
            for algo in algorithms:
                algo_df = size_df[size_df['algorithm'] == algo]
                memory_usage.append(algo_df['memory_usage'].mean())
            
            ax.bar(x, memory_usage, 0.6)
            
            # 축 레이블 설정
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title(f'Memory Usage Comparison - Size: {size}')
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms)
            
            plt.tight_layout()
            
            # 그래프 저장
            if save_path:
                save_dir = os.path.join(self.results_dir, save_path)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'memory_comparison_{size}.png'))
            
            plt.close()
    
    def plot_line_comparison(self, df: pd.DataFrame = None, save_path: str = None):
        """
        알고리즘 실행 시간을 데이터 크기별로 꺾은선 그래프로 시각화
        
        Args:
            df (pandas.DataFrame): 그래프화할 결과 데이터
            save_path (str): 그래프를 저장할 경로
        """
        if df is None:
            data = []
            for algo_name, results in self.results.items():
                for result in results:
                    if not result.get('estimated', False):  # 예상값 제외
                        # 파일명에서 데이터셋 유형 추출 (예: sorted_asc, random 등)
                        dataset_parts = result['dataset'].split('_')
                        if len(dataset_parts) > 1:
                            if 'partially' in dataset_parts[0]:
                                dataset_type = 'partially_sorted'
                            else:
                                dataset_type = '_'.join(dataset_parts[:-1])  # 마지막 숫자 부분 제외
                        else:
                            dataset_type = dataset_parts[0]
                        
                        data.append({
                            'algorithm': algo_name,
                            'dataset_type': dataset_type,
                            'data_size': result['data_size'],
                            'avg_time': result['avg_time']
                        })
            df = pd.DataFrame(data)
        
        # 데이터셋 유형별로 그래프 생성
        dataset_types = df['dataset_type'].unique()
        
        for dataset_type in dataset_types:
            plt.figure(figsize=(12, 8))
            type_df = df[df['dataset_type'] == dataset_type]
            
            # 각 알고리즘별로 꺾은선 그래프 그리기
            for algo in sorted(type_df['algorithm'].unique()):
                algo_df = type_df[type_df['algorithm'] == algo]
                # 데이터 크기 순으로 정렬
                algo_df = algo_df.sort_values('data_size')
                
                # 충분한 데이터 포인트가 있는 경우에만 그래프 그리기
                if len(algo_df) > 1:
                    plt.plot(algo_df['data_size'], algo_df['avg_time'], 
                             marker='o', linestyle='-', label=algo)
            
            plt.xscale('log')  # x축 로그 스케일
            plt.yscale('log')  # y축 로그 스케일
            
            plt.xlabel('Data Size (log scale)')
            plt.ylabel('Average Execution Time (seconds, log scale)')
            plt.title(f'Sorting Algorithm Performance - {dataset_type}')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            
            # 그래프 저장
            if save_path:
                save_dir = os.path.join(self.results_dir, save_path)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'line_comparison_{dataset_type}.png'))
            
            plt.close()
    
    def plot_algorithm_scaling(self, save_path: str = None):
        """
        각 알고리즘의 확장성을 보여주는 그래프 생성
        
        Args:
            save_path (str): 그래프를 저장할 경로
        """
        # 모든 결과에서 평균 시간 데이터 추출
        data = []
        for algo_name, results in self.results.items():
            for result in results:
                if not result.get('estimated', False):  # 예상값 제외
                    data.append({
                        'algorithm': algo_name,
                        'dataset': result['dataset'],
                        'data_size': result['data_size'],
                        'avg_time': result['avg_time']
                    })
        df = pd.DataFrame(data)
        
        # 각 알고리즘별로 그래프 생성
        for algo in sorted(df['algorithm'].unique()):
            plt.figure(figsize=(12, 8))
            algo_df = df[df['algorithm'] == algo]
            
            # 데이터셋 유형별로 그래프 생성
            dataset_types = []
            for dataset in algo_df['dataset'].unique():
                if 'sorted_asc' in dataset:
                    dataset_types.append('sorted_asc')
                elif 'sorted_desc' in dataset:
                    dataset_types.append('sorted_desc')
                elif 'random' in dataset:
                    dataset_types.append('random')
                elif 'partial' in dataset:
                    dataset_types.append('partial')
            
            dataset_types = list(set(dataset_types))
            
            for dataset_type in dataset_types:
                # 해당 데이터셋 유형을 포함하는 파일들 필터링
                type_mask = algo_df['dataset'].str.contains(dataset_type)
                type_df = algo_df[type_mask].sort_values('data_size')
                
                # 충분한 데이터 포인트가 있는 경우에만 그래프 그리기
                if len(type_df) > 1:
                    plt.plot(type_df['data_size'], type_df['avg_time'], 
                             marker='o', linestyle='-', label=dataset_type)
            
            plt.xscale('log')  # x축 로그 스케일
            plt.yscale('log')  # y축 로그 스케일
            
            plt.xlabel('Data Size (log scale)')
            plt.ylabel('Average Execution Time (seconds, log scale)')
            plt.title(f'Performance Scaling - {algo}')
            plt.grid(True, which="both", ls="--")
            plt.legend()
            
            # 참조선 추가 (O(n), O(n log n), O(n^2))
            sizes = sorted(algo_df['data_size'].unique())
            if len(sizes) > 1:
                n = np.array(sizes)
                
                # 비례 상수는 그래프에 맞게 조정
                t_min = algo_df['avg_time'].min()
                
                # O(n) 참조선
                c_n = t_min / sizes[0]
                plt.plot(n, c_n * n, 'k--', alpha=0.3, label='O(n)')
                
                # O(n log n) 참조선
                c_nlogn = t_min / (sizes[0] * np.log2(sizes[0]))
                plt.plot(n, c_nlogn * n * np.log2(n), 'k-.', alpha=0.3, label='O(n log n)')
                
                # O(n^2) 참조선
                c_n2 = t_min / (sizes[0] ** 2)
                plt.plot(n, c_n2 * n ** 2, 'k:', alpha=0.3, label='O(n²)')
            
            plt.legend()
            
            # 그래프 저장
            if save_path:
                save_dir = os.path.join(self.results_dir, save_path)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'scaling_{algo}.png'))
            
            plt.close()
    
    def plot_theoretical_vs_actual(self, save_path: str = None):
        """
        이론적 복잡도와 실제 성능 비교
        
        Args:
            save_path (str): 그래프를 저장할 경로
        """
        # 알고리즘별 복잡도 정의
        complexities = {
            'Cocktail Sort': lambda n: n**2,
            'Comb Sort': lambda n: n * np.log2(n),
            'Intro Sort': lambda n: n * np.log2(n),
            'Library Sort': lambda n: n * np.log2(n),
            'Tim Sort': lambda n: n * np.log2(n),
            'Tournament Sort': lambda n: n * np.log2(n)
        }
        
        # 모든 결과에서 평균 시간 데이터 추출
        data = []
        for algo_name, results in self.results.items():
            for result in results:
                if not result.get('estimated', False) and 'random' in result['dataset']:  # 랜덤 데이터셋만 사용
                    data.append({
                        'algorithm': algo_name,
                        'data_size': result['data_size'],
                        'avg_time': result['avg_time']
                    })
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 8))
        
        # 각 알고리즘별로 실제 시간과 이론적 시간 비교
        for algo in sorted(df['algorithm'].unique()):
            if algo in complexities:
                algo_df = df[df['algorithm'] == algo]
                algo_df = algo_df.sort_values('data_size')
                
                if len(algo_df) > 1:
                    # 실제 측정 시간
                    plt.scatter(algo_df['data_size'], algo_df['avg_time'], 
                                label=f'{algo} (Actual)', marker='o')
                    
                    # 이론적 복잡도에 맞춘 시간
                    sizes = algo_df['data_size'].values
                    times = algo_df['avg_time'].values
                    
                    # 첫 번째 측정값을 기준으로 이론적 시간 계산
                    first_size = sizes[0]
                    first_time = times[0]
                    constant = first_time / complexities[algo](first_size)
                    
                    theoretical_times = [constant * complexities[algo](size) for size in sizes]
                    plt.plot(sizes, theoretical_times, linestyle='--', 
                            label=f'{algo} (Theoretical)')
        
        plt.xscale('log')  # x축 로그 스케일
        plt.yscale('log')  # y축 로그 스케일
        
        plt.xlabel('Data Size (log scale)')
        plt.ylabel('Execution Time (seconds, log scale)')
        plt.title('Theoretical vs Actual Execution Time')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # 그래프 저장
        if save_path:
            save_dir = os.path.join(self.results_dir, save_path)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'theoretical_vs_actual.png'))
        
        plt.close()


# 실행 예시
if __name__ == "__main__":
    # 알고리즘 함수 임포트
    from advanced.cocktail_sort import cocktail_sort
    from advanced.comb_sort import comb_sort
    from advanced.intro_sort import introsort
    from advanced.library_sort import library_sort
    from advanced.tim_sort import tim_sort
    from advanced.tournament_sort import tournament_sort
    
    # 성능 테스터 초기화
    tester = AdvancedPerformanceTester()
    
    # 테스트할 알고리즘
    algorithms = {
        'Cocktail Sort': cocktail_sort,
        'Comb Sort': comb_sort,
        'Intro Sort': introsort,
        'Library Sort': library_sort,
        'Tim Sort': tim_sort,
        'Tournament Sort': tournament_sort
    }
    
    # 테스트할 파일 (크기별로 모든 유형 선택)
    test_files = [
        'sorted_asc_1000.txt', 'sorted_desc_1000.txt', 'random_1000.txt', 
        'partial_30_1000.txt', 'partial_70_1000.txt',
        'sorted_asc_10000.txt', 'sorted_desc_10000.txt', 'random_10000.txt',
        'partial_30_10000.txt', 'partial_70_10000.txt',
        'sorted_asc_100000.txt', 'sorted_desc_100000.txt', 'random_100000.txt',
        'partial_30_100000.txt', 'partial_70_100000.txt',
        'sorted_asc_1000000.txt', 'sorted_desc_1000000.txt', 'random_1000000.txt',
        'partial_30_1000000.txt', 'partial_70_1000000.txt'
    ]
    
    # 테스트 실행 (각 알고리즘 10회 반복)
    results = tester.compare_algorithms(algorithms, test_files)
    
    # 결과 저장
    tester.save_results(results)
    
    # 결과 시각화
    tester.plot_time_comparison(save_path='graphs')
    tester.plot_memory_comparison(save_path='graphs')
    tester.plot_line_comparison(save_path='graphs')
    tester.plot_algorithm_scaling(save_path='graphs')
    tester.plot_theoretical_vs_actual(save_path='graphs')