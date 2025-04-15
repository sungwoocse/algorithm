import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 두 CSV 파일 읽기
basic_df = pd.read_csv("sorting_results.csv")
advanced_df = pd.read_csv("advanced_sorting_results.csv")

# 데이터프레임 합치기
all_df = pd.concat([basic_df, advanced_df])

# 1. 데이터 크기가 100,000인 데이터 추출하고 측정된 시간 값만 사용
df_100k = all_df[all_df['data_size'] == 100000]

# 예상 시간 문자열 포함된 행 제외
df_100k = df_100k[~df_100k['time_or_estimate'].astype(str).str.contains('Estimated')]

# min_time이 -1인 행도 제외
df_100k = df_100k[df_100k['min_time'] > 0]

# n log n 알고리즘만 필터링
n_log_n_algos = ['Merge Sort', 'Quick Sort', 'Heap Sort', 'Intro Sort', 'Tim Sort', 'Tournament Sort', 'Comb Sort']
df_100k_nlogn = df_100k[df_100k['algorithm'].isin(n_log_n_algos)]

# 시간 비교 그래프 생성
plt.figure(figsize=(12, 8))
algorithms = df_100k_nlogn['algorithm'].unique()

# 데이터셋 타입 정의
dataset_types = ['sorted_asc', 'sorted_desc', 'random', 'partial_30', 'partial_70']
datasets = [f"{dtype}_100000.txt" for dtype in dataset_types]

# 데이터가 있는지 확인
valid_datasets = []
for dataset in datasets:
    if any(df_100k_nlogn['dataset'] == dataset):
        valid_datasets.append(dataset)

if not valid_datasets:
    print("Warning: No valid data found for 100,000 element datasets.")
    # 대안으로 10,000 데이터 사용
    df_10k = all_df[all_df['data_size'] == 10000]
    df_10k = df_10k[df_10k['min_time'] > 0]
    df_10k_nlogn = df_10k[df_10k['algorithm'].isin(n_log_n_algos)]
    
    dataset_types = ['sorted_asc', 'sorted_desc', 'random', 'partial_30', 'partial_70']
    valid_datasets = [f"{dtype}_10000.txt" for dtype in dataset_types]
    df_100k_nlogn = df_10k_nlogn
    print("Using 10,000 element datasets instead.")

# 데이터셋별 바 그래프 생성
x = np.arange(len(valid_datasets))
width = 0.8 / len(algorithms) if len(algorithms) > 0 else 0.8

for i, algo in enumerate(algorithms):
    times = []
    for dataset in valid_datasets:
        temp_df = df_100k_nlogn[(df_100k_nlogn['algorithm'] == algo) & (df_100k_nlogn['dataset'] == dataset)]
        if not temp_df.empty:
            # time_or_estimate가 문자열인지 확인
            if isinstance(temp_df['time_or_estimate'].iloc[0], str) and 'Estimated' in temp_df['time_or_estimate'].iloc[0]:
                continue
            times.append(float(temp_df['time_or_estimate'].iloc[0]))
        else:
            continue  # 데이터가 없는 경우 건너뛰기
    
    if times:  # 시간이 있는 경우에만 그래프 생성
        plt.bar(x[:len(times)] + i*width, times, width, label=algo)

plt.xlabel('Dataset Type')
plt.ylabel('Average Execution Time (seconds)')
plt.title('Comparison of O(n log n) Sorting Algorithms (100,000 elements)')
plt.xticks(x + width * (len(algorithms) - 1) / 2, [d.split('_')[0] + '_' + d.split('_')[1] for d in valid_datasets])
plt.legend()
plt.tight_layout()
plt.savefig('time_comparison_100000.png')
plt.close()

# 2. 이론적 vs 실제 성능 그래프
# 알고리즘별 복잡도 정의
complexities = {
    'Bubble Sort': lambda n: n**2,
    'Selection Sort': lambda n: n**2,
    'Insertion Sort': lambda n: n**2,
    'Cocktail Sort': lambda n: n**2,
    'Merge Sort': lambda n: n * np.log2(n),
    'Quick Sort': lambda n: n * np.log2(n),
    'Heap Sort': lambda n: n * np.log2(n),
    'Intro Sort': lambda n: n * np.log2(n),
    'Tim Sort': lambda n: n * np.log2(n),
    'Tournament Sort': lambda n: n * np.log2(n),
    'Comb Sort': lambda n: n * np.log2(n)
}

# 데이터 준비 - 무작위 데이터만 사용
random_data = all_df[all_df['dataset'].str.contains('random')]
# Estimated time 문자열 제거
random_data = random_data[~random_data['time_or_estimate'].astype(str).str.contains('Estimated')]
# 음수 값 제거
random_data = random_data[random_data['min_time'] > 0]

plt.figure(figsize=(12, 8))

# 각 알고리즘별로 그래프 생성
for algo in random_data['algorithm'].unique():
    if algo in complexities:
        algo_df = random_data[random_data['algorithm'] == algo]
        algo_df = algo_df.sort_values('data_size')
        
        if len(algo_df) > 1:
            # 실제 측정 시간
            sizes = algo_df['data_size'].values
            times = []
            
            for _, row in algo_df.iterrows():
                if isinstance(row['time_or_estimate'], str) and 'Estimated' in row['time_or_estimate']:
                    continue
                times.append(float(row['time_or_estimate']))
            
            if len(times) > 1:  # 최소 2개 이상의 데이터 포인트가 필요
                plt.scatter(sizes[:len(times)], times, label=f'{algo} (Actual)')
                
                # 이론적 시간 계산
                first_size = sizes[0]
                first_time = times[0]
                constant = first_time / complexities[algo](first_size)
                theoretical_times = [constant * complexities[algo](size) for size in sizes[:len(times)]]
                plt.plot(sizes[:len(times)], theoretical_times, linestyle='--', label=f'{algo} (Theoretical)')

# 그래프가 비어있는지 확인
if not plt.gca().get_lines() and not plt.gca().collections:
    print("Warning: No valid data for theoretical vs actual plot.")
else:
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Data Size (log scale)')
    plt.ylabel('Execution Time (seconds, log scale)')
    plt.title('Theoretical vs Actual Execution Time')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig('theoretical_vs_actual.png')
    plt.close()

print("그래프가 성공적으로 생성되었습니다.")