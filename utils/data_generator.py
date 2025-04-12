import random
import numpy as np
import os

class DataGenerator:
    """
    정렬 알고리즘 테스트를 위한 다양한 유형의 데이터를 생성하는 클래스
    """
    
    def __init__(self, save_dir='data'):
        """
        초기화 함수
        
        Args:
            save_dir (str): 생성된 데이터를 저장할 디렉토리
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def generate_sorted_data(self, size, ascending=True):
        """
        정렬된 데이터 생성
        
        Args:
            size (int): 데이터 크기
            ascending (bool): True면 오름차순, False면 내림차순
            
        Returns:
            list: 정렬된 정수 리스트
        """
        if ascending:
            return list(range(1, size + 1))
        else:
            return list(range(size, 0, -1))
    
    def generate_random_data(self, size):
        """
        무작위 데이터 생성
        
        Args:
            size (int): 데이터 크기
            
        Returns:
            list: 무작위 정수 리스트
        """
        return random.sample(range(1, size * 10), size)
    
    def generate_partially_sorted_data(self, size, sorted_percent=30):
        """
        부분 정렬된 데이터 생성
        
        Args:
            size (int): 데이터 크기
            sorted_percent (int): 정렬된 부분의 비율(%)
            
        Returns:
            list: 부분적으로 정렬된 정수 리스트
        """
        # 전체 데이터의 sorted_percent%는 정렬된 상태
        sorted_size = int(size * (sorted_percent / 100))
        unsorted_size = size - sorted_size
        
        # 정렬된 부분 생성
        sorted_part = list(range(1, sorted_size + 1))
        
        # 정렬되지 않은 부분 생성
        unsorted_part = random.sample(range(sorted_size + 1, (size + sorted_size) * 10), unsorted_size)
        
        # 두 부분 합치기
        data = sorted_part + unsorted_part
        
        # 합쳐진 부분을 섞기 (단, 완전히 무작위로 만들지는 않음)
        # 부분적으로 정렬된 상태를 유지하기 위해 인접한 요소들끼리만 교환
        for i in range(size // 2):
            idx1 = random.randint(0, size - 2)
            idx2 = idx1 + 1
            data[idx1], data[idx2] = data[idx2], data[idx1]
        
        return data
    
    def save_data(self, data, filename):
        """
        생성된 데이터를 파일로 저장
        
        Args:
            data (list): 저장할 데이터
            filename (str): 파일 이름
        """
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            for item in data:
                f.write(f"{item}\n")
        print(f"데이터가 {filepath}에 저장되었습니다.")
    
    def generate_all_datasets(self, sizes=None):
        """
        모든 유형의 데이터셋 생성 및 저장
        
        Args:
            sizes (list): 생성할 데이터 크기 목록. 기본값은 [1000, 10000, 100000, 1000000]
        """
        if sizes is None:
            sizes = [1000, 10000, 100000, 1000000]
        
        for size in sizes:
            # 오름차순 정렬 데이터
            asc_data = self.generate_sorted_data(size, ascending=True)
            self.save_data(asc_data, f"sorted_asc_{size}.txt")
            
            # 내림차순 정렬 데이터
            desc_data = self.generate_sorted_data(size, ascending=False)
            self.save_data(desc_data, f"sorted_desc_{size}.txt")
            
            # 무작위 데이터
            random_data = self.generate_random_data(size)
            self.save_data(random_data, f"random_{size}.txt")
            
            # 부분 정렬 데이터 (30% 정렬)
            partial_data_30 = self.generate_partially_sorted_data(size, sorted_percent=30)
            self.save_data(partial_data_30, f"partial_30_{size}.txt")
            
            # 부분 정렬 데이터 (70% 정렬)
            partial_data_70 = self.generate_partially_sorted_data(size, sorted_percent=70)
            self.save_data(partial_data_70, f"partial_70_{size}.txt")
            
        print("모든 데이터셋이 생성되었습니다.")

# 사용 예시
if __name__ == "__main__":
    generator = DataGenerator()
    
    # 모든 크기와 유형의 데이터셋 생성
    generator.generate_all_datasets([1000, 10000, 100000, 1000000])
    
    # 또는 특정 유형의 데이터만 생성
    # size = 10000
    # random_data = generator.generate_random_data(size)
    # generator.save_data(random_data, f"random_{size}.txt")