import pandas as pd
import matplotlib.pyplot as plt

# 데이터 준비: feature 이름과 중요도를 딕셔너리 형태로 정의합니다.
data = {
    "feature": [
        "pattern_60_s_a_s", "pattern_79_t_a", "pattern_45_j_a_a", "pattern_68_s_m_a", 
        "pattern_91_m_a_s", "pattern_41_j_s_a", "pattern_12_a_s_a", "pattern_54_j_m_a", 
        "pattern_43_j_s_j", "pattern_66_s_m", "pattern_81_t_a_a", "pattern_88_m_s_a", 
        "pattern_97_m_j_s", "pattern_77_s_j_j", "pattern_86_t_j", "pattern_67_s_m_s", 
        "pattern_87_m_s", "pattern_78_s_j_s", "pattern_46_j_a_m", "pattern_21_a_s_j_a"
    ],
    "importance": [
        0.041314, 0.030017, 0.023234, 0.018250, 0.017394, 0.017230, 0.017097, 0.015823, 
        0.015476, 0.015419, 0.015029, 0.014361, 0.014325, 0.013358, 0.013335, 0.013099, 
        0.012717, 0.012696, 0.012658, 0.012626
    ]
}

# pandas DataFrame으로 변환
df = pd.DataFrame(data)

# 중요도 기준으로 오름차순 정렬 (수평 막대 그래프를 그릴 때 하단부터 작은 값이, 상단부터 큰 값이 보이도록)
df_sorted = df.sort_values(by="importance", ascending=True)

# 시각화 설정
plt.figure(figsize=(10, 6))
plt.barh(df_sorted["feature"], df_sorted["importance"], color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Feature Importance of Patterns")
plt.tight_layout()

# 그래프 출력
plt.show()
