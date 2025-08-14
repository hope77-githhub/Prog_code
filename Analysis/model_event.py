import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler  # StandardScaler 대신 사용
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# 데이터 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

class ModelAnalyzer:
    def __init__(self, file_path):
        """
        모델 분석을 위한 클래스 초기화
        
        Args:
            file_path (str): CSV 파일 경로
        """
        self.file_path = file_path
        self.data = None
        self.features = None
        self.target = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = MinMaxScaler()  # 모든 모델에 Min-Max 스케일 적용

    def load_data(self):
        """데이터 로드 및 기본 전처리"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"데이터 로드 완료! 행: {self.data.shape[0]}, 열: {self.data.shape[1]}")

            # 타겟 변수 지정
            self.target = self.data["success"]

            # event_label, time, gold, success 컬럼을 제외하고, lsa_z 로 시작하는 컬럼도 제외하여 피처로 사용
            exclude_cols = ["event_label", "time", "gold", "success", "game_num"]
            feature_cols = [
                col for col in self.data.columns 
                if col not in exclude_cols and not col.startswith("lsa_z")
            ]
            self.features = self.data[feature_cols]

            # 타겟 클래스 분포 확인
            print(f"\n타겟 클래스 분포:\n{self.target.value_counts()}")
            print(f"사용 피처: {feature_cols}")
            return True
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {str(e)}")
            return False

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        훈련/테스트 데이터 분할 및 스케일링
        
        Args:
            test_size (float): 테스트 세트 비율
            random_state (int): 랜덤 시드
        """
        # 훈련/테스트 분할 (타겟 분포를 고려하여 stratify)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state, stratify=self.target
        )
        
        # Min-Max 스케일링 적용 (훈련 및 테스트 데이터 모두)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # 스케일링된 데이터를 DataFrame으로 변환 (특성명 유지)
        self.X_train_scaled_df = pd.DataFrame(
            self.X_train_scaled,
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled_df = pd.DataFrame(
            self.X_test_scaled,
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        print(f"데이터 분할 및 Min-Max 스케일링 완료! 훈련 세트: {self.X_train.shape[0]}개, 테스트 세트: {self.X_test.shape[0]}개")
        print("모든 특성이 0~1 범위로 스케일링되었습니다.")

    def train_random_forest(self, n_estimators=100, max_depth=None, random_state=42):
        """
        랜덤 포레스트 모델 훈련 (스케일링된 데이터 사용)
        
        Args:
            n_estimators (int): 트리 개수
            max_depth (int): 최대 트리 깊이
            random_state (int): 랜덤 시드
        
        Returns:
            model: 훈련된 랜덤 포레스트 모델
            metrics: 성능 지표 딕셔너리
        """
        print("\n===== 랜덤 포레스트 모델 훈련 =====")
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled_df, self.y_train)
        y_pred = rf_model.predict(self.X_test_scaled_df)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        print(f"랜덤 포레스트 - 정확도: {accuracy:.4f}, F1 점수: {f1:.4f}")
        metrics = {'accuracy': accuracy, 'f1_score': f1}
        return rf_model, metrics

    def plot_rf_feature_importance(self, model, max_features=20):
        """
        랜덤 포레스트 Feature Importance 시각화
        
        Args:
            model: 훈련된 랜덤 포레스트 모델
            max_features (int): 표시할 최대 피처 수
        """
        print("\n===== 랜덤 포레스트 Feature Importance 시각화 =====")
        # 특성 중요도 추출
        feature_importance = pd.DataFrame({
            'Feature': self.features.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        top_features = feature_importance.head(max_features)
        plt.figure(figsize=(12, 10))
        bars = plt.barh(
            y=np.arange(len(top_features)),
            width=top_features['Importance'],
            color='#1f77b4'
        )
        plt.yticks(np.arange(len(top_features)), top_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest - Top Feature Importance', fontsize=15)
        plt.gca().invert_yaxis()  # 중요도가 높은 특성이 위쪽에 표시

        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f'{top_features["Importance"].iloc[i]:.4f}',
                va='center'
            )
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("랜덤 포레스트 Feature Importance 시각화 저장 완료! ('rf_feature_importance.png')")

    def train_xgboost(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
        """
        XGBoost 모델 훈련 (스케일링된 데이터 사용)
        
        Args:
            n_estimators (int): 트리 개수
            learning_rate (float): 학습률
            max_depth (int): 최대 트리 깊이
            random_state (int): 랜덤 시드
        
        Returns:
            model: 훈련된 XGBoost 모델
            metrics: 성능 지표 딕셔너리
        """
        print("\n===== XGBoost 모델 훈련 =====")
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train_scaled_df, self.y_train)
        y_pred = xgb_model.predict(self.X_test_scaled_df)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        print(f"XGBoost - 정확도: {accuracy:.4f}, F1 점수: {f1:.4f}")
        metrics = {'accuracy': accuracy, 'f1_score': f1}
        return xgb_model, metrics

    def calculate_xgb_shap_values(self, model):
        """
        XGBoost 모델의 SHAP 값 계산
        
        Args:
            model: 훈련된 XGBoost 모델
        
        Returns:
            shap_values: 계산된 SHAP 값
            explainer: SHAP explainer 객체
        """
        print("\n===== XGBoost 모델의 SHAP 값 계산 중 =====")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test_scaled_df)
        print("XGBoost 모델의 SHAP 값 계산 완료!")
        return shap_values, explainer

    def plot_xgb_shap_beeswarm(self, shap_values, max_display=20):
        """
        XGBoost 모델의 SHAP Beeswarm 시각화
        
        Args:
            shap_values: 계산된 SHAP 값
            max_display (int): 표시할 최대 피처 수
        """
        print("\n===== XGBoost 모델의 SHAP Beeswarm 플롯 생성 =====")
        plt.figure(figsize=(12, 10))
        plt.title(f'XGBoost 모델의 SHAP 값 (상위 {max_display}개 피처)', fontsize=15)
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.zeros(shap_values.shape[0]),
            data=self.X_test_scaled_df,
            feature_names=self.features.columns
        )
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig('xgb_shap_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("XGBoost 모델의 SHAP Beeswarm 플롯 저장 완료! ('xgb_shap_beeswarm.png')")

    def plot_xgb_shap_summary(self, shap_values, max_display=20):
        """
        XGBoost 모델의 SHAP Summary 시각화
        
        Args:
            shap_values: 계산된 SHAP 값
            max_display (int): 표시할 최대 피처 수
        """
        print("\n===== XGBoost 모델의 SHAP Summary 플롯 생성 =====")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            self.X_test_scaled_df,
            feature_names=self.features.columns,
            max_display=max_display,
            show=False
        )
        plt.title(f'XGBoost 모델의 SHAP Summary (상위 {max_display}개 피처)', fontsize=15)
        plt.tight_layout()
        plt.savefig('xgb_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("XGBoost 모델의 SHAP Summary 플롯 저장 완료! ('xgb_shap_summary.png')")

    def plot_xgb_shap_bar(self, shap_values, max_display=20):
        """
        XGBoost 모델의 SHAP Bar 시각화
        
        Args:
            shap_values: 계산된 SHAP 값
            max_display (int): 표시할 최대 피처 수
        """
        print("\n===== XGBoost 모델의 SHAP Bar 플롯 생성 =====")
        plt.figure(figsize=(12, 8))
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'Feature': self.features.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        top_features = feature_importance_df.head(max_display)
        bars = plt.barh(
            y=np.arange(len(top_features)),
            width=top_features['Importance'],
            color='#ff7f0e'
        )
        plt.yticks(np.arange(len(top_features)), top_features['Feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('XGBoost - SHAP Feature Importance', fontsize=15)
        plt.gca().invert_yaxis()

        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f'{top_features["Importance"].iloc[i]:.4f}',
                va='center'
            )
        plt.tight_layout()
        plt.savefig('xgb_shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("XGBoost 모델의 SHAP Bar 플롯 저장 완료! ('xgb_shap_bar.png')")

    def train_mlp(self, hidden_layer_sizes=(100, 50), max_iter=500, random_state=42):
        """
        MLP(다층 퍼셉트론) 모델 훈련
        
        Args:
            hidden_layer_sizes (tuple): 은닉층 구조
            max_iter (int): 최대 반복 횟수
            random_state (int): 랜덤 시드
        
        Returns:
            model: 훈련된 MLP 모델
            metrics: 성능 지표 딕셔너리
        """
        print("\n===== MLP 모델 훈련 =====")
        mlp_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        mlp_model.fit(self.X_train_scaled, self.y_train)
        y_pred = mlp_model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        print(f"MLP - 정확도: {accuracy:.4f}, F1 점수: {f1:.4f}")
        metrics = {'accuracy': accuracy, 'f1_score': f1}
        return mlp_model, metrics

    def calculate_mlp_permutation_importance(self, model, n_repeats=10, random_state=42):
        """
        MLP 모델의 순열 중요도 계산
        
        Args:
            model: 훈련된 MLP 모델
            n_repeats (int): 반복 횟수
            random_state (int): 랜덤 시드
        
        Returns:
            permutation_importance: 계산된 순열 중요도 결과
        """
        print("\n===== MLP 모델의 순열 중요도 계산 중 =====")
        result = permutation_importance(
            model, self.X_test_scaled, self.y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        print("MLP 모델의 순열 중요도 계산 완료!")
        return result

    def plot_mlp_permutation_importance_plotly(self, result, max_features=20):
        """
        MLP 모델의 순열 중요도 Plotly 시각화
        
        Args:
            result: 계산된 순열 중요도 결과
            max_features (int): 표시할 최대 피처 수
        """
        print("\n===== MLP 모델의 순열 중요도 Plotly 시각화 =====")
        importances = pd.DataFrame({
            'Feature': self.features.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values('Importance', ascending=False)
        top_features = importances.head(max_features)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_features['Feature'],
            x=top_features['Importance'],
            orientation='h',
            error_x=dict(type='data', array=top_features['Std']),
            marker_color='rgb(158,202,225)',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            opacity=0.8,
            name='Permutation Importance'
        ))
        fig.update_layout(
            title='MLP - Permutation Feature Importance',
            xaxis_title='Permutation Importance',
            yaxis_title='Feature',
            height=600,
            width=900,
            font=dict(family="Arial, sans-serif", size=14),
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='rgba(255,255,255,0.9)'
        )

        annotations = []
        for i, row in top_features.iterrows():
            annotations.append(dict(
                x=row['Importance'] + row['Std'] + max(top_features['Importance']) * 0.02,
                y=row['Feature'],
                text=f"{row['Importance']:.4f}",
                font=dict(family='Arial', size=12),
                showarrow=False,
                xanchor='left'
            ))
        fig.update_layout(annotations=annotations)
        fig.write_html('mlp_permutation_importance.html')
        fig.show()
        print("MLP 모델의 순열 중요도 Plotly 시각화 완료! ('mlp_permutation_importance.html'로 저장됨)")
        return fig

def main():
    file_path = "/Users/jh/PycharmProjects/lol_event_dataset.csv"  # 실제 파일 경로로 수정하세요.
    analyzer = ModelAnalyzer(file_path)

    if analyzer.load_data():
        analyzer.prepare_data(test_size=0.2, random_state=42)

        print("\n===== 스케일링된 데이터 통계 =====")
        print(f"훈련 데이터 범위: Min={analyzer.X_train_scaled_df.min().min():.4f}, Max={analyzer.X_train_scaled_df.max().max():.4f}")
        print(f"테스트 데이터 범위: Min={analyzer.X_test_scaled_df.min().min():.4f}, Max={analyzer.X_test_scaled_df.max().max():.4f}")

        rf_model, rf_metrics = analyzer.train_random_forest(n_estimators=100, max_depth=None)
        analyzer.plot_rf_feature_importance(rf_model, max_features=20)

        xgb_model, xgb_metrics = analyzer.train_xgboost(n_estimators=100, learning_rate=0.1, max_depth=6)
        xgb_shap_values, xgb_explainer = analyzer.calculate_xgb_shap_values(xgb_model)
        analyzer.plot_xgb_shap_beeswarm(xgb_shap_values, max_display=20)
        analyzer.plot_xgb_shap_summary(xgb_shap_values, max_display=20)
        analyzer.plot_xgb_shap_bar(xgb_shap_values, max_display=20)

        mlp_model, mlp_metrics = analyzer.train_mlp(hidden_layer_sizes=(100, 50), max_iter=500)
        perm_importance = analyzer.calculate_mlp_permutation_importance(mlp_model, n_repeats=10)
        analyzer.plot_mlp_permutation_importance_plotly(perm_importance, max_features=20)

        print("\n===== 모델 성능 요약 =====")
        print(f"랜덤 포레스트 - 정확도: {rf_metrics['accuracy']:.4f}, F1 점수: {rf_metrics['f1_score']:.4f}")
        print(f"XGBoost - 정확도: {xgb_metrics['accuracy']:.4f}, F1 점수: {xgb_metrics['f1_score']:.4f}")
        print(f"MLP - 정확도: {mlp_metrics['accuracy']:.4f}, F1 점수: {mlp_metrics['f1_score']:.4f}")

        print("\n===== 분석 완료 =====")
        print("시각화 결과가 다음 파일에 저장되었습니다:")
        print("- rf_feature_importance.png (랜덤 포레스트 - Feature Importance)")
        print("- xgb_shap_beeswarm.png (XGBoost - SHAP Beeswarm 플롯)")
        print("- xgb_shap_summary.png (XGBoost - SHAP Summary 플롯)")
        print("- xgb_shap_bar.png (XGBoost - SHAP Bar 플롯)")
        print("- mlp_permutation_importance.html (MLP - 순열 중요도 Plotly 시각화)")
    else:
        print("데이터 로드에 실패했습니다. 파일 경로를 확인해주세요.")

if __name__ == "__main__":
    main()
