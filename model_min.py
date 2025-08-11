import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

def plot_feature_importances_matplotlib(model, X_test, feature_names, model_name="Model", top_n=20, y_test=None):
    """
    주어진 모델에 대해 상위 top_n개 피처 중요도를 matplotlib 가로 막대 그래프로 시각화합니다.
    
    :param model: 학습된 모델 객체
    :param X_test: 테스트 데이터 (DataFrame)
    :param feature_names: 피처 이름 리스트
    :param model_name: 그래프 제목에 사용할 모델 이름
    :param top_n: 표시할 상위 피처 수 (기본 20)
    :param y_test: permutation importance 계산에 사용할 타겟 (필요시)
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        if y_test is None:
            raise ValueError("y_test must be provided for permutation importance calculation.")
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importance = result.importances_mean

    fi_series = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    fi_top = fi_series.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(fi_top.index[::-1], fi_top.values[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.tight_layout()
    plt.show()

def remove_ngram_feature(df):
    """
    주어진 DataFrame에서 컬럼명이 "ngram2_" 또는 "ngram3_"로 시작하는 피처들을 제거한 DataFrame을 반환합니다.
    """
    cols_to_remove = [col for col in df.columns if col.startswith("ngram2_") or col.startswith("ngram3_")]
    new_df = df.drop(columns=cols_to_remove, errors='ignore')
    new_df
    return new_df

class MinWindowModelEvaluator:
    def __init__(self, csv_file: str):
        """
        :param csv_file: 저장된 CSV 파일 경로.
        데이터 형식 예시:
        {
            "minute": 41,
            "gold_delta": 1,
            "total_occurrences": 5,
            "entropy": 2.322,
            "ngram2": {('j','m'): 1, ...},
            "ngram3": {('j','m','a'): 1, ...},
            "t_patterns": { ... },
            "sequential_patterns": {('m','s'): 1, ...},
            "utterance_counts": {'j': 1, 'm': 1, 'a': 1, 's': 1, 't': 1}
        }
        타겟은 "gold_delta"가 아닌 이벤트 데이터셋에서는 보통 "success"로 예측하지만,
        여기서는 기존 코드를 그대로 사용하므로 타겟은 "gold_delta"로 가정합니다.
        피처는 "minute", "gold_delta"를 제외한 나머지 컬럼들을 사용합니다.
        """
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)
        self.target = self.data["gold_delta"]
        self.features = self.data.drop(columns=["minute", "gold_delta"])
    
    def check_missing(self):
        missing_counts = self.data.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        if missing_columns.empty:
            print("No missing values detected in the dataset.")
        else:
            print("Missing values detected in the following columns:")
            for col, count in missing_columns.items():
                print(f"{col}: {count} missing value(s)")
    
    
    def preprocess(self):
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        features_imputed = pd.DataFrame(
            imputer.fit_transform(self.features),
            columns=self.features.columns
        )
        scaler = MinMaxScaler()
        self.features_scaled = pd.DataFrame(
            scaler.fit_transform(features_imputed),
            columns=self.features.columns
        )
    
    def remove_ngram_features(self):
    
        self.features_scaled = remove_ngram_feature(self.features_scaled)
    
    
    def evaluate_models(self, use_smote=True, use_pca=True, n_components=0.95):
        # 전처리 후 n-gram 피처 제거
        self.preprocess()
        self.remove_ngram_features()
        
        X = self.features_scaled
        y = self.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Original training target distribution:", Counter(y_train))
        print("Test target distribution:", Counter(y_test))
        if use_smote:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print("After SMOTE:", Counter(y_train))
        self.X_test = X_test 
        print(self.X_test)
        self.y_test = y_test
        print(self.y_test)

        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
            "MLP": MLPClassifier(max_iter=1000, random_state=42)
        }
        self.trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"{name}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
    
    def plot_feature_importances(self, top_n=20, use_pca=True):
        """
        Matplotlib을 사용하여 상위 top_n개 피처 중요도를 가로 막대 그래프로 시각화합니다.
        PCA를 적용한 경우, PCA 축소된 피처 이름을 사용합니다.
        """
        if not hasattr(self, "trained_models"):
            print("Models are not trained. Run evaluate_models() first.")
            return
        
        # 사용한 피처 세트에 따라 feature names 결정
        if use_pca and hasattr(self, "features_pca"):
            feature_names = self.features_pca.columns
        else:
            feature_names = self.features_scaled.columns
        
        for name, model in self.trained_models.items():
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0])
            else:
                result = permutation_importance(model, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1)
                importance = result.importances_mean
            
            fi_series = pd.Series(importance, index=feature_names).sort_values(ascending=False)
            fi_top = fi_series.head(top_n)
            plt.figure(figsize=(10, 8))
            plt.barh(fi_top.index[::-1], fi_top.values[::-1])
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances - {name}")
            plt.tight_layout()
            plt.show()
    
    def plot_confusion_matrices(self):
        """
        각 학습된 모델에 대해 테스트 데이터셋을 대상으로 confusion matrix를 계산하고, 
        matplotlib를 사용하여 시각화합니다.
        """
        for name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(6, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix - {name}")
            plt.colorbar()
            classes = np.unique(self.y_test)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    csv_file = "/Users/jh/PycharmProjects/mindata_CI_10000.csv"  # 실제 CSV 파일 경로로 수정하세요.
    evaluator = MinWindowModelEvaluator(csv_file)
    evaluator.check_missing()
    evaluator.evaluate_models(use_smote=True, use_pca=True, n_components=0.95)
    evaluator.plot_feature_importances(top_n=20)
    evaluator.plot_confusion_matrices()


