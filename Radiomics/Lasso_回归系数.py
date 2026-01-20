# 核心功能: 绘制LASSO回归系数路径图
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def plot_lasso_path(data: pd.DataFrame, n_alphas: int = 100, cv_folds: int = 5,
                    random_state: int = 42, max_iter: int = 1000, output_path: str = None):
    features = [c for c in data.columns if c not in ["Patient", "Target"]]
    X, y = data[features], data["Target"]
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_features]

    X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))

    alphas = np.logspace(np.log10(1e-3), np.log10(10), n_alphas)
    best_alpha = LassoCV(alphas=alphas, cv=cv_folds,
                         random_state=random_state, max_iter=max_iter,
                         n_jobs=-1).fit(X, y).alpha_

    coef_paths = []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=max_iter, random_state=random_state)
        lasso.fit(X, y)
        coef_paths.append(lasso.coef_)
    coef_paths = np.array(coef_paths)

    non_zero = (np.abs(coef_paths) > 1e-5).sum(1)
    plt.figure(figsize=(8, 6))
    for i, c in enumerate(numeric_features):
        plt.plot(np.log10(alphas), coef_paths[:, i], linewidth=1.2)
    plt.axvline(np.log10(best_alpha), ls=':', color='k')

    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    idx = range(0, len(alphas), max(1, len(alphas) // 8))
    ax2.set_xticks([np.log10(alphas[i]) for i in idx])
    ax2.set_xticklabels([str(non_zero[i]) for i in idx])
    ax2.set_xlabel('Number of Features')

    # plt.xlabel('Log Lambda')
    # plt.ylabel('系数')

    plt.xlabel('')
    plt.ylabel('')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return best_alpha


if __name__ == "__main__":
    data = pd.read_excel(r"F:\Data\Jmzxyy\Osteoporosis\Radiomics\20250618\total.xlsx")
    plot_lasso_path(data, n_alphas=100, cv_folds=5,
                    output_path=r"F:\Data\Jmzxyy\Osteoporosis\Radiomics\20250618\Lasso_Coefficient_Path.png")
