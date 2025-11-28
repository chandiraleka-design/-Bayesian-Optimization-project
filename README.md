Yesterday
Scalable Hyperparameter Optimization via Bayesian Optimization with Custom Acquisition Functions This project requires advanced students to implement and rigorously test a Bayesian Optimization framework for hyperparameter tuning of a complex machine learning model (e.g., Gradient Boosting Machine or a deep neural network surrogate). Instead of relying solely on off-the-shelf tools like Hyperopt or Optuna, students must implement the core components of the Gaussian Process (GP) regression model and at least two distinct acquisition functions (e.g., Expected Improvement (El) and Upper Confidence Bound (UCB)) from scratch using NumPy/SciPy. The target model should be trained on a large, publicly available dataset like the California Housing or a complex synthetic dataset generated using scikit-learn's 'make_regression' with high dimensionality and noise, ensuring the optimization process is computationally significant. The core challenge lies in efficiently updating the GP model (Kronecker product for large datasets if applicable, or efficient matrix inversion) and demonstrating how the choice of acquisition function impacts convergence speed and final model performance compared to grid search or random search. The final submission must include the complete, documented Python implementation and a comparative analysis. Tasks to Complete 1 Implement the core Gaussian Process (GP) regression model, including kernel definitions (e.g., Matern 5/2 or RBF) and the analytical solution for posterior mean and covariance. 2 Implement two distinct acquisition functions (Expected Improvement (EI) and Upper Confidence Bound (UCB)) optimized for numerical stability and efficiency. 3 Integrate the GP model and acquisition functions into a complete Bayesian Optimization loop to tune the hyperparameters of a complex model (e.g., XGBoostRegressor) on the chosen dataset. 4 Conduct a rigorous comparison: run the Bayesian Optimization loop (using both acquisition functions) and compare convergence speed (number of iterations to reach a performance threshold) and final performance metrics against standard Random Search. Expected Deliverables 1 Complete, production-quality Python code implementation of the Bayesian Optimization framework (GP model, acquisition functions, and optimizer loop). 2 Text-based report detailing the mathematical derivation and Implementation choices for the GP kernel and acquisition functions. 3 Comparative analysis output (as text) showing convergence plots (performance vs. iteration count) for Random Search, El, and UCB strategies. 4 Final evaluation summary of the best hyperparameters found and the resulting model performance metrics give me the code for this question and using tasks and give me the output as expected deliverable as a beginner level
Scalable Bayesian optimization with custom GP and acquisition functions
You’re building something real here, Harini: from-scratch Gaussian Processes and acquisition strategies, wrapped in a clean, testable Bayesian Optimization loop. Below is a beginner-friendly, production-quality implementation with clear structure, docstrings, and comments, plus a text report and example outputs to match the expected deliverables.

How to run
Environment: Python 3.9+, NumPy, SciPy, scikit-learn, Matplotlib.

Install: pip install numpy scipy scikit-learn matplotlib

Run: Save the code below as bo_gp.py and run python bo_gp.py. It will:

Generate a high-dimensional regression dataset.

Tune GradientBoostingRegressor via Bayesian Optimization using EI and UCB.

Compare to Random Search.

Print text-based convergence tables and final evaluation summaries.

Complete Python implementation
python
"""
Scalable Bayesian Optimization with custom Gaussian Process (GP) and acquisition functions (EI, UCB).
Beginner-friendly, documented, and modular.

Author: Harini (structure tailored for clarity, correctness, and reproducibility)

Key components:
- GaussianProcess: RBF and Matern 5/2 kernels, posterior via Cholesky for numerical stability.
- Acquisition: Expected Improvement (EI) and Upper Confidence Bound (UCB).
- BayesianOptimizer: integrates GP and acquisition to tune GradientBoostingRegressor hyperparameters.
- Baselines: Random Search for comparison.
- Dataset: High-dimensional synthetic regression via scikit-learn's make_regression.

Notes:
- We treat the objective as validation RMSE to minimize.
- GP models f(x) ~ negative RMSE (so we maximize f). This keeps EI/UCB standard (maximize).
- Acquisition maximization via uniform sampling of candidates (beginner-friendly and robust).
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
from scipy.linalg import cho_factor, cho_solve
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings

# -----------------------
# Utility: Reproducibility
# -----------------------
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
np.random.seed(RNG_SEED)


# -----------------------
# Gaussian Process Core
# -----------------------
@dataclass
class KernelParams:
    lengthscale: float = 1.0
    variance: float = 1.0


def rbf_kernel(X: np.ndarray, Y: np.ndarray, params: KernelParams) -> np.ndarray:
    """
    Radial Basis Function (Squared Exponential) kernel.
    K(x, y) = variance * exp(- ||x - y||^2 / (2 * lengthscale^2))
    """
    # Efficient pairwise squared distances
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    cross = X @ Y.T
    dists = X_sq + Y_sq - 2.0 * cross
    return params.variance * np.exp(-0.5 * dists / (params.lengthscale**2))


def matern52_kernel(X: np.ndarray, Y: np.ndarray, params: KernelParams) -> np.ndarray:
    """
    Matern 5/2 kernel.
    K(r) = variance * (1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) * exp(-sqrt(5) r / l)
    """
    # Compute pairwise Euclidean distances
    # For numerical stability, clip tiny negatives due to float error
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    cross = X @ Y.T
    d2 = np.maximum(X_sq + Y_sq - 2.0 * cross, 0.0)
    r = np.sqrt(d2)
    l = params.lengthscale
    sqrt5_r_l = np.sqrt(5.0) * r / l
    term = (1.0 + sqrt5_r_l + 5.0 * (r**2) / (3.0 * l**2))
    return params.variance * term * np.exp(-sqrt5_r_l)


class GaussianProcess:
    """
    Basic Gaussian Process regressor for scalar outputs.

    - Supports RBF and Matern 5/2 kernels via call-in function.
    - Posterior computed via Cholesky decomposition for stability.
    - Jitter term added to diagonal to handle numerical issues.
    """

    def __init__(
        self,
        kernel_fn: Callable[[np.ndarray, np.ndarray, KernelParams], np.ndarray],
        kernel_params: KernelParams,
        noise_variance: float = 1e-6,
        jitter: float = 1e-8,
    ):
        self.kernel_fn = kernel_fn
        self.kernel_params = kernel_params
        self.noise_variance = noise_variance
        self.jitter = jitter
        self.X_train = None
        self.y_train = None
        self._cho = None
        self._alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GP: compute K + sigma^2 I and its Cholesky, plus alpha = (K + sigma^2 I)^{-1} y.
        """
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64).reshape(-1)

        K = self.kernel_fn(self.X_train, self.X_train, self.kernel_params)
        # Add noise variance and jitter to diagonal
        K[np.diag_indices_from(K)] += (self.noise_variance + self.jitter)

        # Cholesky factorization
        try:
            c, lower = cho_factor(K, check_finite=False)
        except np.linalg.LinAlgError:
            # Fallback: add more jitter if needed
            warnings.warn("Cholesky failed; adding extra jitter.")
            K[np.diag_indices_from(K)] += 1e-6
            c, lower = cho_factor(K, check_finite=False)

        self._cho = (c, lower)
        self._alpha = cho_solve(self._cho, self.y_train, check_finite=False)

    def predict(self, X_star: np.ndarray, return_var: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Posterior mean and variance at test points X_star.
        """
        if self.X_train is None:
            raise ValueError("GP not fitted.")

        K_star = self.kernel_fn(self.X_train, X_star, self.kernel_params)  # shape (n_train, n_star)
        # Posterior mean: mu = K_*^T (K + sigma^2 I)^{-1} y = K_*^T alpha
        mu = K_star.T @ self._alpha

        if not return_var:
            return mu, None

        # Posterior variance: k(X_star, X_star) - K_*^T (K + sigma^2 I)^{-1} K_*
        v = cho_solve(self._cho, K_star, check_finite=False)  # solution of (K+...)*v = K_*
        K_star_star = self.kernel_fn(X_star, X_star, self.kernel_params)
        var = np.maximum(np.diag(K_star_star) - np.sum(K_star * v, axis=0), 0.0)
        return mu, var


# -----------------------
# Acquisition Functions
# -----------------------
def expected_improvement(mu: np.ndarray, sigma2: np.ndarray, best_f: float, xi: float = 1e-4) -> np.ndarray:
    """
    Expected Improvement (maximize). We assume we maximize f(x).
    EI = E[max(0, f(x) - best_f - xi)]
    """
    sigma = np.sqrt(np.maximum(sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - best_f - xi) / (sigma + 1e-12)
        from scipy.stats import norm
        ei = (mu - best_f - xi) * norm.cdf(z) + (sigma) * norm.pdf(z)
        ei[sigma < 1e-12] = 0.0
    return ei


def upper_confidence_bound(mu: np.ndarray, sigma2: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """
    Upper Confidence Bound (maximize).
    UCB = mu + sqrt(beta) * sigma
    """
    sigma = np.sqrt(np.maximum(sigma2, 0.0))
    return mu + np.sqrt(beta) * sigma


# -----------------------
# Objective: model training and validation
# -----------------------
def train_and_eval(params: Dict[str, float], X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Train GradientBoostingRegressor with given hyperparameters and return validation RMSE (to MINIMIZE).
    """
    model = GradientBoostingRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        random_state=RNG_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    return rmse


# -----------------------
# Search Space
# -----------------------
SPACE = {
    "n_estimators": (50, 500),       # integer
    "learning_rate": (0.01, 0.2),    # float
    "max_depth": (2, 8),             # integer
    "subsample": (0.6, 1.0),         # float
}

def sample_uniform(n: int) -> np.ndarray:
    """
    Sample n points uniformly from the search space [lower, upper] in each dim.
    Returns array of shape (n, d) in continuous form; integers will be rounded when evaluated.
    """
    dims = list(SPACE.keys())
    lows = np.array([SPACE[d][0] for d in dims], dtype=np.float64)
    highs = np.array([SPACE[d][1] for d in dims], dtype=np.float64)
    X = rng.uniform(lows, highs, size=(n, len(dims)))
    return X

def array_to_params(x: np.ndarray) -> Dict[str, float]:
    dims = list(SPACE.keys())
    p = {}
    for i, d in enumerate(dims):
        low, high = SPACE[d]
        val = float(x[i])
        if d in ["n_estimators", "max_depth"]:
            p[d] = int(np.clip(np.round(val), low, high))
        else:
            p[d] = float(np.clip(val, low, high))
    return p


# -----------------------
# Bayesian Optimizer
# -----------------------
class BayesianOptimizer:
    """
    Simple Bayesian Optimization loop:
    - Start with n_init random evaluations.
    - Fit GP on f(x) where f = -RMSE (so larger is better).
    - At each iteration, sample candidate points uniformly and pick x maximizing acquisition.
    - Evaluate objective, update dataset, repeat.
    """

    def __init__(
        self,
        kernel_name: str = "matern52",
        lengthscale: float = 1.5,
        variance: float = 1.0,
        noise_variance: float = 1e-6,
        acquisition: str = "EI",
        acq_params: Dict = None,
        n_candidates: int = 2000,
    ):
        self.kernel_name = kernel_name
        self.acquisition = acquisition.upper()
        self.acq_params = acq_params or {}
        self.n_candidates = n_candidates

        kernel_fn = matern52_kernel if kernel_name.lower() == "matern52" else rbf_kernel
        self.gp = GaussianProcess(kernel_fn, KernelParams(lengthscale, variance), noise_variance)

        self.X_obs = None
        self.y_obs = None

    def step(self, objective_fn: Callable[[Dict[str, float]], float]) -> Tuple[Dict[str, float], float]:
        """
        One BO step: fit GP, propose x via acquisition, evaluate objective.
        """
        # Fit GP on observed data: GP models f = -RMSE
        self.gp.fit(self.X_obs, self.y_obs)

        # Propose: candidates + acquisition
        X_cand = sample_uniform(self.n_candidates)
        mu, var = self.gp.predict(X_cand, return_var=True)

        if self.acquisition == "EI":
            best_f = float(np.max(self.y_obs))
            xi = float(self.acq_params.get("xi", 1e-3))
            acq_vals = expected_improvement(mu, var, best_f, xi)
        elif self.acquisition == "UCB":
            beta = float(self.acq_params.get("beta", 2.0))
            acq_vals = upper_confidence_bound(mu, var, beta)
        else:
            raise ValueError("Unknown acquisition: choose 'EI' or 'UCB'.")

        x_next = X_cand[int(np.argmax(acq_vals))]
        params = array_to_params(x_next)
        rmse = objective_fn(params)
        f_val = -rmse

        # Update observations
        self.X_obs = np.vstack([self.X_obs, x_next])
        self.y_obs = np.hstack([self.y_obs, f_val])

        return params, rmse

    def run(
        self,
        objective_fn: Callable[[Dict[str, float]], float],
        n_init: int = 5,
        n_iter: int = 25
    ) -> Dict:
        """
        Run BO for a fixed budget.
        Returns logs including per-iteration best RMSE.
        """
        # Initialize
        self.X_obs = sample_uniform(n_init)
        init_params = [array_to_params(x) for x in self.X_obs]
        init_rmses = [objective_fn(p) for p in init_params]
        self.y_obs = -np.array(init_rmses, dtype=np.float64)

        best_rmse = float(np.min(init_rmses))
        best_params = init_params[int(np.argmin(init_rmses))]

        history = []
        for t in range(1, n_iter + 1):
            params, rmse = self.step(objective_fn)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
            history.append({"iter": t, "params": params, "rmse": rmse, "best_rmse": best_rmse})

        return {
            "best_params": best_params,
            "best_rmse": best_rmse,
            "history": history,
        }


# -----------------------
# Baseline: Random Search
# -----------------------
def random_search(objective_fn: Callable[[Dict[str, float]], float], n_trials: int = 30) -> Dict:
    best_rmse = np.inf
    best_params = None
    history = []
    for i in range(1, n_trials + 1):
        x = sample_uniform(1)[0]
        params = array_to_params(x)
        rmse = objective_fn(params)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
        history.append({"iter": i, "params": params, "rmse": rmse, "best_rmse": best_rmse})
    return {"best_params": best_params, "best_rmse": best_rmse, "history": history}


# -----------------------
# Main experiment
# -----------------------
def main():
    # Create high-dimensional regression dataset
    X, y = make_regression(
        n_samples=5000,
        n_features=50,
        n_informative=40,
        noise=10.0,
        random_state=RNG_SEED
    )

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RNG_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RNG_SEED)

    # Objective function closure
    def objective(params: Dict[str, float]) -> float:
        return train_and_eval(params, X_train, y_train, X_val, y_val)

    # Baseline: Random Search
    rs_result = random_search(objective, n_trials=35)

    # Bayesian Optimization with EI
    bo_ei = BayesianOptimizer(kernel_name="matern52", acquisition="EI", acq_params={"xi": 1e-3})
    ei_result = bo_ei.run(objective, n_init=5, n_iter=30)

    # Bayesian Optimization with UCB
    bo_ucb = BayesianOptimizer(kernel_name="matern52", acquisition="UCB", acq_params={"beta": 2.0})
    ucb_result = bo_ucb.run(objective, n_init=5, n_iter=30)

    # Evaluate best models on test set
    def eval_on_test(params: Dict[str, float]) -> float:
        model = GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=RNG_SEED
        )
        model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse

    rs_test_rmse = eval_on_test(rs_result["best_params"])
    ei_test_rmse = eval_on_test(ei_result["best_params"])
    ucb_test_rmse = eval_on_test(ucb_result["best_params"])

    # ---- Text outputs (beginner-friendly) ----
    print("\n=== Convergence (Validation RMSE) ===")
    def print_history(name: str, hist: List[Dict]):
        print(f"\n{name}:")
        print("iter | rmse_current | rmse_best_so_far")
        for h in hist:
            print(f"{h['iter']:>4} | {h['rmse']:.4f}      | {h['best_rmse']:.4f}")

    print_history("Random Search", rs_result["history"])
    print_history("BayesOpt - EI", ei_result["history"])
    print_history("BayesOpt - UCB", ucb_result["history"])

    print("\n=== Final evaluation summary ===")
    print(f"- Random Search best val RMSE: {rs_result['best_rmse']:.4f}")
    print(f"- EI best val RMSE:            {ei_result['best_rmse']:.4f}")
    print(f"- UCB best val RMSE:           {ucb_result['best_rmse']:.4f}")

    print("\nBest hyperparameters found:")
    print(f"- Random Search: {rs_result['best_params']}")
    print(f"- EI:             {ei_result['best_params']}")
    print(f"- UCB:            {ucb_result['best_params']}")

    print("\nTest RMSE of best configs (trained on train+val):")
    print(f"- Random Search: {rs_test_rmse:.4f}")
    print(f"- EI:            {ei_test_rmse:.4f}")
    print(f"- UCB:           {ucb_test_rmse:.4f}")

    # Optional ASCII convergence sketch (sparkline-like)
    def ascii_curve(hist: List[Dict], label: str):
        bests = [h["best_rmse"] for h in hist]
        mn, mx = min(bests), max(bests)
        rngv = mx - mn if mx > mn else 1e-12
        chars = "▁▂▃▄▅▆▇█"
        line = "".join(chars[min(int((mx - b) / rngv * (len(chars) - 1)), len(chars) - 1)] for b in bests)
        print(f"{label}: {line}  (lower is better)")

    print("\n=== ASCII convergence curves (best validation RMSE) ===")
    ascii_curve(rs_result["history"], "Random Search")
    ascii_curve(ei_result["history"], "EI")
    ascii_curve(ucb_result["history"], "UCB")


if __name__ == "__main__":
    main()
Mathematical and implementation report
This gives you the core derivations and choices, kept straightforward and traceable.

Gaussian process posterior
Prior: We model 
𝑓
∼
𝐺
𝑃
(
0
,
𝑘
(
⋅
,
⋅
)
)
.

Kernel choices:

RBF:

𝑘
RBF
(
𝑥
,
𝑥
′
)
=
𝜎
𝑓
2
exp
⁡
(
−
∥
𝑥
−
𝑥
′
∥
2
2
ℓ
2
)
Matern 5/2:

𝑘
𝜈
=
5
/
2
(
𝑟
)
=
𝜎
𝑓
2
(
1
+
5
𝑟
ℓ
+
5
𝑟
2
3
ℓ
2
)
exp
⁡
(
−
5
𝑟
ℓ
)
where 
𝑟
=
∥
𝑥
−
𝑥
′
∥
, 
ℓ
 is the lengthscale, and 
𝜎
𝑓
2
 is the signal variance.

Posterior mean and variance: Given 
𝑋
∈
𝑅
𝑛
×
𝑑
, observations 
𝑦
∈
𝑅
𝑛
, and test inputs 
𝑋
∗
,

𝐾
=
𝑘
(
𝑋
,
𝑋
)
+
𝜎
𝑛
2
𝐼
,
𝐾
∗
=
𝑘
(
𝑋
,
𝑋
∗
)
,
𝐾
∗
∗
=
𝑘
(
𝑋
∗
,
𝑋
∗
)
𝜇
∗
=
𝐾
∗
⊤
𝐾
−
1
𝑦
,
Σ
∗
=
𝐾
∗
∗
−
𝐾
∗
⊤
𝐾
−
1
𝐾
∗
We compute 
𝐾
−
1
𝑦
 and 
𝐾
−
1
𝐾
∗
 via Cholesky factorization for numerical stability, adding a small jitter on the diagonal.

Acquisition functions
Expected Improvement (EI), maximize:

EI
(
𝑥
)
=
𝐸
[
max
⁡
(
0
,
𝑓
(
𝑥
)
−
𝑓
⋆
−
𝜉
)
]
With predictive normal 
𝑁
(
𝜇
,
𝜎
2
)
, this yields:

𝑧
=
𝜇
−
𝑓
⋆
−
𝜉
𝜎
,
EI
=
(
𝜇
−
𝑓
⋆
−
𝜉
)
Φ
(
𝑧
)
+
𝜎
𝜙
(
𝑧
)
where 
Φ
 and 
𝜙
 are CDF and PDF of the standard normal, and 
𝜉
 promotes exploration.

Upper Confidence Bound (UCB), maximize:

UCB
(
𝑥
)
=
𝜇
(
𝑥
)
+
𝛽
 
𝜎
(
𝑥
)
Larger 
𝛽
 increases exploration.

Objective transformation
We minimize RMSE but GP/acquisitions assume maximization.

Define 
𝑓
(
𝑥
)
=
−
RMSE
(
𝑥
)
. EI and UCB operate directly on 
𝑓
.

Best observed value becomes 
𝑓
⋆
=
max
⁡
𝑖
𝑓
(
𝑥
𝑖
)
=
−
min
⁡
𝑖
RMSE
(
𝑥
𝑖
)
.

Efficiency and numerical stability
Cholesky factorization: Stable inversion of 
𝐾
 without explicit matrix inverse.

Jitter: Adds 
10
−
8
 on the diagonal to avoid near-singular matrices.

Candidate maximization: Uniform sampling over the bounded hyperparameter space for robustness and simplicity. For larger budgets, combine random sampling with local refinement.

Comparative analysis outputs (text)
Below is the structure your run will print. The exact numbers will vary when you execute the script.

Convergence tables: Iteration-wise current RMSE and best-so-far RMSE for Random Search, EI, and UCB.

ASCII curves: Compact visual of best validation RMSE over iterations; lower is better.

Example (illustrative):
