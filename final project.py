"""
Bayesian Optimization Project
Scalable Hyperparameter Optimization via Bayesian Optimization with Custom Acquisition Functions

Tasks:
- Task 1: Implement Gaussian Process regression model + kernels (RBF, Matern 5/2) and posterior.
- Task 2: Implement acquisition functions (Expected Improvement, Upper Confidence Bound).
- Task 3: Integrate GP + acquisition into Bayesian Optimization loop for hyperparameter tuning.
- Task 4: Comparative analysis vs Random Search, reporting convergence and final performance.

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

# ============================================================
# Task 1: Implement Gaussian Process regression model + kernels
# ============================================================

@dataclass
class KernelParams:
    lengthscale: float = 1.0
    variance: float = 1.0

def rbf_kernel(X: np.ndarray, Y: np.ndarray, params: KernelParams) -> np.ndarray:
    """
    RBF (Squared Exponential) kernel.
    K(x, y) = variance * exp(- ||x - y||^2 / (2 * lengthscale^2))
    """
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
    Gaussian Process regressor for scalar outputs.

    - Supports RBF and Matern 5/2 kernels (Task 1).
    - Posterior via Cholesky for numerical stability (Task 1).
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
        K[np.diag_indices_from(K)] += (self.noise_variance + self.jitter)

        try:
            c, lower = cho_factor(K, check_finite=False)
        except np.linalg.LinAlgError:
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

        K_star = self.kernel_fn(self.X_train, X_star, self.kernel_params)  # (n_train, n_star)
        mu = K_star.T @ self._alpha

        if not return_var:
            return mu, None

        v = cho_solve(self._cho, K_star, check_finite=False)
        K_star_star = self.kernel_fn(X_star, X_star, self.kernel_params)
        var = np.maximum(np.diag(K_star_star) - np.sum(K_star * v, axis=0), 0.0)
        return mu, var

# ============================================================
# Task 2: Implement acquisition functions (EI and UCB)
# ============================================================

def expected_improvement(mu: np.ndarray, sigma2: np.ndarray, best_f: float, xi: float = 1e-3) -> np.ndarray:
    """
    Expected Improvement (maximize) (Task 2).
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
    Upper Confidence Bound (maximize) (Task 2).
    UCB = mu + sqrt(beta) * sigma
    """
    sigma = np.sqrt(np.maximum(sigma2, 0.0))
    return mu + np.sqrt(beta) * sigma

# ============================================================
# Search space and helpers (used in Task 3 and Task 4)
# ============================================================

SPACE = {
    "n_estimators": (50, 500),       # integer
    "learning_rate": (0.01, 0.2),    # float
    "max_depth": (2, 8),             # integer
    "subsample": (0.6, 1.0),         # float
}

def sample_uniform(n: int) -> np.ndarray:
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

def train_and_eval(params: Dict[str, float],
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Train GradientBoostingRegressor and return validation RMSE (minimize).
    Used by Task 3 and Task 4.
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
    # Compute RMSE manually for compatibility (no 'squared' arg)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# ============================================================
# Task 3: Integrate GP + acquisition into Bayesian Optimization loop
# ============================================================

class BayesianOptimizer:
    """
    Bayesian Optimization loop (Task 3):
    - Start with n_init random evaluations.
    - Fit GP on f(x) = -RMSE to convert to maximization.
    - Sample candidate points and pick x maximizing acquisition (EI/UCB).
    - Evaluate, update, repeat.
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
        One BO step:
        - Fit GP (Task 1)
        - Compute acquisition values (Task 2)
        - Select next params and evaluate objective
        """
        self.gp.fit(self.X_obs, self.y_obs)

        X_cand = sample_uniform(self.n_candidates)
        mu, var = self.gp.predict(X_cand, return_var=True)

        if self.acquisition == "EI":
            best_f = float(np.max(self.y_obs))
            xi = float(self.acq_params.get("xi", 1e-3))
            acq_vals = expected_improvement(mu, var, best_f, xi)  # Task 2 used here
        elif self.acquisition == "UCB":
            beta = float(self.acq_params.get("beta", 2.0))
            acq_vals = upper_confidence_bound(mu, var, beta)       # Task 2 used here
        else:
            raise ValueError("Unknown acquisition: choose 'EI' or 'UCB'.")

        x_next = X_cand[int(np.argmax(acq_vals))]
        params = array_to_params(x_next)
        rmse = objective_fn(params)
        f_val = -rmse

        self.X_obs = np.vstack([self.X_obs, x_next])
        self.y_obs = np.hstack([self.y_obs, f_val])

        return params, rmse

    def run(self, objective_fn: Callable[[Dict[str, float]], float], n_init: int = 5, n_iter: int = 25) -> Dict:
        """
        Run BO for a fixed budget.
        Returns logs including per-iteration best RMSE.
        """
        # Initialize random evaluations (Task 3 setup)
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

        return {"best_params": best_params, "best_rmse": best_rmse, "history": history}

# ============================================================
# Task 4: Comparative analysis vs Random Search
# ============================================================

def random_search(objective_fn: Callable[[Dict[str, float]], float], n_trials: int = 30) -> Dict:
    """
    Random Search baseline (Task 4).
    """
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

# ============================================================
# Experiment driver: dataset + objective + runs
# Shows where each Task is used in practice
# ============================================================

def main():
    # Dataset: high-dimensional synthetic regression
    X, y = make_regression(
        n_samples=5000,
        n_features=50,
        n_informative=40,
        noise=10.0,
        random_state=RNG_SEED
    )

    # Splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RNG_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RNG_SEED)

    # Objective closure used by Task 3 and Task 4
    def objective(params: Dict[str, float]) -> float:
        return train_and_eval(params, X_train, y_train, X_val, y_val)

    # ----- Task 4: Random Search baseline -----
    rs_result = random_search(objective, n_trials=35)

    # ----- Task 3: Bayesian Optimization with EI -----
    bo_ei = BayesianOptimizer(kernel_name="matern52", acquisition="EI", acq_params={"xi": 1e-3})
    ei_result = bo_ei.run(objective, n_init=5, n_iter=30)

    # ----- Task 3: Bayesian Optimization with UCB -----
    bo_ucb = BayesianOptimizer(kernel_name="matern52", acquisition="UCB", acq_params={"beta": 2.0})
    ucb_result = bo_ucb.run(objective, n_init=5, n_iter=30)

    # Evaluate best configs on test set (final deliverable summary)
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
        # Compute RMSE manually for compatibility (no 'squared' arg)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return rmse

    rs_test_rmse = eval_on_test(rs_result["best_params"])
    ei_test_rmse = eval_on_test(ei_result["best_params"])
    ucb_test_rmse = eval_on_test(ucb_result["best_params"])

    # ----- Text deliverables: convergence and final summaries (Task 4 reporting) -----
    print("\n=== Convergence (Validation RMSE) ===")

    def print_history(name: str, hist: List[Dict]):
        print(f"\n{name}:")
        print("iter | rmse_current | rmse_best_so_far")
        for h in hist:
            print(f"{h['iter']:>4} | {h['rmse']:.4f}      | {h['best_rmse']:.4f}")

    print_history("Random Search (Task 4)", rs_result["history"])
    print_history("BayesOpt - EI (Task 3 uses Task 1 & Task 2)", ei_result["history"])
    print_history("BayesOpt - UCB (Task 3 uses Task 1 & Task 2)", ucb_result["history"])

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

    # Simple ASCII convergence curves (optional visualization)
    def ascii_curve(hist: List[Dict], label: str):
        bests = [h["best_rmse"] for h in hist]
        mn, mx = min(bests), max(bests)
        rngv = mx - mn if mx > mn else 1e-12
        chars = "▁▂▃▄▅▆▇█"
        line = "".join(chars[min(int((mx - b) / rngv * (len(chars) - 1)), len(chars) - 1)] for b in bests)
        print(f"{label}: {line}  (lower is better)")

    print("\n=== ASCII convergence curves (best validation RMSE) ===")
    ascii_curve(rs_result["history"], "Random Search (Task 4)")
    ascii_curve(ei_result["history"], "EI (Task 3 uses Task 1 & Task 2)")
    ascii_curve(ucb_result["history"], "UCB (Task 3 uses Task 1 & Task 2)")

if __name__ == "__main__":
    main()
