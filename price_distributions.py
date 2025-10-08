
import numpy as np
import pandas as pd

class PriceDistributionGenerator:
    def __init__(self, T: int, N: int, seed: int = None):
        self.T = T
        self.N = N
        if seed is not None:
            np.random.seed(seed)

    def generate_normal(self, mean1: float, std1: float, mean2: float, std2: float):
        prices_s1 = np.random.normal(loc=mean1, scale=std1, size=(self.T, self.N))
        prices_s2 = np.random.normal(loc=mean2, scale=std2, size=(self.T, self.N))
        return pd.DataFrame(prices_s1), pd.DataFrame(prices_s2)

    def generate_lognormal(self, mean1: float, sigma1: float, mean2: float, sigma2: float):
        prices_s1 = np.random.lognormal(mean=mean1, sigma=sigma1, size=(self.T, self.N))
        prices_s2 = np.random.lognormal(mean=mean2, sigma=sigma2, size=(self.T, self.N))
        return pd.DataFrame(prices_s1), pd.DataFrame(prices_s2)

    def generate_gamma(self, shape1: float, scale1: float, shape2: float, scale2: float):
        prices_s1 = np.random.gamma(shape=shape1, scale=scale1, size=(self.T, self.N))
        prices_s2 = np.random.gamma(shape=shape2, scale=scale2, size=(self.T, self.N))
        return pd.DataFrame(prices_s1), pd.DataFrame(prices_s2)

    def generate_pareto(self, alpha1: float, scale1: float, alpha2: float, scale2: float):
        prices_s1 = scale1 * (np.random.pareto(a=alpha1, size=(self.T, self.N)) + 1)
        prices_s2 = scale2 * (np.random.pareto(a=alpha2, size=(self.T, self.N)) + 1)
        return pd.DataFrame(prices_s1), pd.DataFrame(prices_s2)

    def generate_weibull(self, a1: float, scale1: float, a2: float, scale2: float):
        prices_s1 = scale1 * np.random.weibull(a=a1, size=(self.T, self.N))
        prices_s2 = scale2 * np.random.weibull(a=a2, size=(self.T, self.N))
        return pd.DataFrame(prices_s1), pd.DataFrame(prices_s2)

    def generate_beta(self, a1: float, b1: float, scale1: float, a2: float, b2: float, scale2: float):
        prices_s1 = scale1 * np.random.beta(a=a1, b=b1, size=(self.T, self.N))
        prices_s2 = scale2 * np.random.beta(a=a2, b=b2, size=(self.T, self.N))
        return pd.DataFrame(prices_s1), pd.DataFrame(prices_s2)

    def generate_triangular(self, left1: float, mode1: float, right1: float,
                                  left2: float, mode2: float, right2: float):
        prices_s1 = np.random.triangular(left=left1, mode=mode1, right=right1, size=(self.T, self.N))
        prices_s2 = np.random.triangular(left=left2, mode=mode2, right=right2, size=(self.T, self.N))
        return pd.DataFrame(prices_s1), pd.DataFrame(prices_s2)
    

    def generate_by_name(self, dist: str, params: dict):
        """
        Dispatches to the appropriate distribution function based on dist name.

        Args:
            dist (str): Distribution name. Supported: 
                ['normal', 'lognormal', 'gamma', 'pareto', 'weibull', 'beta', 'triangular']
            params (dict): Dictionary of parameters specific to that distribution.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]
        """
        dist = dist.lower()
        if dist == "normal":
            return self.generate_normal(**params)
        elif dist == "lognormal":
            return self.generate_lognormal(**params)
        elif dist == "gamma":
            return self.generate_gamma(**params)
        elif dist == "pareto":
            return self.generate_pareto(**params)
        elif dist == "weibull":
            return self.generate_weibull(**params)
        elif dist == "beta":
            return self.generate_beta(**params)
        elif dist == "triangular":
            return self.generate_triangular(**params)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")


