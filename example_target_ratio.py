"""
==============================================
Example of ploting target flag ratio


==============================================

"""
from evaluation import plot_trg_ratio

n_samples = 100

dt = {
'target_flg': np.random.choice([0, 1], n_samples)
,'visits': np.random.poisson(lam=1,size=n_samples)
,'days': np.random.poisson(lam=2,size=n_samples)
,'weeks': np.random.poisson(lam=3,size=n_samples)
,'amount': np.random.poisson(lam=5,size=n_samples)
}

data = pd.DataFrame(dt)

plot_trg_ratio(data, 'target_flg', trg_value=1)
