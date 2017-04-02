"""
==============================================
Example of ploting learning curve

==============================================

"""
from evaluation import plot_trg_ratio
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
fit_func = model.fit

n_sample_seq =  [100, 1000, 2000, 3000, 4000, 6000, 7000, 8000, 10000]
lscore = get_learning_curve_score(fit_func, X_train, y_train, X_test, y_test, )
lscore_df = pd.DataFrame(lscore)

lscore_df.plot(x='n_sample', y=['train','test'], title='Learning Curve Plot',ylim=(0, 1.0))
