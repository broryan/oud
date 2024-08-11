from pycaret.datasets import get_data
from pycaret.anomaly import *

df = create_four_clusters_test_data()
setup(data=df)
knn = create_model('knn')
knn_predictions = predict_model(model=knn, data=df)
