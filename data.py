import numpy as np
import urllib.request
urllib.request.urlretrieve(
'https://hub.jovian.ml/wp-content/uploads/2020/08/climate.csv','climate.txt')

#climate_data = np.array([[73,67,43],
                        #[91,88,64],
                        #[87,134,58],
                        #[102,43,37],
                        #[69,96,70]])
climate_data = (np.genfromtxt('climate.txt', delimiter = ',', skip_header = 1))
weights = np.array([0.3,0.2,0.5])
yields = climate_data @ weights
climate_results = np.concatenate((climate_data, yields.reshape(10000, 1)), axis =1)
