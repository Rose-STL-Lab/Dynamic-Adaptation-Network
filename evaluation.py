import torch
import numpy as np

mag_factors = np.array([0.13095428, 0.24349220, 0.2945909, 0.35664663, 0.4189510,
                       0.42764676, 0.59501153, 0.6283648, 0.63587710, 0.6628148,
                       0.62152666, 0.71263870, 0.8592970, 0.84638290, 0.8700263,
                       0.84896370, 0.94084600, 0.9711123, 0.96279530, 0.9361194,
                       0.89304050, 1.11027920, 1.1488024, 1.08242170, 1.2189536])
test_factors = np.array([3, 8, 13, 18, 23]).astype(int)
train_factors = np.array(list(set(list(range(1,26))) - set(test_factors))). astype(int)


E1, E2 = [], []
for i in range(1, 4):
    results = torch.load("DyAd_" + str(i) + ".pt")
    rmse, preds, trues = results["future"]
    rmse2, preds2, trues2 = results["domain"]
    rmse_future = np.mean(np.sqrt(np.mean((preds - trues).reshape(20,-1,50,20,64,64)**2, axis = (1,2,3,4,5)))*(1/mag_factors[train_factors-1]))
    rmse_domain = np.mean(np.sqrt(np.mean((preds2 - trues2).reshape(5,-1,50,20,64,64)**2, axis = (1,2,3,4,5)))*(1/mag_factors[test_factors-1]))
    E1.append(rmse_future)
    E2.append(rmse_domain)
    
print("Future:", np.round([np.mean(E1), np.std(E1)], 4))
print("Domain:", np.round([np.mean(E2), np.std(E2)], 4))
