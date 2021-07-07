from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import json

with open('out\\output.json','r') as out:
    data = out.read()

y_pred = json.loads(data)
del y_pred['Height']
y_true =  {'Waist': 95, 'Neck': 38, 'Cuff': 18, 'Shoulder': 50, 'Arm': 60}
# for a in obj.values():
#     print(a)
# print(type(obj))

mse = mean_squared_error(list(y_true.values()), list(y_pred.values()), sample_weight=None, multioutput='uniform_average', squared=True)
mape = mean_absolute_percentage_error(list(y_true.values()), list(y_pred.values()), sample_weight=None, multioutput='uniform_average')
print("MSE: " + str(mse))
print("ACC (using MAPE): " + str((1 - mape)*100) + " %")