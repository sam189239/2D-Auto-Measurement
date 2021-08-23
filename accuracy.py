from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import json

with open('out\\output.json','r') as out:
    y_pred = json.loads(out.read())
with open('in\\actual_measure.json','r') as out:
    y_true = json.loads(out.read())

# y_pred = json.loads(data)
del y_pred['Height']
del y_pred['feedback']
# y_true =  {'Waist': 95, 'Neck': 38, 'Cuff': 18, 'Shoulder': 50, 'Arm': 60}
name = input("Name of sample: ")

mse = mean_squared_error(list(y_true.values()), list(y_pred.values()), sample_weight=None, multioutput='uniform_average', squared=True)
mape = mean_absolute_percentage_error(list(y_true.values()), list(y_pred.values()), sample_weight=None, multioutput='uniform_average')
print("MSE: " + str(mse))
print("ACC (using MAPE): " + str((1 - mape)*100) + " %")

acc = [name, mse, (1 - mape)*100]
from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

append_list_as_row("out\\acc_out.csv",acc)