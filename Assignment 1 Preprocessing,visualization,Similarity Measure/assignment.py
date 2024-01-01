import numpy as np
import matplotlib.pyplot as plt
import math
def one_hot_encoding(array):
    unique_vals = [np.unique(array[:, i]) for i in range(array.shape[1])]
    one_hot_cols = []
    for i in range(array.shape[1]):
        one_hot_cols.append(np.eye(unique_vals[i].size)[np.searchsorted(unique_vals[i], array[:, i])])
    arr_one_hot = np.column_stack(one_hot_cols)
    return arr_one_hot
    
#Loading Dataset
records=np.loadtxt("adult.data",delimiter=",",dtype="str")

#Converting one attrribute fnlwgt(Final weight) to an array to perform basic operations such as mean,median etc.
fnlwgtdata=records[:,2]
fnlwgt=np.array(fnlwgtdata,dtype="int")

#Finding Average,Mean,Median and Standard Deviation of this numerical attribute
print("Average of Fnlwgt(Final Weight) is:",np.round(np.average(fnlwgt),3))
print("Mean of Fnlwgt(Final Weight) is:",np.round(np.mean(fnlwgt),3))
print("Median of Fnlwgt(Final Weight) is:",np.round(np.median(fnlwgt),3))
print("Standard Deviation of Fnlwgt(Final Weight) is:",np.round(np.std(fnlwgt),3))

#Visualizing attributes on ScatterPlot
occupation=np.array(records[:,7],dtype="str")
age=np.array(records[:,0],dtype=int)

plt.title("Occupation Vs Age")
plt.scatter(occupation,age,color="yellow")
plt.xlabel("OCCUPATION")
plt.ylabel("AGE")
plt.show()

#Finding Similarity between two categorical attributes
relationship=np.array(records[:,7],dtype="str")
gender=np.array(records[:,9],dtype="str")
categ_array= np.column_stack((relationship, gender))

arr=one_hot_encoding(categ_array)
coefficient=[]
for i in range(len(arr)-1):

        q = np.logical_and(arr[i], arr[i+1]).sum()
        r = np.logical_and(np.logical_not(arr[i]), arr[i+1]).sum()
        s = np.logical_and(arr[i], np.logical_not(arr[i+1])).sum()
        coeff= q / (q + r + s)
        coefficient.append(coeff)
        
print("Average Similarity between two categorical attrubutes(Relationship and gender)using jaccard coeeficient is:",np.round(np.average(coefficient),3))

#Finding Similarity between two numerical attributes

capital_gain=np.array(records[:,10],dtype="int")
capital_loss=np.array(records[:,11],dtype="int")
numeric_array= np.column_stack((capital_gain, capital_loss))
euc_dis=[]
i=0
for i in range(len(numeric_array)-1):
     j=i+1
     dis=math.sqrt(abs((numeric_array[i,0]-numeric_array[j,0])**2)+abs((numeric_array[i,1]-numeric_array[j,1])**2))
     euc_dis.append(dis)

print("Average Similarity between two numerical  attrubutes(capital_gain and capital_loss)using Euclidean distance is:",np.round(np.average(euc_dis),3))
