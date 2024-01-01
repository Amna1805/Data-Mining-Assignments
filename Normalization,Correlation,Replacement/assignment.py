import numpy as np

#Loading Dataset
records=np.loadtxt("adult.data",delimiter=",",dtype="str")
print("Adults Records:")
print(records)
print()
#Normalize one numeric attribute(Fnlwgt) using normalization formulas

fnlwgtdata=records[:,2]
fnlwgt=np.array(fnlwgtdata,dtype="int")

                    #1)min-max normalization
new_max=1
new_min=0
max_fnlwgt=np.max(fnlwgt)
min_fnlwgt=np.min(fnlwgt)
norm1_fnlwgt=np.zeros(len(fnlwgt))
for i in range(len(fnlwgt)):
        norm1_fnlwgt[i]=(((fnlwgt[i]-min_fnlwgt)/(max_fnlwgt-min_fnlwgt))*(new_max-new_min))+new_min
print("Normalize attribute using min-max Normalization is",norm1_fnlwgt)

                    #2)z-score normalization

norm2_fnlwgt=np.zeros(len(fnlwgt))
mean=np.mean(fnlwgt)
std=np.std(fnlwgt)
for i in range(len(fnlwgt)):
        norm2_fnlwgt[i]=(fnlwgt[i]-mean)/std
print("Normalize attribute using z-score Normalization is",norm2_fnlwgt)

                    #3)Decimal Scaling normalization

norm3_fnlwgt=np.zeros(len(fnlwgt))
str_arr=fnlwgt.astype(str)
j=max(len(s) for s in str_arr)
for i in range(len(fnlwgt)):
        norm3_fnlwgt[i]=(fnlwgt[i])/(np.power(10,j))
print("Normalize attribute using Decimal Scaling Normalization is",norm3_fnlwgt)
print()

#Correlation between two numeric attributes(capita_gain,capital_loss)
capital_gain=np.array(records[:,10],dtype="int")
capital_loss=np.array(records[:,11],dtype="int")
#By built in Funtion
corel=np.corrcoef(capital_gain,capital_loss)[0,1]
print("Correlation between two numeric attributes(capita_gain,capital_loss) using Built-in Function is:",corel)
#By using Formula
mean_gain=np.mean(capital_gain)
mean_loss=np.mean(capital_loss)
std_gain=np.std(capital_gain)
std_loss=np.std(capital_loss)
n=len(capital_gain)
sum=0
for i in range(len(capital_gain) and len(capital_loss)):
       sum+=(capital_gain[i]-mean_gain)*(capital_loss[i]-mean_loss)
corelate=(sum)/(n*std_gain*std_loss)
print("Correlation between two numeric attributes(capita_gain,capital_loss) using Formula is:",corelate)
print()
#Random sample of a given size with replacement of Array fnlwgt
fnlwgt_sample=np.random.choice(fnlwgt,size=500,replace=True)
print("Random sample of a given size with replacement of Array fnlwgt  with given size of 500 is:",fnlwgt_sample)
