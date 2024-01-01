import pandas as pd
# Load dataset as Pandas DataFrame
df = pd.read_csv("data.csv",encoding='ISO-8859-1')
print("TRANSACTION RECORDS")
print(df.head())
invoice= df.groupby('InvoiceNo')['Description'].apply(list).values.tolist()
# Convert all items in each invoice to strings
transactions = [[str(item) for item in transaction] for transaction in invoice]
from apyori import apriori

# Define the minimum support, confidence, and lift values
min_support = 0.01
min_confidence = 0.5
min_lift = 3


# Find frequent itemsets
frequent_itemsets = list(apriori(transactions, min_support=min_support))
print("Frequent Itemsets are:")
print("----------------------------------------------------------------------")
for item in frequent_itemsets:
    set=item[0]
    items=[x for x in set]
    print("Frequent ItemSet:"+items[0])
    print("Support:"+str(item[1]))
    print("Confidence:"+str(item[2][0][2]))
    print("Lift:"+str(item[2][0][3]))
    print("================================================")

# Find strong association rules
rules = list(apriori(transactions, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift))
print("Strong Association Rules for Our transaction records are:")
print("----------------------------------------------------------------------")
for item in rules:
    set=item[0]
    items=[x for x in set]
    print("Asociation Rule:"+items[0]+"->"+items[1])
    print("Support:"+str(item[1]))
    print("Confidence:"+str(item[2][0][2]))
    print("Lift:"+str(item[2][0][3]))
    print("================================================")
