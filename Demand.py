import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from Population import pop
from ResidentialSubscriber import trs
from NonResidentialSubscriber import tnrs
from HouseholdPrice import hspr
from NonHouseholdPrice import nhspr

print("Estimates for 2018 are:","\n","Population:",pop,"\n","Total Residential Subscriber:",trs,"\n","Total Non-Residential Subscriber:",tnrs,"\n","Average Household Price:",hspr,"\n","Average Non-Household Price:",nhspr)

df = pd.read_csv("dataset.txt", sep=';')

estimations = [pop, trs, tnrs, hspr, nhspr]

for i in estimations:
    df.append(i,ignore_index=True)
    
print(estimations)