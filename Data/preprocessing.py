import pandas as pd
import re


def clean(strs):
    strs = re.sub(r'<.*?>', '', strs)
    strs = re.sub(r'[^a-zA-Z. ]', '', strs)
    return strs


for filename in ['alignment', 'monitoring', 'nonSafety', 'robustness', 'systemic']:
    df = pd.read_csv('Combined-and-Crawled/'+filename+'.csv')
    cleanedDF = pd.DataFrame(columns = ['text'])
    for j in range(df.shape[0]):
        title = clean(df.iloc[j, 0])
        abs = clean(df.iloc[j,1])
        cleanedDF = cleanedDF.append({'text': title + ' ' + abs}, ignore_index=True)
    cleanedDF.to_csv('Processed/' + filename + '.csv')

# allSafety = pd.DataFrame(columns=['text'])
# for filename in ['alignment', 'misc', 'monitoring', 'robustness', 'systemic']:
#     df = pd.read_csv('Crawled/'+filename+'.csv')
#     allSafety = allSafety.append(df)

# allSafety.to_csv("Processed/allSafety.csv")
