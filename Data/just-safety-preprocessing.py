import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split

PROPORTION_TEST = 0.1
def clean(strs):
    # strs = re.sub(r'\\n', ' ', strs)
    # strs = re.sub(r'[^a-zA-Z0-9.,\':;%\- ]', '', strs)
    strs = strs.replace('\n', '')
    strs = strs.replace('"', '')
    return strs

filenames = ['alignment', 'monitoring', 'robustness', 'systemic', 'nonSafety']
train = pd.DataFrame(columns = ['text', 'label'])
test = pd.DataFrame(columns = ['text', 'label'])
for i in range(len(filenames) - 1):
    df = pd.read_csv('Combined-and-Crawled/'+filenames[i]+'.csv')
    cleanedDF = pd.DataFrame(columns = ['text', 'label'])
    for j in range(df.shape[0]):
        title = clean(df.iloc[j, 0])
        abs = clean(df.iloc[j,1])
        cleanedDF = cleanedDF.append({'text': title + ' ' + abs, 'label': 0}, ignore_index=True)
    trainSub, testSub = train_test_split(cleanedDF, test_size=PROPORTION_TEST)
    train = train.append(trainSub)
    test = test.append(testSub)

df = pd.read_csv('Combined-and-Crawled/nonSafety' + '.csv')
cleanedDF = pd.DataFrame(columns = ['text', 'label'])
for j in range(df.shape[0]):
    title = clean(df.iloc[j, 0])
    abs = clean(df.iloc[j,1])
    cleanedDF = cleanedDF.append({'text': title + ' ' + abs, 'label': 1}, ignore_index=True)
trainSub, testSub = train_test_split(cleanedDF, test_size=PROPORTION_TEST)
train = train.append(trainSub)
test = test.append(testSub)

train.to_csv('Just-Safety/train.csv', index=False)
test.to_csv('Just-Safety/test.csv', index=False)