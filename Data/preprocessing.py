import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split

PROPORTION_TEST = 0.1
def clean(strs):
    strs = re.sub(r'<.*?>', '', strs)
    strs = re.sub(r'[^a-zA-Z. ]', '', strs)
    return strs


# for filename in ['alignment', 'monitoring', 'nonSafety', 'robustness', 'systemic']:
#     df = pd.read_csv('Combined-and-Crawled/'+filename+'.csv')
#     cleanedDF = pd.DataFrame(columns = ['text'])
#     for j in range(df.shape[0]):
#         title = clean(df.iloc[j, 0])
#         abs = clean(df.iloc[j,1])
#         cleanedDF = cleanedDF.append({'text': title + ' ' + abs}, ignore_index=True)
#     cleanedDF.to_csv('Processed/' + filename + '.csv')

# all = pd.DataFrame(columns=['text', 'class'])
# filenames = ['alignment', 'monitoring', 'robustness', 'systemic', 'nonSafety']
# for i in range(len(filenames)):
#     df = pd.read_csv('Combined-and-Crawled/'+filenames[i]+'.csv')
#     df['class'] = i
#     all = all.append(df)
# all.to_csv("Processed/all.csv")

filenames = ['alignment', 'monitoring', 'robustness', 'systemic', 'nonSafety']
train = pd.DataFrame(columns = ['text', 'label'])
test = pd.DataFrame(columns = ['text', 'label'])
for i in range(len(filenames)):
    df = pd.read_csv('Combined-and-Crawled/'+filenames[i]+'.csv')
    cleanedDF = pd.DataFrame(columns = ['text', 'label'])
    for j in range(df.shape[0]):
        title = clean(df.iloc[j, 0])
        abs = clean(df.iloc[j,1])
        cleanedDF = cleanedDF.append({'text': title + ' ' + abs, 'label': i}, ignore_index=True)
    trainSub, testSub = train_test_split(cleanedDF, test_size=PROPORTION_TEST)
    train = train.append(trainSub)
    test = test.append(testSub)

train.to_csv('Processed/train.csv', index=False)
test.to_csv('Processed/test.csv', index=False)