import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pandas as pd 
import matplotlib.pyplot as plt
#%matplotlib inline



dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)




#mappings = som.win_map(X)



l = []
dist_map = som.distance_map()
mappings = som.win_map(X)

for x in range(int(dist_map.shape[0])):
    for y in range(int(dist_map.shape[1])):
        if dist_map[x,y] > .9:
            if len(mappings[(x,y)]) > 0:
                l.append((x, y))


frauds = mappings[l[0]]

for i in range(1,len(l)):
    frauds = np.concatenate((frauds, mappings[l[i]]))
                
                                

frauds = sc.inverse_transform(frauds)
print(frauds[:,0])



customers = dataset.iloc[:,1:].values

is_fraud = np.zeros(len(dataset))


for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)



#ANN
from keras.models import Sequential  #initialize
from keras.layers import Dense       #layers

classifier = Sequential()
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim = 15))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(customers, is_fraud, batch_size=1, epochs=8)
        
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred) , axis=1)
y_pred = y_pred[y_pred[:,1].argsort()]

#print(y_pred)




#####  Comparing Diff Models ####




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(customers, is_fraud, test_size=0.3, random_state=101)

print(customers[:2])
print('\n')


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print('---------RANDOM_FOREST_CLASSIFIER------------')


print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')



from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)



print('------------DTREE_CLASSIFIER---------------')


print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)



print('---------LOGISTIC_REGRESSION_MODEL------------')


print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')



print(X_test)  


