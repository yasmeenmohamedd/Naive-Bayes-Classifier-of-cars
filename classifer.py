from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

Color=['Red','Red','Red','Yellow','Yellow','Yellow','Yellow','Yellow','Red','Red']

Type=['Sports','Sports','Sports','Sports','Sports','SUV','SUV','SUV','SUV','Sports']

Origin=['Domestic','Domestic','Domestic','Domestic','Imported','Imported','Imported','Domestic','Imported','Imported']

Stolen=['Yes','No','Yes','No','Yes','No','Yes','No','No','Yes']
le = preprocessing.LabelEncoder()
# Converting string labels into numbers
Color_encoded=le.fit_transform(Color)
type_encoded=le.fit_transform(Type)
Origin_encoded=le.fit_transform(Origin)
Stolen_encoded=le.fit_transform(Stolen)
print("Color:",Color_encoded)
print("Type:",type_encoded)
print("Origin:",Origin_encoded)
print("Stolen:",Stolen_encoded)
combineFeatures = list(zip(Color_encoded,type_encoded,Origin_encoded))
print("CombineFeatures:",combineFeatures)
model = GaussianNB()

model.fit(combineFeatures,Stolen_encoded)

predicted = model.predict([[0,0,0]])
print("Predicted Value:", predicted)