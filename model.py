import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_model():

    df = pd.read_csv("Artworks.csv")

    # select only needed columns
    df = df[['Department','Height (cm)','Width (cm)','Classification']]

    # remove missing values
    df = df.dropna()

    # reduce dataset size for speed
    df = df.sample(8000, random_state=42)

    # convert department to numbers
    df = pd.get_dummies(df, columns=['Department'])

    X = df.drop("Classification", axis=1)
    y = df["Classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy, X.columns