#importing additional libraries for EDA and vizsualization
import matplotlib.pyplot as plt
import seaborn as sns

#importing more models for model selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#importing utilities for hyperparameter tuning and model evaluation
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#step 1: Extended EDA
#convert the dataset to a DataFrame for easier manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

#basic statistics of the features
feature_stats = df.describe()

#pairplot to visualize the relationships between features
sns.pairplot(df, hue='species')
plt.show()

#correlation matrix
corr_matrix = df.corr()

#step 2: feature Engineering
#model Selection with cross-validation for a variety of models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'k-Nearest Neighbors': KNeighborsClassifier()
}

#dictionary to store the best models and scores for each classifier
best_models = {}
cross_val_scores = {}

#scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#cross-validation and hyperparameter tuning for each model
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    cv_score = cross_val_score(pipeline, X, y, cv=5)
    cross_val_scores[name] = cv_score.mean()
    
    #hyperparameter tuning (a simple example, usually this would be more comprehensive)
    if name == 'k-Nearest Neighbors':
        param_grid = {'classifier__n_neighbors': [3, 5, 7, 9]}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(X_scaled, y)
        best_models[name] = grid_search.best_estimator_
    else:
        best_models[name] = model

#displays the cross-validation scores for each model
cross_val_scores
