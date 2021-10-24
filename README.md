# Predicting the departure of employees within 2 years based on their HR data

![alt text](https://www.insperity.com/wp-content/uploads/2015/06/Everything-You-Need-to-Know-When-an-Employee-Takes-a-Leave-of-Absence-lg.png)


## Project background
Turnover can be a major issue in a company. 
It takes time to train an employee and the fact that they leave the company can become a major financial issue if it 
becomes too frequent.

**The objective of this mini-project is to go through all the steps of a Machine Learning project in order to answer a 
concrete business problem!**

Let's imagine that we are in a company with too many departures and that the management wants to develop a tool to 
identify employees likely to leave the company.

We have 2 years of historical HR data and have been able to identify which employees have and have not left the company.
An important business constraint is that HR does not have a lot of time available, so the employees identified must be 
really likely to leave.

These business constraints will have an impact on the modelling of the proposed solution.

The data I worked on came from Kaggle: https://www.kaggle.com/tejashvi14/employee-future-prediction


## Stages of work

The main steps of this project are:
- The handling and understanding of the data 
- A quick preprocessing of the data
- Machine Learning modelling
- Deployment of the model using an API.

![alt text](https://codetiburon.com/app/uploads/2019/03/Machine-learning-in-HR-industry-750x420.png)

## Project structure

**In order to view the graphs in the data_exploration.ipynb notebook** I recommend using: https://nbviewer.org/ .
Plotly cannot be directly visualized on Github!

The structure of this project is:
- **data:** contains the data (raw, train, test)
- **notebooks:** the data exploration, preprocessing and Machine Learning notebooks
- **pipeline:** the objects created in the notebooks for preprocessing and Machine Learning
- **src:** the scripts and functions useful for preprocessing and deployment
- **tests:** the tests related to the above scripts
- **environment.yml**: this creates the environment in which this project has been developed using the 
``conda env create --file environment.yml`` command at the root of the project
- **main.py:** API associated with the best Machine Learning model.


## Tools used:
The exploration and preprocessing steps are classical.

The Machine Learning part offers more interesting approaches with several models trained, optimised and compared.

The optimisation of hyperparameters is done with 2 methods:
- a RandomizeSearch then a GridSearch
- a **Bayesian Optimisation** of the hyperparameters.


Scikit-Learn             |  XGBoost
:-------------------------:|:-------------------------:
<img src="https://e7.pngegg.com/pngimages/905/45/png-clipart-scikit-learn-python-scikit-logo-brand-learning-text-computer-thumbnail.png" >  |  <img src="https://www.datacorner.fr/wp-content/uploads/2018/09/xgboost.png" > 

HyperOpt             |   FastAPI
:-------------------------:|:-------------------------:
<img src="https://miro.medium.com/max/2000/1*2gsysrNnSD-n8HDCHmpZFw.jpeg" > | <img src="https://d3uyj2gj5wa63n.cloudfront.net/wp-content/uploads/2021/02/fastapi-logo.png" >


References:
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- http://hyperopt.github.io/hyperopt/
- https://fastapi.tiangolo.com/.
