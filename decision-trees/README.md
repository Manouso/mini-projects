# Decision Trees - Titanic Dataset

This project demonstrates how to use preprocessed data from the project data-preprocessing to predict target values by implementing decision trees
and compares this model with logistic regression to find which one performs the best. 

---

## Project Structure

- **decision-trees/**
  - `decision_trees.ipynb` — Main notebook for experiments  
  - `requirements.txt` — Requirements  
  - `README.md` — Project overview

---

## Objectives
 
- Use decision trees to classify passenger survival outcomes.
- Understand the concept of splitting criteria (Gini)
- Compare accuracy with logistic regression

--- 

## Dataset

- Columns: 
   - `Survived`: Index of survival (1 = survived, 0 = didn't survived)     
   - `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)            
   - `Sex`: Male/Female         
   - `Age`: Age in years         
   - `FamilySize`: How many family members were aboard            
   - `Fare`: Passenger Fare       
   - `Cabin`: Number of Cabin       
   - `Embarked`: Port of embarkation (C, Q, S) 
   - `IsAlone`: Binary indicator for traveling alone (1 = alone, 0 = not alone)

---

## Requirements

```bash
pip install -r requirements.txt
```
---

## Insights 