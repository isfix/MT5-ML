#--- Genetic Algorithm Hyperparameter Optimization ---
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score
import xgboost as xgb
import lightgbm as lgb
import json
import random
from deap import base, creator, tools, algorithms

# Load the single labeled dataset
df = pd.read_csv('data/final_dataset.csv', index_col='time', parse_dates=True)

with open('models/model_info.json', 'r') as f:
    model_info = json.load(f)

feature_cols = model_info['feature_names']
X = df[feature_cols]
y = df['label']

# For GA, we will use cross-validation on the entire dataset.
# We'll split it inside the evaluation function.
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print(f"Optimizing {model_info['model_type']} on {len(X_train)} training samples and validating on {len(X_val)} samples.")

# Define parameter search spaces
if model_info['model_type'] == 'XGBoost':
    param_bounds = {
        'n_estimators': (50, 300),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }
else:  # LightGBM
    param_bounds = {
        'n_estimators': (50, 300),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }

# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Gene generators for each parameter
toolbox.register("n_estimators", random.randint, 50, 300)
toolbox.register("max_depth", random.randint, 3, 10)
toolbox.register("learning_rate", random.uniform, 0.01, 0.3)
toolbox.register("subsample", random.uniform, 0.6, 1.0)
toolbox.register("colsample_bytree", random.uniform, 0.6, 1.0)

# Individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.learning_rate,
                  toolbox.subsample, toolbox.colsample_bytree), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_individual(individual):
    """Evaluate fitness of parameter combination"""
    params = {
        'n_estimators': int(individual[0]),
        'max_depth': int(individual[1]),
        'learning_rate': individual[2],
        'subsample': individual[3],
        'colsample_bytree': individual[4],
        'random_state': 42
    }
    
    if model_info['model_type'] == 'XGBoost':
        params['eval_metric'] = 'logloss'
        model = xgb.XGBClassifier(**params)
    else:
        params['verbose'] = -1
        model = lgb.LGBMClassifier(**params)
    
    # Train on training set, evaluate on validation set
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        precision = precision_score(y_val, y_pred, zero_division=0)
        return (precision,)
    except:
        return (0.0,)  # Return poor score if model fails

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run Genetic Algorithm
print("Starting Genetic Algorithm optimization...")
random.seed(42)
np.random.seed(42)

population = toolbox.population(n=20)  # Small population for speed
NGEN = 10  # Limited generations for demonstration

# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# Evolution
population, logbook = algorithms.eaSimple(
    population, toolbox, cxpb=0.7, mutpb=0.3, ngen=NGEN, 
    stats=stats, verbose=True
)

# Get best individual
best_individual = tools.selBest(population, 1)[0]
best_params = {
    'n_estimators': int(best_individual[0]),
    'max_depth': int(best_individual[1]),
    'learning_rate': best_individual[2],
    'subsample': best_individual[3],
    'colsample_bytree': best_individual[4],
    'random_state': 42
}

if model_info['model_type'] == 'XGBoost':
    best_params['eval_metric'] = 'logloss'
else:
    best_params['verbose'] = -1

print(f"\n=== Optimization Complete ===")
print(f"Best fitness (precision): {best_individual.fitness.values[0]:.4f}")
print(f"Best parameters: {best_params}")

# Save optimized parameters
with open('models/optimized_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print("Optimized parameters saved to optimized_params.json")
