import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import KFold
import mlflow

class LinearRegression:
    def __init__(self,
                 lr=0.001,
                 num_epochs=500,
                 batch_size=50,
                 method='batch',
                 init_method='zeros',
                 use_momentum=False,
                 momentum=0.9,
                 polynomial=False,
                 degree=2,
                 regularization=None,
                 kfold_splits=3,
                 random_state=42):
        
        # Training params
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method
        self.init_method = init_method.lower()
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.velocity = None
        self.regularization = regularization
        
        # Polynomial
        self.polynomial = polynomial
        self.degree = degree
        self.poly_transformer = None
        
        # Scaling and encoding
        self.scaler = None
        self.label_encoders = {}
        self.categorical_cols = []
        self.feature_cols = None
        
        # KFold
        self.kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=random_state)
        self.kfold_scores = []

    # ---------------- Metrics ----------------
    def mse(self, ytrue, ypred):
        return np.mean((ypred - ytrue)**2)

    def r2(self, ytrue, ypred):
        ss_res = np.sum((ytrue - ypred)**2)
        ss_tot = np.sum((ytrue - np.mean(ytrue))**2)
        return 1 - (ss_res / ss_tot)

    # ---------------- Weight Initialization ----------------
    def _initialize_weights(self, n_features):
        if self.init_method == 'xavier':
            limit = np.sqrt(1 / n_features)
            return np.random.uniform(-limit, limit, size=n_features)
        else:
            return np.zeros(n_features)

    # ---------------- Preprocessing ----------------
    def _preprocess_X(self, X, fit=True):
        X = X.copy()
        # Identify numeric and categorical columns
        if self.feature_cols is None:
            self.feature_cols = X.columns.tolist()
        if not self.categorical_cols:
            self.categorical_cols = X.select_dtypes(include='object').columns.tolist()

        # Encode categorical columns
        for col in self.categorical_cols:
            if fit:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    raise ValueError(f"LabelEncoder for {col} not found")
                # Handle unseen labels
                X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

        # Polynomial features
        X_poly = X.values
        if self.polynomial:
            if fit:
                poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                X_poly = poly.fit_transform(X_poly)
                self.poly_transformer = poly
            else:
                if self.poly_transformer is None:
                    raise ValueError("Polynomial transformer not fitted")
                X_poly = self.poly_transformer.transform(X_poly)

        # Scaling
        if fit:
            self.scaler = MinMaxScaler()
            X_poly = self.scaler.fit_transform(X_poly)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            X_poly = self.scaler.transform(X_poly)

        # Add bias
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]
        return X_poly

    # ---------------- Train Step ----------------
    def _train_step(self, X, y):
        yhat = X @ self.theta
        grad = (X.T @ (yhat - y)) / X.shape[0]
        if self.regularization is not None:
            grad += self.regularization.derivation(self.theta)
        if self.use_momentum:
            self.velocity = self.momentum * self.velocity + self.lr * grad
            self.theta -= self.velocity
        else:
            self.theta -= self.lr * grad
        return self.mse(y, yhat)

    # ---------------- Fit ----------------
    def fit(self, X_train, y_train):
        self.kfold_scores = []
        self.val_loss_old = np.inf

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Preprocess
            X_fold_train_proc = self._preprocess_X(X_fold_train, fit=True)
            X_fold_val_proc = self._preprocess_X(X_fold_val, fit=False)
            
            self.theta = self._initialize_weights(X_fold_train_proc.shape[1])
            if self.use_momentum:
                self.velocity = np.zeros_like(self.theta)

            # Log fold
            mlflow.log_params({
                f"fold{fold}_lr": self.lr,
                f"fold{fold}_init_method": self.init_method,
                f"fold{fold}_use_momentum": self.use_momentum,
                f"fold{fold}_momentum": self.momentum,
                f"fold{fold}_polynomial": self.polynomial,
                f"fold{fold}_degree": self.degree
            })

            for epoch in range(self.num_epochs):
                perm = np.random.permutation(X_fold_train_proc.shape[0])
                X_shuffled = X_fold_train_proc[perm]
                y_shuffled = y_fold_train.to_numpy()[perm] if hasattr(y_fold_train, "to_numpy") else y_fold_train[perm]

                # Train step
                if self.method == 'sto':
                    for i in range(X_shuffled.shape[0]):
                        train_loss = self._train_step(X_shuffled[i].reshape(1, -1), y_shuffled[i])
                elif self.method == 'mini':
                    for i in range(0, X_shuffled.shape[0], self.batch_size):
                        X_batch = X_shuffled[i:i+self.batch_size]
                        y_batch = y_shuffled[i:i+self.batch_size]
                        train_loss = self._train_step(X_batch, y_batch)
                else:
                    train_loss = self._train_step(X_shuffled, y_shuffled)

                val_loss = self.mse(X_fold_val_proc @ self.theta, y_fold_val.to_numpy())
                if np.allclose(val_loss, self.val_loss_old):
                    break
                self.val_loss_old = val_loss

            self.kfold_scores.append(val_loss)
            print(f"Fold {fold}: {val_loss}")

    # ---------------- Predict ----------------
    def predict(self, X):
        X_proc = self._preprocess_X(X, fit=False)
        return X_proc @ self.theta

    # ---------------- Coefficients ----------------
    def coef_(self):
        return self.theta[1:]

    def intercept_(self):
        return self.theta[0]
