##Orchestrate the workflow

import data_loader
import model
import train
import predict
import config

from sklearn.decomposition import PCA

def main():
    # Load data
    df = data_loader.load_data(config.DATA_PATH)
    
    # Preprocess data
    df = data_loader.preprocess_data(df)
    
    # Select features
    data = data_loader.select_features(df)
    
    # Encode data
    data = data_loader.encode_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = data_loader.split_data(data)

    # Scaled data
    X_train_scaled, X_test_scaled =data_loader.scaled_data(X_train, X_test)
    
    # Apply SMOTE to training data
    X_train_res, y_train_res = data_loader.apply_smote(X_train_scaled, y_train)
    
    # Apply PCA
    pca = PCA(n_components=config.N_COMPONENTS)
    X_train_reduced = pca.fit_transform(X_train_res)
    X_test_reduced = pca.transform(X_test_scaled)
    
    # Train model
    trained_model = train.train_model(X_train_reduced, y_train_res, X_test_reduced, y_test, input_dim=X_train_reduced.shape[1])
    
    # Evaluate model
    predict.evaluate_model(trained_model, X_test_reduced, y_test)
    
if __name__ == "__main__":
    main()
