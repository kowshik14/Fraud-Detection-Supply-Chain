## Model Training

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import model

def train_model(X_train, y_train, X_val, y_val, input_dim):
    # Building the model
    classifier = model.build_model(input_dim)
    
    # Callbacks
    checkpointer = ModelCheckpoint(filepath="model_supply.keras", verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
    csv_logger = CSVLogger('training_log.csv', separator=',', append=False)

    # Train the model
    classifier.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpointer, csv_logger])
    
    return classifier
