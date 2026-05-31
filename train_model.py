from fraud_engine import MODEL_PATH, train_and_save_model

if __name__ == "__main__":
    path = train_and_save_model(MODEL_PATH)
    print(f"Model trained and saved at: {path.resolve()}")

