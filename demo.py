# Batch Prediction
# Training Pipeline

from Insurance.pipeline.batch_prediction import start_batch_prediction

file_path = "insurance.csv"

if __name__ == "__main__":
    try:
        output = start_batch_prediction(input_file_path= file_path)
    except Exception as e:
        print(e)