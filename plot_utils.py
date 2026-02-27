# libarary for plotting graphs and visualization data
import matplotlib.pyplot as plt

# function to plot comparision between real and predicted prices
def plot_predictions(y_true, y_pred):
    
    # Plot real vs predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(y_true, color='black', label='Real Price')
    plt.plot(y_pred, color='green', label='Predicted Price')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()