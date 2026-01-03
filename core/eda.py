
import matplotlib.pyplot as plt
def plot_eda(df):
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['close'])
    ax.set_title("Price Evolution")
    return fig
