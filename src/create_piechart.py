import matplotlib.pyplot as plt

# Create a pie chart
def create_piechart(prediction, class_labels):
    fig, ax = plt.subplots()
    ax.pie(prediction, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=['C3', 'C2'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig