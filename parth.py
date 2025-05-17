import numpy as np

class SimpleLinearRegression:
    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)
        self.slope = np.sum((x - self.x_mean) * (y - self.y_mean)) / np.sum((x - self.x_mean) ** 2)
        self.intercept = self.y_mean - self.slope * self.x_mean

    def predict_y(self, x):
        return self.slope * x + self.intercept

    def predict_x(self, y):
        if self.slope == 0:
            raise ValueError("Slope is zero, cannot predict x from y.")
        return (y - self.intercept) / self.slope

# Example usage:
if __name__ == "__main__":
    x_vals = [1, 2, 3, 4, 5]
    y_vals = [2, 4, 5, 4, 5]
    model = SimpleLinearRegression()
    model.fit(x_vals, y_vals)
    print("Predict y for x=6:", model.predict_y(6))
    print("Predict x for y=7:", model.predict_x(7))