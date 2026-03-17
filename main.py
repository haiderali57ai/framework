from abc import ABC, abstractmethod

class ModelConfig:
    """Stores model settings - demonstrating Composition and Magic Methods."""
    def __init__(self, model_name, learning_rate=0.01, epochs=10):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __repr__(self):
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"

class BaseModel(ABC):
    """Abstract Base Class - demonstrating Abstraction and Class Attributes."""
    model_count = 0

    def __init__(self, config: ModelConfig):
        self.config = config
        BaseModel.model_count += 1

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass

class LinearRegressionModel(BaseModel):
    """Concrete implementation - demonstrating Inheritance and super()."""
    def __init__(self, config):
        super().__init__(config)

    def train(self, data):
        print(f"{self.config.model_name}: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    def evaluate(self, data):
        print(f"{self.config.model_name}: Evaluation MSE = 0.042")

class NeuralNetworkModel(BaseModel):
    """Concrete implementation - demonstrating Method Overriding."""
    def __init__(self, config, layers):
        super().__init__(config)
        self.layers = layers

    def train(self, data):
        print(f"{self.config.model_name} {self.layers}: Training on {len(data)} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    def evaluate(self, data):
        print(f"{self.config.model_name}: Evaluation Accuracy = 91.5%")

class DataLoader:
    """Independent class for Aggregation."""
    def __init__(self, dataset):
        self.dataset = dataset

class Trainer:
    """Orchestrator class - demonstrating Polymorphism."""
    def __init__(self, model: BaseModel, data_loader: DataLoader):
        self.model = model
        self.data_loader = data_loader

    def run(self):
        print(f"--- Training {self.model.config.model_name} ---")
        # Polymorphism: calls the correct train/evaluate based on object type
        self.model.train(self.data_loader.dataset)
        self.model.evaluate(self.data_loader.dataset)

# --- Execution ---
if __name__ == "__main__":
    # 1. Setup Configs
    lr_config = ModelConfig("LinearRegression", 0.01, 10)
    nn_config = ModelConfig("NeuralNetwork", 0.001, 20)
    
    print(lr_config)
    print(nn_config)

    # 2. Instantiate Models
    model1 = LinearRegressionModel(lr_config)
    model2 = NeuralNetworkModel(nn_config, [64, 32, 1])
    print(f"Models created: {BaseModel.model_count}")

    # 3. Load Data
    data = DataLoader([1, 2, 3, 4, 5])

    # 4. Train and Evaluate
    Trainer(model1, data).run()
    Trainer(model2, data).run()
