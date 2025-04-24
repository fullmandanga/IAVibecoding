import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, num_layers, num_neurons):
        super().__init__()
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(10, self.num_neurons[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(self.num_neurons[i-1], self.num_neurons[i]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(num_neurons[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Define the fitness function
def fitness(nn):
    try:
        # Dummy training data
        inputs = torch.randn(100, 10)
        targets = torch.randn(100, 1)

        # Define the loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(nn.parameters(), lr=0.01)

        # Train the neural network
        for i in range(10):
            # Zero the gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = nn(inputs)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Return the loss as the fitness
        return -loss.item()
    except Exception as e:
        print(f"Error in fitness function: {e}")
        return -1000 # Return a very low fitness value

# Define the selection function
def selection(population, num_parents):
    # Sort the population by fitness
    population = sorted(population, key=lambda nn: fitness(nn), reverse=True)

    # Return the top num_parents neural networks
    return population[:num_parents]

# Define the crossover function
def crossover(parents, num_children):
    children = []
    for i in range(num_children):
        # Select two random parents
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        # The child will have the same architecture as parent1
        num_layers = parent1.num_layers
        num_neurons = parent1.num_neurons
        child = SimpleNN(num_layers=num_layers, num_neurons=num_neurons)

        # Copy the weights from the parents
        for name, param in child.named_parameters():
            if name in parent1.state_dict() and name in parent2.state_dict():
                if random.random() < 0.5:
                    param.data = parent1.state_dict()[name].data
                else:
                    param.data = parent2.state_dict()[name].data

        # Add the child to the list of children
        children.append(child)

    # Return the list of children
    return children

# Define the mutation function
def mutation(population, mutation_rate, best_networks):
    for nn in population:
        for name, param in nn.named_parameters():
            if random.random() < mutation_rate:
                # Use the best networks to guide the mutation
                if best_networks:
                    best_network = random.choice(best_networks)
                    param.data += (best_network.state_dict()[name].data - param.data) * 0.1
                else:
                    param.data += torch.randn(param.data.shape) * 0.1

# Define the evolutionary algorithm
def evolve(population, num_parents, num_children, mutation_rate, best_networks, learning_rate):
    # Select the parents
    parents = selection(population, num_parents)

    # Crossover the parents to create the children

    # Crossover the parents to create the children
    children = crossover(parents, num_children)

    # Mutate the children
    mutation(children, mutation_rate, best_networks)

    # Return the new population
    return parents + children

# Create the initial population
population_size = 10
population = []
# Create the initial architecture
num_layers = random.randint(1, 3)
num_neurons = [random.randint(10, 100) for _ in range(num_layers)]
if not num_neurons:
    num_neurons = [random.randint(10, 100)]
for i in range(population_size):
    population.append(SimpleNN(num_layers=num_layers, num_neurons=num_neurons))

# Evolve the population
num_generations = 10
num_parents = 5
num_children = 5
mutation_rate = 0.1
best_networks = []

for i in range(num_generations):
    # Evolve the population
    population = evolve(population, num_parents, num_children, mutation_rate, best_networks)

    # Get the best network
    best_network = sorted(population, key=lambda nn: fitness(nn), reverse=True)[0]
    best_networks.append(best_network)

    # Print the fitness of the best neural network
    print(f"Generation {i}: Best fitness = {fitness(best_network)}")
