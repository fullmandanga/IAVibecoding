import json

class MemoryModule:
    def __init__(self, memory_file="agi_memory.json"):
        self.memory_file = memory_file
        self.memory = self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as f:
                memory = json.load(f)
        except FileNotFoundError:
            memory = {}
        return memory

    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=4)

    def store_knowledge(self, key, value):
        self.memory[key] = value
        self.save_memory()

    def retrieve_knowledge(self, key):
        return self.memory.get(key)

    def share_knowledge(self, other_memory_module, key):
        value = self.retrieve_knowledge(key)
        if value:
            other_memory_module.store_knowledge(key, value)
            print(f"Knowledge '{key}' shared with {other_memory_module.memory_file}")
        else:
            print(f"Knowledge '{key}' not found in memory.")

if __name__ == '__main__':
    # Example usage
    memory1 = MemoryModule("memory1.json")
    memory2 = MemoryModule("memory2.json")

    memory1.store_knowledge("problem_solving_strategy", "Decompose the problem into smaller subproblems.")
    memory1.store_knowledge("bayesian_optimization_params", {"lr": 0.001, "optimizer": "adam"})

    print(f"Memory 1: {memory1.memory}")
    print(f"Memory 2: {memory2.memory}")

    memory1.share_knowledge(memory2, "problem_solving_strategy")

    print(f"Memory 1: {memory1.memory}")
    print(f"Memory 2: {memory2.memory}")
