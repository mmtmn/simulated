import numpy as np
import cmath
import time
from scipy.constants import G, k, elementary_charge
import pygame

# Constants
NUM_PARTICLES = 100
TIME_STEP = 0.01
INPUT_NODES = 6  # Increased input nodes for more complex behaviors
HIDDEN_NODES = 20  # Increased hidden nodes for more complex behaviors
OUTPUT_NODES = 3
SCREEN_SIZE = 800
PARTICLE_RADIUS = 2
NUM_ITERATIONS = 1000
BOUNDARY = 10.0  # Boundary for the simulation space
LEARNING_RATE = 0.01
REWARD_DISTANCE = 1.0  # Distance within which a particle is considered to have achieved its goal
MUTATION_RATE = 0.01  # Mutation rate for genetic algorithm

# Particle class
class Particle:
    def __init__(self):
        self.position = np.random.uniform(-BOUNDARY, BOUNDARY, 3)
        self.momentum = np.random.uniform(-1.0, 1.0, 3)
        self.wavefunction = 1.0 + 0.0j
        self.nn = NeuralNetwork()
        self.goal = np.random.choice(['seek', 'avoid'])
        self.target = np.random.randint(0, NUM_PARTICLES)
        self.reward = 0

    def update(self, particles):
        target_particle = particles[self.target]
        inputs = np.concatenate((self.position, target_particle.position))
        self.nn.forward_pass(inputs)
        behavior = self.nn.output_layer

        if self.goal == 'seek':
            self.momentum += behavior * (target_particle.position - self.position) * TIME_STEP
        elif self.goal == 'avoid':
            self.momentum -= behavior * (target_particle.position - self.position) * TIME_STEP

        self.momentum = np.clip(self.momentum, -1.0, 1.0)
        self.update_reward(target_particle)
        self.nn.update_weights(LEARNING_RATE, self.reward)

    def update_reward(self, target_particle):
        distance = np.linalg.norm(self.position - target_particle.position)
        if self.goal == 'seek' and distance < REWARD_DISTANCE:
            self.reward = 1
        elif self.goal == 'avoid' and distance > REWARD_DISTANCE:
            self.reward = 1
        else:
            self.reward = -1

    def mutate(self):
        self.nn.mutate(MUTATION_RATE)

# NeuralNetwork class
class NeuralNetwork:
    def __init__(self):
        self.weights_input_hidden = np.random.uniform(-1.0, 1.0, (INPUT_NODES, HIDDEN_NODES))
        self.weights_hidden_output = np.random.uniform(-1.0, 1.0, (HIDDEN_NODES, OUTPUT_NODES))
        self.hidden_layer = np.zeros(HIDDEN_NODES)
        self.output_layer = np.zeros(OUTPUT_NODES)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward_pass(self, inputs):
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.weights_input_hidden))
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output))

    def update_weights(self, learning_rate, reward):
        error = reward - self.output_layer
        self.weights_hidden_output += learning_rate * np.outer(self.hidden_layer, error)
        self.weights_input_hidden += learning_rate * np.outer(self.hidden_layer, np.dot(error, self.weights_hidden_output.T))

    def mutate(self, mutation_rate):
        mutation_mask_input_hidden = np.random.rand(*self.weights_input_hidden.shape) < mutation_rate
        mutation_mask_hidden_output = np.random.rand(*self.weights_hidden_output.shape) < mutation_rate
        self.weights_input_hidden += mutation_mask_input_hidden * np.random.uniform(-0.1, 0.1, self.weights_input_hidden.shape)
        self.weights_hidden_output += mutation_mask_hidden_output * np.random.uniform(-0.1, 0.1, self.weights_hidden_output.shape)

# Environment class for handling different interactions
class Environment:
    def __init__(self, particles):
        self.particles = particles

    def update_particle(self, particle):
        potential = 0.5 * particle.position[0] ** 2
        particle.wavefunction *= cmath.exp(-1j * (potential + 0.5 * particle.momentum[0] ** 2) * TIME_STEP)
        particle.position += particle.momentum * TIME_STEP
        particle.momentum -= particle.position * TIME_STEP
        self.handle_boundaries(particle)

    def handle_boundaries(self, particle):
        for i in range(3):
            if particle.position[i] > BOUNDARY:
                particle.position[i] = BOUNDARY
                particle.momentum[i] *= -1
            elif particle.position[i] < -BOUNDARY:
                particle.position[i] = -BOUNDARY
                particle.momentum[i] *= -1

    def evolve_system(self):
        for particle in self.particles:
            self.update_particle(particle)

    def simulate_gravity(self):
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i != j:
                    dx = p2.position - p1.position
                    distance = np.linalg.norm(dx)
                    if distance > 0:
                        force = G / distance ** 2
                        p1.momentum += force * dx / distance * TIME_STEP

    def simulate_electromagnetic_forces(self):
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i != j:
                    dx = p2.position - p1.position
                    distance = np.linalg.norm(dx)
                    if distance > 0:
                        force = (elementary_charge ** 2) / (4 * np.pi * distance ** 2)
                        p1.momentum += force * dx / distance * TIME_STEP

    def simulate_collisions(self):
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i != j:
                    dx = p2.position - p1.position
                    distance = np.linalg.norm(dx)
                    if distance < 2 * PARTICLE_RADIUS:
                        p1.momentum, p2.momentum = p2.momentum, p1.momentum

    def simulate_thermodynamics(self):
        total_energy = sum(0.5 * np.linalg.norm(p.momentum) ** 2 for p in self.particles)
        temperature = total_energy / (len(self.particles) * k)
        print(f"Average Temperature: {temperature:.6f} K")

    def simulate_statistical_mechanics(self):
        avg_position = np.mean([p.position for p in self.particles], axis=0)
        avg_momentum = np.mean([p.momentum for p in self.particles], axis=0)
        print(f"Average Position: ({avg_position[0]:.6f}, {avg_position[1]:.6f}, {avg_position[2]:.6f})")
        print(f"Average Momentum: ({avg_momentum[0]:.6f}, {avg_momentum[1]:.6f}, {avg_momentum[2]:.6f})")

    def simulate_entanglement(self):
        for i in range(0, len(self.particles), 2):
            if i + 1 < len(self.particles):
                self.particles[i].wavefunction *= self.particles[i + 1].wavefunction
                self.particles[i + 1].wavefunction = self.particles[i].wavefunction

    def advanced_error_correction(self):
        for p in self.particles:
            if abs(p.wavefunction) > 1.0:
                p.wavefunction = 1.0 + 0.0j
                p.position = np.clip(p.position, -BOUNDARY, BOUNDARY)
                p.momentum = np.clip(p.momentum, -1.0, 1.0)

def simulate_consciousness_with_nn(particles):
    for i, particle in enumerate(particles):
        particle.update(particles)
        print(f"Consciousness Output for Particle {i}: ({particle.nn.output_layer[0]:.6f}, {particle.nn.output_layer[1]:.6f}, {particle.nn.output_layer[2]:.6f})")

def run_simulation(particles):
    env = Environment(particles)
    env.simulate_gravity()
    env.simulate_electromagnetic_forces()
    env.simulate_collisions()
    env.evolve_system()
    env.simulate_entanglement()
    env.advanced_error_correction()

def main():
    np.random.seed(int(time.time()))
    particles = [Particle() for _ in range(NUM_PARTICLES)]
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Particle Simulation")
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update particle positions in the simulation
        run_simulation(particles)

        # Clear the screen
        screen.fill((0, 0, 0))

        # Draw particles
        for particle in particles:
            x = int((particle.position[0] + BOUNDARY) / (2 * BOUNDARY) * SCREEN_SIZE)
            y = int((particle.position[1] + BOUNDARY) / (2 * BOUNDARY) * SCREEN_SIZE)
            color = (0, 255, 0) if particle.goal == 'seek' else (255, 0, 0)
            pygame.draw.circle(screen, color, (x, y), PARTICLE_RADIUS)

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()
    simulate_consciousness_with_nn(particles)
    env = Environment(particles)
    env.simulate_thermodynamics()
    env.simulate_statistical_mechanics()
    for i, particle in enumerate(particles):
        print(f"Particle {i}: Position ({particle.position[0]:.6f}, {particle.position[1]:.6f}, {particle.position[2]:.6f}), Wavefunction ({particle.wavefunction.real:.6f} + {particle.wavefunction.imag:.6f}i)")

if __name__ == "__main__":
    main()