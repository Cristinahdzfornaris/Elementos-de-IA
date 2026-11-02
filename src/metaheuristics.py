"""
Módulo D: Metaheurísticas
Optimización combinatoria para determinar el orden óptimo de visita de POIs.
Problema tipo TSP/Orienteering con restricciones de recursos.
"""

import random
import math
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    from .environment import Environment, POI
    from .search import dijkstra
except ImportError:
    from environment import Environment, POI
    from search import dijkstra


@dataclass
class Solution:
    """Solución: orden de visita de POIs."""
    poi_order: List[int]  
    fitness: float
    total_cost: float
    total_value: float
    feasible: bool


class MetaheuristicOptimizer:
    """
    Optimizador usando metaheurísticas para el problema de orden de visita de POIs.
    """
    
    def __init__(self, env: Environment, max_evaluations: int = 10000):
        self.env = env
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        
        # Precomputar distancias entre todos los pares de puntos relevantes
        self.distance_cache = self._precompute_distances()
    
    def _precompute_distances(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """
        Precomputa las distancias (costos de camino) entre todos los puntos relevantes:
        base, antenas, y POIs.
        """
        cache = {}
        points = [(self.env.base.x, self.env.base.y)]
        
        for antenna in self.env.antennas:
            points.append((antenna.x, antenna.y))
        
        for poi in self.env.pois:
            points.append((poi.x, poi.y))
        
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i != j:
                    path, metrics = dijkstra(self.env, p1, p2)
                    cache[(p1, p2)] = metrics.total_cost if path else float('inf')
        
        return cache
    
    def _get_distance(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """Retorna la distancia precomputada entre dos posiciones."""
        return self.distance_cache.get((from_pos, to_pos), float('inf'))
    
    def _evaluate_solution(self, poi_order: List[int]) -> Solution:
        """
        Evalúa una solución (orden de visita de POIs).
        Inserta automáticamente paradas de recarga en la Base cuando la batería es insuficiente.
        """
        self.evaluations += 1
        
        if not poi_order:
            return Solution(
                poi_order=[],
                fitness=0.0,
                total_cost=0.0,
                total_value=0.0,
                feasible=True
            )
        
        base_pos = (self.env.base.x, self.env.base.y)
        current_pos = base_pos
        battery = self.env.max_battery
        total_cost = 0.0
        total_value = 0.0
        feasible = True
        
        for poi_id in poi_order:
            poi = next((p for p in self.env.pois if p.id == poi_id), None)
            if poi is None:
                feasible = False
                break
            
            target_pos = (poi.x, poi.y)
            travel_cost = self._get_distance(current_pos, target_pos)
            dist_to_base = self._get_distance(target_pos, base_pos)
            collection_cost = 3.0  # Recolección + análisis
            
            if travel_cost == float('inf') or dist_to_base == float('inf'):
                feasible = False
                break

            # SEGURIDAD DE BATERÍA: ¿Alcanza para ir al POI y regresar a la base?
            needed_battery = travel_cost + collection_cost + dist_to_base
            
            # Si no alcanza, ir primero a la Base a recargar antes de ir al POI
            if battery < needed_battery:
                cost_to_base = self._get_distance(current_pos, base_pos)
                if cost_to_base == float('inf') or battery < cost_to_base:
                    # Ni siquiera alcanza para volver a la base: infactible
                    feasible = False
                    break
                
                # Desvío a la base para recargar
                total_cost += cost_to_base
                battery = self.env.max_battery  # Batería recargada
                current_pos = base_pos
                travel_cost = self._get_distance(current_pos, target_pos)

            # Viajar al POI y recolectar
            step_cost = travel_cost + collection_cost
            total_cost += step_cost
            battery -= step_cost
            
            if battery < 0:
                feasible = False
                break
            
            total_value += poi.scientific_value
            current_pos = target_pos
        
        # Al terminar todos los POIs, regresar a la base
        final_return_cost = self._get_distance(current_pos, base_pos)
        if final_return_cost == float('inf') or battery < final_return_cost:
            feasible = False
        else:
            total_cost += final_return_cost
            battery -= final_return_cost
        
        # Función de fitness
        if feasible:
            # Maximizar valor científico y minimizar el costo total del recorrido
            fitness = total_value - (total_cost * 0.1)
        else:
            fitness = -abs(battery) * 10 - total_cost
        
        return Solution(
            poi_order=poi_order,
            fitness=fitness,
            total_cost=total_cost,
            total_value=total_value,
            feasible=feasible
        )
    
    def _generate_random_solution(self) -> Solution:
        """Genera una solución aleatoria."""
        all_poi_ids = [poi.id for poi in self.env.pois]
        random.shuffle(all_poi_ids)
        return self._evaluate_solution(all_poi_ids)
    
    def _generate_neighbor(self, solution: Solution, method: str = "swap") -> Solution:
        """
        Genera una solución vecina usando el operador especificado.
        - swap: Intercambia dos POIs
        - 2opt: Invierte un segmento del camino
        - insert: Mueve un POI a otra posición
        """
        if len(solution.poi_order) < 2:
            return solution
        
        new_order = solution.poi_order.copy()
        
        if method == "swap":
            i, j = random.sample(range(len(new_order)), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        
        elif method == "2opt":
            i, j = sorted(random.sample(range(len(new_order)), 2))
            new_order[i:j+1] = reversed(new_order[i:j+1])
        
        elif method == "insert":
            i = random.randint(0, len(new_order) - 1)
            j = random.randint(0, len(new_order) - 1)
            poi = new_order.pop(i)
            new_order.insert(j, poi)
        
        return self._evaluate_solution(new_order)
    
    def hill_climbing(self, seed: int = 42) -> Solution:
        """
        Hill Climbing: búsqueda local voraz.
        """
        random.seed(seed)
        self.evaluations = 0
        
        current = self._generate_random_solution()
        
        while self.evaluations < self.max_evaluations:
            # Generar vecino
            neighbor = self._generate_neighbor(current, method="swap")
            
            if neighbor.fitness > current.fitness:
                current = neighbor
            else:
                # Intentar con otro operador
                neighbor = self._generate_neighbor(current, method="2opt")
                if neighbor.fitness > current.fitness:
                    current = neighbor
                else:
                    # Máximo local encontrado
                    break
        
        return current
    
    def simulated_annealing(self, seed: int = 42, 
                           initial_temp: float = 1000.0, 
                           cooling_rate: float = 0.995) -> Solution:
        """
        Simulated Annealing: recocido simulado con enfriamiento geométrico.
        """
        random.seed(seed)
        self.evaluations = 0
        
        current = self._generate_random_solution()
        best = current
        
        temperature = initial_temp
        
        while self.evaluations < self.max_evaluations and temperature > 0.01:
            # Generar vecino
            neighbor = self._generate_neighbor(current, method="swap")
            
            delta = neighbor.fitness - current.fitness
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current = neighbor
                if current.fitness > best.fitness:
                    best = current
            
            temperature *= cooling_rate
        
        return best
    
    def genetic_algorithm(self, seed: int = 42, 
                         population_size: int = 50, 
                         crossover_rate: float = 0.8, 
                         mutation_rate: float = 0.2) -> Solution:
        """
        Algoritmo Genético con selección por torneo y cruce OX (Order Crossover).
        """
        random.seed(seed)
        self.evaluations = 0
        
        # Inicializar población
        population = [self._generate_random_solution() for _ in range(population_size)]
        
        generation = 0
        while self.evaluations < self.max_evaluations:
            # Selección por torneo
            new_population = []
            
            while len(new_population) < population_size:
                # Torneo
                tournament = random.sample(population, min(5, len(population)))
                parent1 = max(tournament, key=lambda s: s.fitness)
                tournament = random.sample(population, min(5, len(population)))
                parent2 = max(tournament, key=lambda s: s.fitness)
                
                # Cruce
                if random.random() < crossover_rate:
                    child1, child2 = self._order_crossover(parent1, parent2)
                    child1 = self._evaluate_solution(child1)
                    child2 = self._evaluate_solution(child2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutación
                if random.random() < mutation_rate:
                    child1 = self._generate_neighbor(child1, method="swap")
                if random.random() < mutation_rate:
                    child2 = self._generate_neighbor(child2, method="swap")
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
            generation += 1
        
        best = max(population, key=lambda s: s.fitness)
        return best
    
    def _order_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX): cruce para permutaciones.
        """
        if len(parent1.poi_order) < 2:
            return parent1.poi_order.copy(), parent2.poi_order.copy()
        
        size = len(parent1.poi_order)
        
        # Seleccionar dos puntos de corte
        i, j = sorted(random.sample(range(size), 2))
        
        def ox(p1, p2):
            child = [-1] * size
            # Copiar segmento de p1
            child[i:j+1] = p1.poi_order[i:j+1]
            
            # Llenar el resto con orden de p2
            p2_remaining = [x for x in p2.poi_order if x not in child[i:j+1]]
            
            idx = 0
            for k in range(size):
                if child[k] == -1:
                    child[k] = p2_remaining[idx]
                    idx += 1
            
            return child
        
        return ox(parent1, parent2), ox(parent2, parent1)


def hill_climbing(env: Environment, max_evaluations: int = 10000, 
                  seed: int = 42) -> Solution:
    """Wrapper para Hill Climbing."""
    optimizer = MetaheuristicOptimizer(env, max_evaluations)
    return optimizer.hill_climbing(seed)


def simulated_annealing(env: Environment, max_evaluations: int = 10000, 
                        seed: int = 42, initial_temp: float = 1000.0, 
                        cooling_rate: float = 0.995) -> Solution:
    """Wrapper para Simulated Annealing."""
    optimizer = MetaheuristicOptimizer(env, max_evaluations)
    return optimizer.simulated_annealing(seed, initial_temp, cooling_rate)


def genetic_algorithm(env: Environment, max_evaluations: int = 10000, 
                      seed: int = 42, population_size: int = 50, 
                      crossover_rate: float = 0.8, mutation_rate: float = 0.2) -> Solution:
    """Wrapper para Algoritmo Genético."""
    optimizer = MetaheuristicOptimizer(env, max_evaluations)
    return optimizer.genetic_algorithm(seed, population_size, crossover_rate, mutation_rate)


if __name__ == "__main__":
    # Prueba de metaheurísticas
    try:
        from .environment import generate_instance
    except ImportError:
        from environment import generate_instance
    
    env = generate_instance(20, 20, num_pois=5, rock_ratio=0.15, seed=42)
    
    print("Probando Hill Climbing...")
    solution = hill_climbing(env, max_evaluations=1000, seed=42)
    print(f"  Fitness: {solution.fitness:.2f}")
    print(f"  POI order: {solution.poi_order}")
    print(f"  Total cost: {solution.total_cost:.2f}")
    print(f"  Total value: {solution.total_value:.2f}")
    print(f"  Feasible: {solution.feasible}")
    
    print("\nProbando Simulated Annealing...")
    solution = simulated_annealing(env, max_evaluations=1000, seed=42)
    print(f"  Fitness: {solution.fitness:.2f}")
    print(f"  POI order: {solution.poi_order}")
    print(f"  Total cost: {solution.total_cost:.2f}")
    print(f"  Total value: {solution.total_value:.2f}")
    print(f"  Feasible: {solution.feasible}")
    
    print("\nProbando Algoritmo Genético...")
    solution = genetic_algorithm(env, max_evaluations=1000, seed=42)
    print(f"  Fitness: {solution.fitness:.2f}")
    print(f"  POI order: {solution.poi_order}")
    print(f"  Total cost: {solution.total_cost:.2f}")
    print(f"  Total value: {solution.total_value:.2f}")
    print(f"  Feasible: {solution.feasible}")
