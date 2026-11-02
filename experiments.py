"""
Script Experimental Principal
Ejecuta el protocolo experimental completo para evaluar todos los módulos (A, B, D y E - Bonificable).
"""

import sys
import os
import time
import statistics
from typing import List, Dict, Tuple
import csv

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.environment import generate_instance, Environment
from src.planner import ClassicalPlanner
from src.metaheuristics import hill_climbing, simulated_annealing, genetic_algorithm
from src.search import dijkstra, a_star, weighted_a_star, euclidean_distance, SearchMetrics
from src.rules import RoverRuleExpertSystem, Decision


class ExperimentRunner:
    """Ejecuta el protocolo experimental."""
    
    def __init__(self, seeds: List[int] = None):
        self.seeds = seeds if seeds else [42, 101, 202, 303, 404]
        self.results = {
            "search": [],
            "planner": [],
            "metaheuristics": [],
            "rules": []
        }
    
    def generate_instances(self) -> List[Environment]:
        """Genera instancias de prueba con variabilidad de tamaño y dificultad."""
        instances = []
        configs = [
            {"width": 15, "height": 15, "num_pois": 4, "rock_ratio": 0.10, "desc": "Pequeño / Fácil"},
            {"width": 20, "height": 20, "num_pois": 5, "rock_ratio": 0.15, "desc": "Medio / Normal"},
            {"width": 20, "height": 20, "num_pois": 6, "rock_ratio": 0.25, "desc": "Medio / Rocoso"},
            {"width": 25, "height": 25, "num_pois": 8, "rock_ratio": 0.15, "desc": "Grande / Normal"},
            {"width": 30, "height": 30, "num_pois": 10, "rock_ratio": 0.30, "desc": "Grande / Difícil"}
        ]
        
        for i, seed in enumerate(self.seeds):
            cfg = configs[i % len(configs)]
            env = generate_instance(
                width=cfg["width"], 
                height=cfg["height"], 
                num_pois=cfg["num_pois"], 
                rock_ratio=cfg["rock_ratio"], 
                seed=seed
            )
            instances.append(env)
            print(f"Instancia {i+1} [{cfg['desc']}] generada con semilla {seed}: {env}")
        return instances

    def run_search_experiments(self, instances: List[Environment]):
        """Ejecuta experimentos de Búsqueda evaluando múltiples rutas por instancia."""
        print("\n" + "="*70)
        print("MÓDULO A: BÚSQUEDA EN ESPACIOS DE ESTADOS")
        print("="*70)
        
        algorithms = [
            ("Dijkstra", lambda env, s, g: dijkstra(env, s, g)),
            ("A* (Euclídea)", lambda env, s, g: a_star(env, s, g, heuristic=lambda x1, y1, x2, y2: euclidean_distance(x1, y1, x2, y2))),
            ("A* (Admisible)", lambda env, s, g: a_star(env, s, g)),
            ("Weighted A* (w=1.5)", lambda env, s, g: weighted_a_star(env, s, g, weight=1.5)),
            ("Weighted A* (w=3.0)", lambda env, s, g: weighted_a_star(env, s, g, weight=3.0))
        ]
        
        for i, env in enumerate(instances):
            seed = self.seeds[i]
            print(f"\n--- Instancia {i+1} (semilla {seed}) ---")
            
            start = (env.base.x, env.base.y)
            target_pois = env.pois[:min(3, len(env.pois))]
            
            for p_idx, poi in enumerate(target_pois):
                goal = (poi.x, poi.y)
                print(f"  Ruta {p_idx+1}: Base {start} -> POI_{p_idx+1} {goal}")
                
                for alg_name, alg_func in algorithms:
                    path, metrics = alg_func(env, start, goal)
                    
                    result = {
                        "instance": i + 1,
                        "seed": seed,
                        "route": f"Base->POI_{p_idx+1}",
                        "algorithm": alg_name,
                        "path_length": metrics.path_length,
                        "total_cost": metrics.total_cost,
                        "nodes_expanded": metrics.nodes_expanded,
                        "cpu_time": metrics.cpu_time,
                        "found_path": len(path) > 0
                    }
                    self.results["search"].append(result)
                    print(f"    {alg_name:<22} | Costo: {metrics.total_cost:6.2f} | Nodos: {metrics.nodes_expanded:4d} | Tiempo: {metrics.cpu_time:.5f}s")
    
    def run_planner_experiments(self, instances: List[Environment]):
        """Ejecuta experimentos de Planificación Clásica."""
        print("\n" + "="*70)
        print("MÓDULO SIMBÓLICO: PLANIFICACIÓN CLÁSICA")
        print("="*70)
        
        for i, env in enumerate(instances):
            seed = self.seeds[i]
            print(f"\n--- Instancia {i+1} (semilla {seed}) ---")
            
            planner = ClassicalPlanner(env)
            plan, total_cost, stats = planner.plan_mission()
            
            result = {
                "instance": i + 1,
                "seed": seed,
                "plan_length": stats["plan_length"],
                "total_cost": total_cost,
                "actions_by_type": stats["actions_by_type"]
            }
            
            self.results["planner"].append(result)
            
            print(f"Plan encontrado:")
            print(f"  Longitud: {stats['plan_length']} acciones")
            print(f"  Costo total: {total_cost:.2f}")
            print(f"  Acciones por tipo: {stats['actions_by_type']}")
    
    def run_metaheuristic_experiments(self, instances: List[Environment], runs_per_instance: int = 15):
        """Ejecuta experimentos de Metaheurísticas."""
        print("\n" + "="*70)
        print("MÓDULO D: METAHEURÍSTICAS")
        print("="*70)
        
        algorithms = [
            ("Hill Climbing", lambda env, seed: hill_climbing(env, max_evaluations=5000, seed=seed)),
            ("Simulated Annealing", lambda env, seed: simulated_annealing(env, max_evaluations=5000, seed=seed)),
            ("Genetic Algorithm", lambda env, seed: genetic_algorithm(env, max_evaluations=5000, seed=seed))
        ]
        
        for i, env in enumerate(instances):
            seed = self.seeds[i]
            print(f"\n--- Instancia {i+1} (semilla {seed}) ---")
            
            for alg_name, alg_func in algorithms:
                print(f"\n  {alg_name} ({runs_per_instance} corridas):")
                
                fitness_values = []
                cost_values = []
                value_values = []
                feasible_count = 0
                
                for run in range(runs_per_instance):
                    run_seed = seed + run * 1000
                    solution = alg_func(env, run_seed)
                    
                    fitness_values.append(solution.fitness)
                    cost_values.append(solution.total_cost)
                    value_values.append(solution.total_value)
                    if solution.feasible:
                        feasible_count += 1
                
                mean_fitness = statistics.mean(fitness_values)
                std_fitness = statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0.0
                mean_cost = statistics.mean(cost_values)
                std_cost = statistics.stdev(cost_values) if len(cost_values) > 1 else 0.0
                mean_value = statistics.mean(value_values)
                std_value = statistics.stdev(value_values) if len(value_values) > 1 else 0.0
                feasible_rate = feasible_count / runs_per_instance
                
                result = {
                    "instance": i + 1,
                    "seed": seed,
                    "algorithm": alg_name,
                    "runs": runs_per_instance,
                    "mean_fitness": mean_fitness,
                    "std_fitness": std_fitness,
                    "mean_cost": mean_cost,
                    "std_cost": std_cost,
                    "mean_value": mean_value,
                    "std_value": std_value,
                    "feasible_rate": feasible_rate
                }
                
                self.results["metaheuristics"].append(result)
                
                print(f"    Fitness: {mean_fitness:.2f} ± {std_fitness:.2f}")
                print(f"    Costo: {mean_cost:.2f} ± {std_cost:.2f}")
                print(f"    Valor científico: {mean_value:.2f} ± {std_value:.2f}")
                print(f"    Tasa factibilidad: {feasible_rate:.2%}")

    def run_rules_experiments(self, instances: List[Environment]):
        """
        Ejecuta experimentos del Módulo E (Sistemas Basados en Reglas - Bonificable).
        Evalúa el motor de Forward Chaining en situaciones operativas reales.
        """
        print("\n" + "="*70)
        print("MÓDULO E: SISTEMAS BASADOS EN REGLAS (BONIFICABLE)")
        print("="*70)

        expert = RoverRuleExpertSystem()

        for i, env in enumerate(instances):
            seed = self.seeds[i]
            print(f"\n--- Instancia {i+1} (semilla {seed}) ---")

            # 1. Decisión estratégica inicial en Base
            f_init = expert.extract_facts_from_rover(
                env, rover_pos=(env.base.x, env.base.y), battery=env.max_battery,
                memory_used=0, collected_ids=set(), analyzed_ids=set(), transmitted_ids=set()
            )
            dec_init, r_init, rules_init = expert.decide(f_init)
            print(f"  [Inicio Misión] Decisión: {dec_init.value:<28} | Regla: {rules_init[-1] if rules_init else 'None'}")
            print(f"                  Motivo: {r_init}")

            # 2. Decisión de recolección in situ en el primer POI
            p0 = env.pois[0]
            f_poi = expert.extract_facts_from_rover(
                env, rover_pos=(p0.x, p0.y), battery=env.max_battery * 0.85,
                memory_used=0, collected_ids=set(), analyzed_ids=set(), transmitted_ids=set()
            )
            dec_poi, r_poi, rules_poi = expert.decide(f_poi)
            print(f"  [Llegada a POI] Decisión: {dec_poi.value:<28} | Regla: {rules_poi[-1] if rules_poi else 'None'}")

            # 3. Decisión ante contingencia crítica de batería
            f_crit = expert.extract_facts_from_rover(
                env, rover_pos=(env.width - 2, env.height - 2), battery=18.0,
                memory_used=1, collected_ids={0}, analyzed_ids={0}, transmitted_ids=set()
            )
            dec_crit, r_crit, rules_crit = expert.decide(f_crit)
            print(f"  [Alerta Batería] Decisión: {dec_crit.value:<27} | Reglas disparadas: {rules_crit}")

            # Guardar resultados
            self.results["rules"].append({
                "instance": i + 1,
                "seed": seed,
                "initial_decision": dec_init.value,
                "poi_decision": dec_poi.value,
                "emergency_decision": dec_crit.value,
                "emergency_rules": str(rules_crit)
            })
    
    def print_summary_tables(self):
        """Imprime tablas resumen de los resultados."""
        print("\n" + "="*70)
        print("TABLAS RESUMEN")
        print("="*70)
        
        # Tabla de búsqueda
        print("\n--- Módulo A: Búsqueda (Media ± Desviación) ---")
        print(f"{'Algoritmo':<25} {'Costo':<15} {'Nodos':<12} {'Tiempo (s)':<12}")
        print("-" * 70)
        
        alg_groups = {}
        for r in self.results["search"]:
            alg = r["algorithm"]
            if alg not in alg_groups:
                alg_groups[alg] = []
            alg_groups[alg].append(r)
        
        for alg, results in alg_groups.items():
            costs = [r["total_cost"] for r in results if r["found_path"]]
            nodes = [r["nodes_expanded"] for r in results]
            times = [r["cpu_time"] for r in results]
            
            mean_cost = statistics.mean(costs) if costs else 0
            std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
            mean_nodes = statistics.mean(nodes) if nodes else 0
            std_nodes = statistics.stdev(nodes) if len(nodes) > 1 else 0
            mean_time = statistics.mean(times) if times else 0
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            
            print(f"{alg:<25} {mean_cost:.2f} ± {std_cost:.2f}    {mean_nodes:.0f} ± {std_nodes:.0f}    {mean_time:.4f} ± {std_time:.4f}")
        
        # Tabla de metaheurísticas
        print("\n--- Módulo D: Metaheurísticas (Media ± Desviación) ---")
        print(f"{'Algoritmo':<25} {'Fitness':<15} {'Costo':<15} {'Valor':<15}")
        print("-" * 75)
        
        mh_alg_groups = {}
        for r in self.results["metaheuristics"]:
            alg = r["algorithm"]
            if alg not in mh_alg_groups:
                mh_alg_groups[alg] = []
            mh_alg_groups[alg].append(r)
        
        for alg, results in mh_alg_groups.items():
            fitnesses = [r["mean_fitness"] for r in results]
            costs = [r["mean_cost"] for r in results]
            values = [r["mean_value"] for r in results]
            
            mean_fitness = statistics.mean(fitnesses) if fitnesses else 0
            std_fitness = statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0
            mean_cost = statistics.mean(costs) if costs else 0
            std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
            mean_value = statistics.mean(values) if values else 0
            std_value = statistics.stdev(values) if len(values) > 1 else 0
            
            print(f"{alg:<25} {mean_fitness:.2f} ± {std_fitness:.2f}    {mean_cost:.2f} ± {std_cost:.2f}    {mean_value:.2f} ± {std_value:.2f}")
    
    def save_results_to_csv(self, filename: str = "results.csv"):
        """Guarda los resultados en un archivo CSV."""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Resultados de búsqueda
            writer.writerow(["=== MÓDULO A: BÚSQUEDA ==="])
            writer.writerow(["Instance", "Seed", "Algorithm", "Path Length", "Total Cost", 
                            "Nodes Expanded", "CPU Time", "Found Path"])
            for r in self.results["search"]:
                writer.writerow([r["instance"], r["seed"], r["algorithm"], 
                               r["path_length"], r["total_cost"], 
                               r["nodes_expanded"], r["cpu_time"], r["found_path"]])
            
            # Resultados de planificación
            writer.writerow([])
            writer.writerow(["=== MÓDULO SIMBÓLICO: PLANIFICACIÓN ==="])
            writer.writerow(["Instance", "Seed", "Plan Length", "Total Cost", "Actions by Type"])
            for r in self.results["planner"]:
                writer.writerow([r["instance"], r["seed"], r["plan_length"], 
                               r["total_cost"], str(r["actions_by_type"])])
            
            # Resultados de metaheurísticas
            writer.writerow([])
            writer.writerow(["=== MÓDULO D: METAHEURÍSTICAS ==="])
            writer.writerow(["Instance", "Seed", "Algorithm", "Runs", "Mean Fitness", "Std Fitness",
                            "Mean Cost", "Std Cost", "Mean Value", "Std Value", "Feasible Rate"])
            for r in self.results["metaheuristics"]:
                writer.writerow([r["instance"], r["seed"], r["algorithm"], r["runs"],
                               r["mean_fitness"], r["std_fitness"],
                               r["mean_cost"], r["std_cost"],
                               r["mean_value"], r["std_value"], r["feasible_rate"]])

            # Resultados de reglas (Bono E)
            writer.writerow([])
            writer.writerow(["=== MÓDULO E: SISTEMA BASADO EN REGLAS (BONO) ==="])
            writer.writerow(["Instance", "Seed", "Initial Decision", "POI Decision", "Emergency Decision", "Emergency Rules Fired"])
            for r in self.results["rules"]:
                writer.writerow([r["instance"], r["seed"], r["initial_decision"], 
                                 r["poi_decision"], r["emergency_decision"], r["emergency_rules"]])
        
        print(f"\nResultados guardados en {filename}")


def main():
    """Función principal."""
    print("="*70)
    print("PROYECTO 1: ROVER AUTÓNOMO - PROTOCOLO EXPERIMENTAL COMPLETO")
    print("="*70)
    
    runner = ExperimentRunner(seeds=[42, 101, 202, 303, 404])
    
    # 1. Generar instancias
    print("\nGenerando instancias...")
    instances = runner.generate_instances()
    
    # 2. Ejecutar experimentos de todos los módulos
    runner.run_search_experiments(instances)
    runner.run_planner_experiments(instances)
    runner.run_metaheuristic_experiments(instances, runs_per_instance=15)
    runner.run_rules_experiments(instances)  # <--- ¡Módulo E incluido!
    
    # 3. Imprimir resumen y guardar resultados
    runner.print_summary_tables()
    runner.save_results_to_csv("results.csv")
    
    print("\n" + "="*70)
    print("EXPERIMENTOS COMPLETADOS EXITOSAMENTE")
    print("="*70)


if __name__ == "__main__":
    main()