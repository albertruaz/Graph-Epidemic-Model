from utils import Config
from epidemic_simulator import EpidemicSimulator


def run_basic_simulation():
    """Basic simulation without animation"""
    print("=== Basic Simulation (No Animation) ===")
    
    cfg = Config(
        N=50,
        N1=3,
        N2=2,
        N3=1,
        tau=0.3,
        alpha=0.1,
        xi=0.02,
        T=50,
        seed=42,
        enable_animation=False  # Animation disabled
    )
    
    simulator = EpidemicSimulator(cfg, init_infected=list(range(3)))
    results = simulator.run_simulation(save_results=True, create_plots=True)
    
    print(f"R0: {results['r0']:.2f}")
    print(f"Results saved to: {results['result_path']}")
    return results

def run_animation_simulation():
    """Simulation with full animation features"""
    print("\n=== Animation Simulation ===")
    
    cfg = Config(
        N=30,        # Smaller network for better animation performance
        N1=2,
        N2=2,
        N3=1,
        tau=0.4,
        alpha=0.08,
        xi=0.03,
        T=40,
        seed=42,
        # Animation settings
        enable_animation=True,
        save_mp4=True,
        save_gif=True,
        show_animation=False,  # Set to True to display animation window
        animation_fps=8,
        animation_interval=150
    )
    
    simulator = EpidemicSimulator(cfg, init_infected=[cfg.N//2])
    results = simulator.run_simulation(save_results=True, create_plots=True)
    
    print(f"R0: {results['r0']:.2f}")
    print(f"Results with animations saved to: {results['result_path']}")
    return results

if __name__ == "__main__":
    print("SIRS Epidemic Model with Configurable Animation Features")
    print("=" * 60)
    
    # Run different types of simulations
    basic_results = run_animation_simulation()
    
    # Uncomment the following lines to run animation examples:
    # animation_results = run_animation_simulation()
    # custom_results = run_custom_animation()
    # demo_results = demonstrate_config_control()
    
    print("\n" + "=" * 60)
    print("Demo completed! Check the results folders for outputs.")
    print("\nTo enable animations, uncomment the animation examples above.")
    print("Animation features are controlled by Config parameters:")
    print("  - enable_animation: bool - Master switch")
    print("  - save_mp4: bool - Save MP4 files")
    print("  - save_gif: bool - Save GIF files")
    print("  - show_animation: bool - Display animation window")
    print("  - animation_fps: int - Animation speed")
    print("  - animation_interval: int - Frame interval (ms)") 