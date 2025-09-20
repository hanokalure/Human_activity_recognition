import torch
import psutil
import time
import json
from pathlib import Path
import subprocess
from datetime import datetime
import threading
import matplotlib.pyplot as plt
import numpy as np

class PerformanceMonitor:
    """Monitor system and GPU performance during training"""
    
    def __init__(self, log_dir="C:\\ASH_PROJECT\\outputs\\logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'timestamps': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'ram_usage': [],
            'gpu_temp': []
        }
        
        print(f"📊 PerformanceMonitor initialized")
        print(f"Log directory: {self.log_dir}")
    
    def get_gpu_info(self):
        """Get GPU information and usage"""
        if not torch.cuda.is_available():
            return None, None, None
        
        try:
            # GPU usage and memory
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # Try to get GPU usage percentage (requires nvidia-ml-py or nvidia-smi)
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_usage, gpu_temp = result.stdout.strip().split(',')
                    gpu_usage = float(gpu_usage)
                    gpu_temp = float(gpu_temp)
                else:
                    gpu_usage, gpu_temp = None, None
            except:
                gpu_usage, gpu_temp = None, None
            
            return {
                'memory_allocated_gb': gpu_memory_allocated,
                'memory_reserved_gb': gpu_memory_reserved,
                'memory_total_gb': gpu_memory_total,
                'memory_usage_percent': (gpu_memory_reserved / gpu_memory_total) * 100,
                'gpu_usage_percent': gpu_usage,
                'temperature_c': gpu_temp
            }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None
    
    def get_system_info(self):
        """Get CPU and RAM usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage_percent': cpu_percent,
                'ram_total_gb': memory.total / 1024**3,
                'ram_used_gb': memory.used / 1024**3,
                'ram_usage_percent': memory.percent
            }
        except Exception as e:
            print(f"Error getting system info: {e}")
            return None
    
    def _monitor_loop(self, interval=5):
        """Background monitoring loop"""
        while self.is_monitoring:
            timestamp = datetime.now()
            
            # Get GPU info
            gpu_info = self.get_gpu_info()
            
            # Get system info
            sys_info = self.get_system_info()
            
            # Store metrics
            self.metrics['timestamps'].append(timestamp)
            
            if gpu_info:
                self.metrics['gpu_usage'].append(gpu_info.get('gpu_usage_percent', 0))
                self.metrics['gpu_memory'].append(gpu_info.get('memory_usage_percent', 0))
                self.metrics['gpu_temp'].append(gpu_info.get('temperature_c', 0))
            else:
                self.metrics['gpu_usage'].append(0)
                self.metrics['gpu_memory'].append(0)
                self.metrics['gpu_temp'].append(0)
            
            if sys_info:
                self.metrics['cpu_usage'].append(sys_info.get('cpu_usage_percent', 0))
                self.metrics['ram_usage'].append(sys_info.get('ram_usage_percent', 0))
            else:
                self.metrics['cpu_usage'].append(0)
                self.metrics['ram_usage'].append(0)
            
            time.sleep(interval)
    
    def start_monitoring(self, interval=5):
        """Start performance monitoring"""
        if self.is_monitoring:
            print("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"🚀 Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            print("Monitoring not running")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        print("⏹️ Performance monitoring stopped")
    
    def get_current_stats(self):
        """Get current performance statistics"""
        gpu_info = self.get_gpu_info()
        sys_info = self.get_system_info()
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'gpu': gpu_info,
            'system': sys_info
        }
        
        return stats
    
    def print_current_stats(self):
        """Print current performance statistics"""
        stats = self.get_current_stats()
        
        print("\n📊 Current Performance Stats:")
        print("=" * 40)
        
        if stats['gpu']:
            gpu = stats['gpu']
            print(f"GPU Usage: {gpu.get('gpu_usage_percent', 'N/A')}%")
            print(f"GPU Memory: {gpu['memory_usage_percent']:.1f}% ({gpu['memory_reserved_gb']:.2f}/{gpu['memory_total_gb']:.2f} GB)")
            print(f"GPU Temp: {gpu.get('temperature_c', 'N/A')}°C")
        
        if stats['system']:
            sys = stats['system']
            print(f"CPU Usage: {sys['cpu_usage_percent']:.1f}%")
            print(f"RAM Usage: {sys['ram_usage_percent']:.1f}% ({sys['ram_used_gb']:.2f}/{sys['ram_total_gb']:.2f} GB)")
        
        print("=" * 40)
    
    def save_metrics(self, filename=None):
        """Save collected metrics to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        # Convert timestamps to strings
        metrics_to_save = self.metrics.copy()
        metrics_to_save['timestamps'] = [ts.isoformat() for ts in self.metrics['timestamps']]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"💾 Metrics saved to: {filepath}")
        return filepath
    
    def plot_metrics(self, save_plot=True):
        """Plot performance metrics"""
        if not self.metrics['timestamps']:
            print("No metrics to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Monitoring Results', fontsize=16)
        
        timestamps = self.metrics['timestamps']
        
        # GPU Usage
        if any(x > 0 for x in self.metrics['gpu_usage'] if x is not None):
            axes[0, 0].plot(timestamps, self.metrics['gpu_usage'], 'b-', linewidth=2)
            axes[0, 0].set_title('GPU Usage (%)')
            axes[0, 0].set_ylabel('Usage %')
            axes[0, 0].grid(True)
        
        # GPU Memory
        axes[0, 1].plot(timestamps, self.metrics['gpu_memory'], 'r-', linewidth=2)
        axes[0, 1].set_title('GPU Memory Usage (%)')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True)
        
        # CPU Usage
        axes[1, 0].plot(timestamps, self.metrics['cpu_usage'], 'g-', linewidth=2)
        axes[1, 0].set_title('CPU Usage (%)')
        axes[1, 0].set_ylabel('CPU %')
        axes[1, 0].grid(True)
        
        # RAM Usage
        axes[1, 1].plot(timestamps, self.metrics['ram_usage'], 'orange', linewidth=2)
        axes[1, 1].set_title('RAM Usage (%)')
        axes[1, 1].set_ylabel('RAM %')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.log_dir / f"performance_plot_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 Plot saved to: {plot_path}")
        
        plt.show()

class TrainingProfiler:
    """Profile training performance and bottlenecks"""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start_timer(self, name):
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name):
        """End timing an operation"""
        if name not in self.start_times:
            print(f"Warning: Timer '{name}' was not started")
            return
        
        elapsed = time.time() - self.start_times[name]
        
        if name not in self.timings:
            self.timings[name] = []
        
        self.timings[name].append(elapsed)
        del self.start_times[name]
        
        return elapsed
    
    def get_stats(self):
        """Get timing statistics"""
        stats = {}
        
        for name, times in self.timings.items():
            stats[name] = {
                'count': len(times),
                'total': sum(times),
                'average': np.mean(times),
                'min': min(times),
                'max': max(times),
                'std': np.std(times)
            }
        
        return stats
    
    def print_stats(self):
        """Print timing statistics"""
        stats = self.get_stats()
        
        print("\n⏱️ Training Profiler Results:")
        print("=" * 60)
        print(f"{'Operation':<20} {'Count':<8} {'Avg Time':<12} {'Total Time':<12}")
        print("=" * 60)
        
        for name, stat in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            print(f"{name:<20} {stat['count']:<8} {stat['average']:<12.3f}s {stat['total']:<12.2f}s")
        
        print("=" * 60)

def benchmark_model_performance(model, input_shape=(4, 3, 8, 112, 112), device='cuda', num_iterations=100):
    """Benchmark model inference performance"""
    model.eval()
    device = torch.device(device)
    
    # Warm up
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    
    for _ in range(num_iterations):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = input_shape[0] / avg_time  # frames per second
    
    print(f"\n🏃 Model Performance Benchmark:")
    print(f"Input shape: {input_shape}")
    print(f"Device: {device}")
    print(f"Iterations: {num_iterations}")
    print(f"Average inference time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"Throughput: {fps:.2f} FPS")
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'fps': fps,
        'all_times': times
    }

def main():
    """Example usage of performance monitoring"""
    # Create monitor
    monitor = PerformanceMonitor()
    
    # Print current stats
    monitor.print_current_stats()
    
    # Start monitoring
    monitor.start_monitoring(interval=2)
    
    # Simulate some work
    print("Simulating training for 30 seconds...")
    time.sleep(30)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Save and plot results
    monitor.save_metrics()
    monitor.plot_metrics()
    
    # Example profiler usage
    profiler = TrainingProfiler()
    
    # Simulate some operations
    for i in range(10):
        profiler.start_timer('data_loading')
        time.sleep(0.1)  # Simulate data loading
        profiler.end_timer('data_loading')
        
        profiler.start_timer('forward_pass')
        time.sleep(0.05)  # Simulate forward pass
        profiler.end_timer('forward_pass')
        
        profiler.start_timer('backward_pass')
        time.sleep(0.08)  # Simulate backward pass
        profiler.end_timer('backward_pass')
    
    profiler.print_stats()

if __name__ == "__main__":
    main()