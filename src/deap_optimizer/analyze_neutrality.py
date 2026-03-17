#!/usr/bin/env python3
"""
Parameter Neutrality Analysis Script

Analyzes neutrality_snapshots.json to discover:
1. Which parameters are frequently neutral (don't affect fitness)
2. Conditional relationships (param X is neutral when param Y has value Z)
3. Parameter dependency graphs
4. Recommendations for search space reduction
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class ParamStats:
    """Statistics for a single parameter across all snapshots."""
    name: str
    times_neutral: int = 0
    times_converged: int = 0
    times_seen: int = 0
    neutral_contexts: List[Dict] = None  # Context when this param was neutral
    converged_values: List[float] = None  # Values when converged
    
    def __post_init__(self):
        if self.neutral_contexts is None:
            self.neutral_contexts = []
        if self.converged_values is None:
            self.converged_values = []
    
    @property
    def neutrality_rate(self) -> float:
        return self.times_neutral / self.times_seen if self.times_seen > 0 else 0
    
    @property
    def convergence_rate(self) -> float:
        return self.times_converged / self.times_seen if self.times_seen > 0 else 0


def load_snapshots(path: str) -> Dict:
    """Load the neutrality snapshots JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def analyze_parameter_frequencies(snapshots: List[Dict]) -> Dict[str, ParamStats]:
    """Analyze how often each parameter is neutral vs converged."""
    stats = {}
    
    for snapshot in snapshots:
        neutral_params = set(snapshot.get('neutral_params', []))
        converged_params = set(snapshot.get('converged_params', []))
        param_context = snapshot.get('param_context', {})
        
        all_params = neutral_params | converged_params
        
        for param in all_params:
            if param not in stats:
                stats[param] = ParamStats(name=param)
            
            stats[param].times_seen += 1
            
            if param in neutral_params:
                stats[param].times_neutral += 1
                # Store the context (other param values) when this was neutral
                context = {
                    'fitness': snapshot.get('global_best_fitness'),
                    'other_params': {}
                }
                for other_param, other_data in param_context.items():
                    if other_param != param and other_data.get('is_converged'):
                        context['other_params'][other_param] = other_data.get('best_value')
                stats[param].neutral_contexts.append(context)
            
            if param in converged_params:
                stats[param].times_converged += 1
                if param in param_context:
                    stats[param].converged_values.append(
                        param_context[param].get('best_value')
                    )
    
    return stats


def find_conditional_neutrality(
    snapshots: List[Dict],
    param_stats: Dict[str, ParamStats],
    min_occurrences: int = 3
) -> Dict[str, List[Dict]]:
    """
    Find conditions under which parameters become neutral.
    
    Returns rules like:
    "long_ema_span_1 is neutral when long_ema_span_0 < 300"
    """
    rules = defaultdict(list)
    
    for param_name, stats in param_stats.items():
        if stats.times_neutral < min_occurrences:
            continue
        
        # Analyze contexts when this param was neutral
        # Look for patterns in other converged params
        other_param_values = defaultdict(list)
        
        for context in stats.neutral_contexts:
            for other_param, value in context['other_params'].items():
                if value is not None:
                    other_param_values[other_param].append(value)
        
        # For each other param, check if there's a consistent pattern
        for other_param, values in other_param_values.items():
            if len(values) < min_occurrences:
                continue
            
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            # If values are tightly clustered, this might be a condition
            if std_val < 0.1 * abs(mean_val) if mean_val != 0 else std_val < 0.01:
                rules[param_name].append({
                    'condition_param': other_param,
                    'condition_type': 'near_value',
                    'value': mean_val,
                    'std': std_val,
                    'occurrences': len(values),
                    'description': f"{param_name} is neutral when {other_param} ≈ {mean_val:.4f} (±{std_val:.4f})"
                })
            
            # Check for threshold patterns (all values above or below a point)
            # This could indicate "param X doesn't matter when param Y > threshold"
            if max_val < mean_val * 1.5 and min_val > 0:  # All positive, bounded
                rules[param_name].append({
                    'condition_param': other_param,
                    'condition_type': 'range',
                    'min': min_val,
                    'max': max_val,
                    'occurrences': len(values),
                    'description': f"{param_name} is neutral when {other_param} in [{min_val:.4f}, {max_val:.4f}]"
                })
    
    return dict(rules)


def find_co_neutral_params(snapshots: List[Dict], min_co_occurrence: int = 3) -> List[Tuple[str, str, int]]:
    """Find parameters that are frequently neutral together."""
    co_occurrence = defaultdict(int)
    
    for snapshot in snapshots:
        neutral_params = snapshot.get('neutral_params', [])
        # Count pairs
        for i, p1 in enumerate(neutral_params):
            for p2 in neutral_params[i+1:]:
                pair = tuple(sorted([p1, p2]))
                co_occurrence[pair] += 1
    
    # Filter and sort by frequency
    results = [
        (p1, p2, count) 
        for (p1, p2), count in co_occurrence.items() 
        if count >= min_co_occurrence
    ]
    results.sort(key=lambda x: -x[2])
    
    return results


def find_never_neutral_params(param_stats: Dict[str, ParamStats]) -> List[str]:
    """Find parameters that are never neutral (always matter)."""
    return [
        name for name, stats in param_stats.items()
        if stats.times_neutral == 0 and stats.times_seen > 0
    ]


def find_always_neutral_params(
    param_stats: Dict[str, ParamStats],
    threshold: float = 0.8
) -> List[Tuple[str, float]]:
    """Find parameters that are almost always neutral (rarely matter)."""
    results = [
        (name, stats.neutrality_rate)
        for name, stats in param_stats.items()
        if stats.neutrality_rate >= threshold and stats.times_seen >= 3
    ]
    results.sort(key=lambda x: -x[1])
    return results


def analyze_convergence_patterns(param_stats: Dict[str, ParamStats]) -> Dict[str, Dict]:
    """Analyze where parameters tend to converge."""
    patterns = {}
    
    for name, stats in param_stats.items():
        if len(stats.converged_values) < 3:
            continue
        
        values = np.array(stats.converged_values)
        patterns[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'n_samples': len(values),
            'is_consistent': bool(np.std(values) < 0.1 * abs(np.mean(values)) if np.mean(values) != 0 else np.std(values) < 0.01)
        }
    
    return patterns


def build_dependency_graph(
    snapshots: List[Dict],
    param_stats: Dict[str, ParamStats]
) -> Dict[str, List[str]]:
    """
    Build a graph showing which params might control/gate other params.
    
    If param A is always converged when param B is neutral, A might gate B.
    """
    # For each neutral param, track which params were converged
    gating_evidence = defaultdict(lambda: defaultdict(int))
    
    for snapshot in snapshots:
        neutral_params = set(snapshot.get('neutral_params', []))
        converged_params = set(snapshot.get('converged_params', []))
        
        for neutral_p in neutral_params:
            for converged_p in converged_params:
                gating_evidence[neutral_p][converged_p] += 1
    
    # Find strong gating relationships
    graph = {}
    for neutral_p, converged_counts in gating_evidence.items():
        stats = param_stats.get(neutral_p)
        if not stats or stats.times_neutral < 3:
            continue
        
        # Find params that are almost always converged when this is neutral
        gaters = []
        for converged_p, count in converged_counts.items():
            if count >= stats.times_neutral * 0.8:  # 80% co-occurrence
                gaters.append(converged_p)
        
        if gaters:
            graph[neutral_p] = gaters
    
    return graph


def generate_recommendations(
    param_stats: Dict[str, ParamStats],
    always_neutral: List[Tuple[str, float]],
    never_neutral: List[str],
    convergence_patterns: Dict[str, Dict],
    conditional_rules: Dict[str, List[Dict]]
) -> List[str]:
    """Generate actionable recommendations."""
    recommendations = []
    
    # Recommend fixing always-neutral params
    if always_neutral:
        recommendations.append("\n🎯 PARAMETERS TO CONSIDER FIXING (frequently neutral):")
        for param, rate in always_neutral[:10]:
            if param in convergence_patterns:
                pattern = convergence_patterns[param]
                recommendations.append(
                    f"  • {param}: neutral {rate*100:.0f}% of time. "
                    f"When it converges, typically to {pattern['mean']:.4f} (±{pattern['std']:.4f})"
                )
            else:
                recommendations.append(f"  • {param}: neutral {rate*100:.0f}% of time")
    
    # Highlight critical params
    if never_neutral:
        recommendations.append("\n⚡ CRITICAL PARAMETERS (never neutral - always affect fitness):")
        for param in never_neutral[:10]:
            if param in convergence_patterns:
                pattern = convergence_patterns[param]
                recommendations.append(
                    f"  • {param}: converges to {pattern['mean']:.4f} (±{pattern['std']:.4f})"
                )
            else:
                recommendations.append(f"  • {param}")
    
    # Conditional recommendations
    if conditional_rules:
        recommendations.append("\n🔗 CONDITIONAL RELATIONSHIPS DETECTED:")
        for param, rules in list(conditional_rules.items())[:10]:
            for rule in rules[:2]:
                recommendations.append(f"  • {rule['description']}")
    
    # Search space reduction estimate
    if always_neutral:
        n_fixable = len([p for p, r in always_neutral if r > 0.9])
        total_params = len(param_stats)
        if n_fixable > 0:
            reduction = (1 - (total_params - n_fixable) / total_params) * 100
            recommendations.append(
                f"\n📉 SEARCH SPACE REDUCTION: Could potentially fix {n_fixable}/{total_params} "
                f"parameters ({reduction:.0f}% reduction)"
            )
    
    return recommendations


def print_report(
    data: Dict,
    param_stats: Dict[str, ParamStats],
    always_neutral: List[Tuple[str, float]],
    never_neutral: List[str],
    co_neutral: List[Tuple[str, str, int]],
    convergence_patterns: Dict[str, Dict],
    conditional_rules: Dict[str, List[Dict]],
    dependency_graph: Dict[str, List[str]],
    recommendations: List[str]
):
    """Print a comprehensive analysis report."""
    snapshots = data['snapshots']
    
    print("=" * 80)
    print("PARAMETER NEUTRALITY ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nDataset: {len(snapshots)} snapshots analyzed")
    print(f"Parameters tracked: {len(param_stats)}")
    
    # Summary statistics
    print("\n" + "-" * 40)
    print("PARAMETER SUMMARY")
    print("-" * 40)
    
    # Sort by neutrality rate
    sorted_stats = sorted(
        param_stats.items(),
        key=lambda x: (-x[1].neutrality_rate, x[0])
    )
    
    print(f"\n{'Parameter':<45} {'Neutral%':>10} {'Converged%':>10} {'Seen':>6}")
    print("-" * 75)
    for name, stats in sorted_stats:
        print(f"{name:<45} {stats.neutrality_rate*100:>9.1f}% {stats.convergence_rate*100:>9.1f}% {stats.times_seen:>6}")
    
    # Co-neutral params
    if co_neutral:
        print("\n" + "-" * 40)
        print("FREQUENTLY CO-NEUTRAL PARAMETERS")
        print("-" * 40)
        print("(Parameters that are often neutral together - may be related)")
        for p1, p2, count in co_neutral[:15]:
            print(f"  {p1} + {p2}: {count} times")
    
    # Dependency graph
    if dependency_graph:
        print("\n" + "-" * 40)
        print("POTENTIAL PARAMETER DEPENDENCIES")
        print("-" * 40)
        print("(When param X is neutral, these params are usually converged)")
        for neutral_p, gaters in list(dependency_graph.items())[:10]:
            print(f"  {neutral_p} may be gated by:")
            for g in gaters[:5]:
                print(f"    → {g}")
    
    # Convergence patterns
    if convergence_patterns:
        print("\n" + "-" * 40)
        print("CONVERGENCE PATTERNS")
        print("-" * 40)
        consistent = [(k, v) for k, v in convergence_patterns.items() if v['is_consistent']]
        if consistent:
            print("\nParameters with consistent convergence values:")
            for name, pattern in sorted(consistent, key=lambda x: x[1]['std'])[:10]:
                print(f"  {name}: {pattern['mean']:.6f} (±{pattern['std']:.6f}, n={pattern['n_samples']})")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 80)


def export_analysis_json(
    output_path: str,
    data: Dict,
    param_stats: Dict[str, ParamStats],
    always_neutral: List[Tuple[str, float]],
    never_neutral: List[str],
    co_neutral: List[Tuple[str, str, int]],
    convergence_patterns: Dict[str, Dict],
    conditional_rules: Dict[str, List[Dict]],
    dependency_graph: Dict[str, List[str]],
    recommendations: List[str]
):
    """Export analysis results to JSON for further processing."""
    analysis = {
        "summary": {
            "total_snapshots": len(data['snapshots']),
            "total_parameters": len(param_stats),
            "analysis_timestamp": __import__('datetime').datetime.now().isoformat()
        },
        "parameter_stats": {
            name: {
                "neutrality_rate": stats.neutrality_rate,
                "convergence_rate": stats.convergence_rate,
                "times_neutral": stats.times_neutral,
                "times_converged": stats.times_converged,
                "times_seen": stats.times_seen
            }
            for name, stats in param_stats.items()
        },
        "always_neutral_params": [
            {"name": name, "neutrality_rate": rate}
            for name, rate in always_neutral
        ],
        "never_neutral_params": never_neutral,
        "co_neutral_pairs": [
            {"param1": p1, "param2": p2, "count": count}
            for p1, p2, count in co_neutral
        ],
        "convergence_patterns": convergence_patterns,
        "conditional_rules": conditional_rules,
        "dependency_graph": dependency_graph,
        "recommendations": recommendations
    }
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n📄 Analysis exported to: {output_path}")


def main():
    """Main analysis entry point."""
    # Find the snapshots file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    snapshots_path = os.path.join(script_dir, 'neutrality_snapshots.json')
    
    if not os.path.exists(snapshots_path):
        print(f"❌ Snapshots file not found: {snapshots_path}")
        return
    
    print(f"📂 Loading snapshots from: {snapshots_path}")
    data = load_snapshots(snapshots_path)
    snapshots = data.get('snapshots', [])
    
    if not snapshots:
        print("❌ No snapshots found in file")
        return
    
    print(f"✅ Loaded {len(snapshots)} snapshots")
    
    # Run analyses
    print("\n🔍 Analyzing parameter frequencies...")
    param_stats = analyze_parameter_frequencies(snapshots)
    
    print("🔍 Finding always/never neutral parameters...")
    always_neutral = find_always_neutral_params(param_stats, threshold=0.5)
    never_neutral = find_never_neutral_params(param_stats)
    
    print("🔍 Finding co-neutral parameters...")
    co_neutral = find_co_neutral_params(snapshots, min_co_occurrence=3)
    
    print("🔍 Analyzing convergence patterns...")
    convergence_patterns = analyze_convergence_patterns(param_stats)
    
    print("🔍 Finding conditional neutrality rules...")
    conditional_rules = find_conditional_neutrality(snapshots, param_stats, min_occurrences=3)
    
    print("🔍 Building dependency graph...")
    dependency_graph = build_dependency_graph(snapshots, param_stats)
    
    print("🔍 Generating recommendations...")
    recommendations = generate_recommendations(
        param_stats, always_neutral, never_neutral,
        convergence_patterns, conditional_rules
    )
    
    # Print report
    print_report(
        data, param_stats, always_neutral, never_neutral,
        co_neutral, convergence_patterns, conditional_rules,
        dependency_graph, recommendations
    )
    
    # Export to JSON
    analysis_output_path = os.path.join(script_dir, 'neutrality_analysis.json')
    export_analysis_json(
        analysis_output_path, data, param_stats, always_neutral,
        never_neutral, co_neutral, convergence_patterns,
        conditional_rules, dependency_graph, recommendations
    )


if __name__ == '__main__':
    main()
