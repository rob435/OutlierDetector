#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from narrative_analyzer import NarrativeAnalyzer

def display_narrative_timeline():
    """Display detailed timeline of narrative activities"""
    print("=== COMPREHENSIVE NARRATIVE ANALYSIS ===")
    
    # First show configuration breakdown
    from main import get_narrative_groups
    all_groups = get_narrative_groups()
    filtered_groups = {k: v for k, v in all_groups.items() if len(v) >= 2}
    
    print(f"Total configured narratives: {len(all_groups)}")
    print(f"Narratives with 2+ coins (trackable): {len(filtered_groups)}")
    print(f"Single-coin narratives (filtered out): {len(all_groups) - len(filtered_groups)}")
    
    analyzer = NarrativeAnalyzer()
    
    # Get enhanced summary
    summary = analyzer.get_narrative_summary()
    print(f"Actually detected in historical data: {summary.get('detected_narratives_count', 0)}")
    print(f"Historical activities found: {summary.get('historical_activities_count', 0)}")
    
    detected_narratives = summary.get('detected_narratives', [])
    missing_narratives = set(filtered_groups.keys()) - set(detected_narratives)
    if missing_narratives:
        print(f"Never detected (no 5%+ pumps): {', '.join(sorted(missing_narratives))}")
    
    # Show current activity
    current_pumping = summary.get('current_pumping_narrative')
    if current_pumping:
        print(f"\nCurrently pumping: {current_pumping}")
    
    current_activities = summary.get('current_narrative_activities', [])
    if current_activities:
        print(f"\nCurrent narrative activities:")
        for activity in current_activities[:3]:
            print(f"  - {activity['narrative']} (Strength: {activity['narrative_strength']:.1%})")
            print(f"    Coins: {', '.join(activity['coins_in_top'])}")
    
    # Get detailed timeline
    timeline = analyzer.get_historical_narrative_timeline()
    
    if not timeline:
        print("\nNo historical narrative activities detected.")
        return
    
    # Group activities by narrative to show dates
    narrative_dates = {}
    for activity in timeline:
        narrative = activity['narrative']
        if narrative not in narrative_dates:
            narrative_dates[narrative] = []
        narrative_dates[narrative].append(activity['date'])
    
    print(f"\n=== NARRATIVE ACTIVITY DATES ===")
    for narrative in sorted(narrative_dates.keys()):
        dates = sorted(set(narrative_dates[narrative]))  # Remove duplicates and sort
        print(f"{narrative}: {len(dates)} days active")
        print(f"  Dates: {', '.join(dates)}")
        print()
    
    print(f"\n=== DETAILED TIMELINE ({len(timeline)} activities) ===")
    print(f"{'#':<3} {'Date':<10} {'Time':<5} {'Narrative':<20} {'Performance':<12} {'Pos%':<6} {'Coins':<6} {'Top Performers'}")
    print("-" * 120)
    
    for i, activity in enumerate(timeline, 1):
        narrative = activity['narrative'][:18]  # Truncate long names
        performers = ', '.join(activity['top_performing_coins'][:3])  # Show top 3
        remaining = len(activity['all_coins']) - 3
        if remaining > 0:
            performers += f" (+{remaining})"
        
        print(f"{i:<3} {activity['date']:<10} {activity['time']:<5} {narrative:<20} {activity['performance']:<12} {activity['positive_ratio']:<6} {activity['coins_count']:<6} {performers}")

if __name__ == "__main__":
    display_narrative_timeline()