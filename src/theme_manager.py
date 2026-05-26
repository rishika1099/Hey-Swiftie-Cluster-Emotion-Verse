"""
THEME MANAGER MODULE
====================
This module manages visual themes based on Taylor Swift albums.

KEY CONCEPTS:
- Dynamic theming: Changing website appearance based on emotional content
- Color psychology: Using colors that match emotional tones
- Album aesthetics: Each era has a distinct visual identity
"""

import json
from pathlib import Path

class ThemeManager:
    """
    Manages album themes for dynamic UI changes.
    
    CONCEPT: Visual Emotional Resonance
    - Colors and aesthetics affect how we feel
    - Matching visual theme to emotional content enhances the experience
    - Each Taylor Swift album has a distinct visual identity we can leverage
    """
    
    def __init__(self):
        """
        Initialize theme manager with album theme configurations.
        """
        # Define themes for each album/era
        # CONCEPT: Theme Configuration
        # Each theme includes colors, fonts, and CSS properties
        # Based on actual album aesthetics and color palettes
        self.themes = {
            'lover': {
                'name': 'Lover',
                'primary_color': '#ffb3d9',
                'secondary_color': '#b8e6f5',
                'accent_color': '#ff69b4',
                'background': 'linear-gradient(135deg, #ffc0e5 0%, #b8e6f5 100%)',
                'text_color': '#4a0e4e',
                'font_family': "'Playfair Display', serif",
                'border_style': '3px solid #ffb3d9',
                'border_radius': '20px',
                'vibe': 'romantic, dreamy, pastel, soft',
                'emotions': ['joy', 'love', 'romantic'],
                'description': 'Pastel pinks and blues, romantic and dreamy'
            },
            
            'reputation': {
                'name': 'reputation',
                'primary_color': '#000000',
                'secondary_color': '#2d2d2d',
                'accent_color': '#00ff00',  # Snake green
                'background': 'linear-gradient(135deg, #1a1a1a 0%, #000000 100%)',
                'text_color': '#ffffff',
                'font_family': "'Bebas Neue', sans-serif",
                'border_style': '2px solid #00ff00',
                'border_radius': '5px',
                'vibe': 'dark, edgy, bold, powerful',
                'emotions': ['anger', 'empowerment', 'confidence'],
                'description': 'Black with neon accents, edgy and bold'
            },
            
            '1989': {
                'name': '1989',
                'primary_color': '#87ceeb',
                'secondary_color': '#ff69b4',
                'accent_color': '#ffd700',
                'background': 'linear-gradient(135deg, #87ceeb 0%, #ffb6c1 100%)',
                'text_color': '#2c3e50',
                'font_family': "'Poppins', sans-serif",
                'border_style': '3px solid #ff69b4',
                'border_radius': '15px',
                'vibe': 'upbeat, pop, colorful, fun',
                'emotions': ['joy', 'excitement', 'fun'],
                'description': 'Bright blues and pinks, pop and energetic'
            },
            
            'folklore': {
                'name': 'folklore',
                'primary_color': '#8b7d6b',
                'secondary_color': '#d4c5b9',
                'accent_color': '#5f4b3b',
                'background': 'linear-gradient(135deg, #d4c5b9 0%, #a89f91 100%)',
                'text_color': '#3d3226',
                'font_family': "'Crimson Text', serif",
                'border_style': '2px solid #8b7d6b',
                'border_radius': '10px',
                'vibe': 'cottagecore, nostalgic, muted, introspective',
                'emotions': ['nostalgia', 'melancholy', 'introspection'],
                'description': 'Muted earth tones, cottagecore and nostalgic'
            },
            
            'evermore': {
                'name': 'evermore',
                'primary_color': '#8b4513',
                'secondary_color': '#daa520',
                'accent_color': '#cd853f',
                'background': 'linear-gradient(135deg, #daa520 0%, #8b4513 100%)',
                'text_color': '#2f1e0f',
                'font_family': "'Crimson Text', serif",
                'border_style': '2px solid #8b4513',
                'border_radius': '10px',
                'vibe': 'autumn, warm, reflective, woodsy',
                'emotions': ['reflection', 'bittersweet', 'acceptance'],
                'description': 'Warm autumn colors, reflective and cozy'
            },
            
            'red': {
                'name': 'Red',
                'primary_color': '#dc143c',
                'secondary_color': '#8b0000',
                'accent_color': '#ffa500',
                'background': 'linear-gradient(135deg, #ff6b6b 0%, #8b0000 100%)',
                'text_color': '#ffffff',
                'font_family': "'Montserrat', sans-serif",
                'border_style': '3px solid #dc143c',
                'border_radius': '12px',
                'vibe': 'passionate, intense, autumn, emotional',
                'emotions': ['passion', 'heartbreak', 'intensity'],
                'description': 'Deep reds and oranges, passionate and intense'
            },
            
            'midnights': {
                'name': 'Midnights',
                'primary_color': '#1e3a5f',
                'secondary_color': '#4a5f8f',
                'accent_color': '#b8c5d6',
                'background': 'linear-gradient(135deg, #0f1c3f 0%, #1e3a5f 100%)',
                'text_color': '#e8f0ff',
                'font_family': "'Raleway', sans-serif",
                'border_style': '2px solid #4a5f8f',
                'border_radius': '15px',
                'vibe': 'late night, dreamy, mysterious, moody',
                'emotions': ['introspection', 'mystery', 'late-night thoughts'],
                'description': 'Deep blues and purples, moody and mysterious'
            },
            
            'fearless': {
                'name': 'Fearless',
                'primary_color': '#ffd700',
                'secondary_color': '#fff8dc',
                'accent_color': '#ffb347',
                'background': 'linear-gradient(135deg, #fff8dc 0%, #ffd700 100%)',
                'text_color': '#5d4e37',
                'font_family': "'Dancing Script', cursive",
                'border_style': '3px solid #ffd700',
                'border_radius': '20px',
                'vibe': 'golden, sparkly, youthful, hopeful',
                'emotions': ['hope', 'joy', 'innocence'],
                'description': 'Golden and sparkly, youthful and hopeful'
            },
            
            'speak_now': {
                'name': 'Speak Now',
                'primary_color': '#9370db',
                'secondary_color': '#dda0dd',
                'accent_color': '#ba55d3',
                'background': 'linear-gradient(135deg, #dda0dd 0%, #9370db 100%)',
                'text_color': '#4b0082',
                'font_family': "'Merriweather', serif",
                'border_style': '3px solid #9370db',
                'border_radius': '18px',
                'vibe': 'whimsical, fairytale, purple, dreamy',
                'emotions': ['wonder', 'storytelling', 'imagination'],
                'description': 'Purple and whimsical, fairy tale aesthetic'
            }
        }
        
        # Map cluster characteristics to albums
        # CONCEPT: Mapping Emotions to Visual Themes
        # We match cluster emotional profiles to appropriate album aesthetics
        self.cluster_to_theme_mapping = {
            'rules': [
                {
                    'condition': 'high valence + high arousal',
                    'themes': ['1989', 'fearless', 'lover'],
                    'description': 'Upbeat and joyful → bright, colorful themes'
                },
                {
                    'condition': 'low valence + low arousal',
                    'themes': ['folklore', 'evermore', 'midnights'],
                    'description': 'Sad and calm → muted, introspective themes'
                },
                {
                    'condition': 'low valence + high arousal',
                    'themes': ['reputation', 'red'],
                    'description': 'Angry/intense → dark, bold themes'
                },
                {
                    'condition': 'high valence + low arousal',
                    'themes': ['lover', 'speak_now'],
                    'description': 'Peaceful happiness → soft, dreamy themes'
                }
            ]
        }
    
    def get_theme_for_cluster(self, cluster_stats):
        """
        Select appropriate theme based on cluster characteristics.
        
        Args:
            cluster_stats (dict): Cluster statistics including valence, arousal, etc.
            
        Returns:
            dict: Theme configuration
            
        CONCEPT: Dynamic Theme Selection
        - We analyze the emotional characteristics
        - Pick a theme that matches the emotional tone
        - This creates visual coherence with emotional content
        """
        valence = cluster_stats.get('avg_valence', 0.5)
        arousal = cluster_stats.get('avg_arousal', 0.5)
        dominance = cluster_stats.get('avg_dominance', 0.5)
        
        # Determine theme based on emotional characteristics
        # CONCEPT: Rule-Based Selection
        # We use if-else rules to map emotions to themes
        
        if valence > 0.6 and arousal > 0.6:
            # High energy, positive → upbeat themes
            theme_options = ['1989', 'fearless', 'lover']
        elif valence < 0.4 and arousal < 0.4:
            # Low energy, negative → introspective themes
            theme_options = ['folklore', 'evermore', 'midnights']
        elif valence < 0.4 and arousal > 0.6:
            # High energy, negative → intense themes
            theme_options = ['reputation', 'red']
        elif dominance > 0.6:
            # Empowered → bold themes
            theme_options = ['reputation', '1989']
        else:
            # Default to balanced themes
            theme_options = ['lover', 'speak_now', 'midnights']
        
        # Pick first option (you can add more sophisticated selection)
        selected_theme = theme_options[0]
        
        return self.themes[selected_theme]
    
    def get_theme_by_name(self, theme_name):
        """
        Get theme configuration by album name.
        
        Args:
            theme_name (str): Album/theme name
            
        Returns:
            dict: Theme configuration or default
        """
        theme_name = theme_name.lower().replace(' ', '_')
        return self.themes.get(theme_name, self.themes['lover'])
    
    def generate_css(self, theme):
        """
        Generate CSS code for a theme.
        
        Args:
            theme (dict): Theme configuration
            
        Returns:
            str: CSS code
            
        CONCEPT: Programmatic CSS Generation
        - Instead of hardcoding CSS, we generate it from theme config
        - This makes themes easy to modify and maintain
        - One source of truth for all styling
        """
        css = f"""
        /* {theme['name']} Theme */
        .gradio-container {{
            background: {theme['background']} !important;
            font-family: {theme['font_family']} !important;
            color: {theme['text_color']} !important;
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {theme['primary_color']} !important;
            font-family: {theme['font_family']} !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        /* Input boxes */
        .input-text textarea, .input-text input {{
            border: {theme['border_style']} !important;
            border-radius: {theme['border_radius']} !important;
            background: rgba(255,255,255,0.9) !important;
            font-family: {theme['font_family']} !important;
        }}
        
        /* Buttons */
        .primary-button {{
            background: {theme['primary_color']} !important;
            color: {theme['text_color']} !important;
            border: none !important;
            border-radius: {theme['border_radius']} !important;
            font-weight: bold;
            padding: 12px 24px;
            font-family: {theme['font_family']} !important;
        }}
        
        .primary-button:hover {{
            background: {theme['secondary_color']} !important;
            transform: scale(1.05);
            transition: all 0.3s ease;
        }}
        
        /* Output boxes */
        .output-markdown {{
            background: rgba(255,255,255,0.85) !important;
            padding: 30px !important;
            border-radius: {theme['border_radius']} !important;
            border: {theme['border_style']} !important;
            font-family: {theme['font_family']} !important;
            color: {theme['text_color']} !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        /* Accent elements */
        .accent {{
            color: {theme['accent_color']} !important;
        }}
        
        /* Smooth transitions */
        * {{
            transition: background 0.5s ease, color 0.3s ease;
        }}
        """
        
        return css
    
    def get_all_themes(self):
        """
        Get all available themes.
        
        Returns:
            dict: All theme configurations
        """
        return self.themes
    
    def save_themes(self, output_path='themes/themes.json'):
        """
        Save theme configurations to JSON file.
        
        Args:
            output_path (str): Path to save JSON file
            
        CONCEPT: Configuration Management
        - Storing themes in JSON makes them easy to edit
        - Non-programmers can modify colors without touching code
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.themes, f, indent=2)
        
        print(f"✓ Saved themes to: {output_path}")


# Example usage
if __name__ == "__main__":
    """
    Test theme manager and generate CSS files.
    """
    
    # Initialize manager
    manager = ThemeManager()
    
    # Show all themes
    print("=== Available Themes ===\n")
    for theme_name, theme in manager.themes.items():
        print(f"{theme['name']}:")
        print(f"  Vibe: {theme['vibe']}")
        print(f"  Description: {theme['description']}")
        print(f"  Colors: {theme['primary_color']}, {theme['secondary_color']}")
        print()
    
    # Generate CSS for each theme
    print("\n=== Generating CSS Files ===")
    themes_dir = Path('themes')
    themes_dir.mkdir(exist_ok=True)
    
    for theme_name, theme in manager.themes.items():
        css = manager.generate_css(theme)
        css_path = themes_dir / f"{theme_name}.css"
        with open(css_path, 'w') as f:
            f.write(css)
        print(f"✓ Generated: {css_path}")
    
    # Save themes configuration
    manager.save_themes()
    
    # Test theme selection
    print("\n=== Testing Theme Selection ===")
    test_cases = [
        {'avg_valence': 0.8, 'avg_arousal': 0.7, 'avg_dominance': 0.6},  # Happy & energetic
        {'avg_valence': 0.2, 'avg_arousal': 0.3, 'avg_dominance': 0.3},  # Sad & calm
        {'avg_valence': 0.3, 'avg_arousal': 0.8, 'avg_dominance': 0.7},  # Angry & intense
    ]
    
    for i, stats in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Valence: {stats['avg_valence']}, Arousal: {stats['avg_arousal']}")
        theme = manager.get_theme_for_cluster(stats)
        print(f"  → Selected Theme: {theme['name']}")
        print(f"  → {theme['description']}")
    
    print("\n✓ Theme manager testing complete!")
