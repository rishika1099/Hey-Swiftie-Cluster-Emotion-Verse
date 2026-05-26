"""
LETTER GENERATOR MODULE - OPENAI VERSION
=========================================
This module generates Taylor Swift-style lyrical letters using OpenAI API.
"""

import os

class LetterGenerator:
    """
    Generates personalized letters using OpenAI API (GPT-3.5-turbo or GPT-4).
    
    BEST QUALITY:
    - No download needed
    - Fastest generation (~1 second)
    - Best quality letters
    - Very affordable (~$0.0001 per letter)
    """
    
    def __init__(self, model_name='gpt-4o-mini', api_key=None):
        """
        Initialize the letter generator with OpenAI API.
        
        Args:
            model_name (str): OpenAI model to use
                - 'gpt-4o-mini': Best value (fast, cheap, great quality)
                - 'gpt-3.5-turbo': Cheaper, still good
                - 'gpt-4o': Best quality (more expensive)
            api_key (str): Your OpenAI API key
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not installed!\n"
                "Install with: pip install openai"
            )
        
        # Get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "\n" + "="*60 + "\n"
                "❌ OPENAI API KEY NOT FOUND!\n"
                "="*60 + "\n"
                "Please provide your API key in one of two ways:\n\n"
                "Option 1 - Pass it directly (app.py line ~43):\n"
                "  letter_generator = LetterGenerator('gpt-4o-mini', api_key='YOUR_KEY_HERE')\n\n"
                "Option 2 - Set environment variable:\n"
                "  Windows: set OPENAI_API_KEY=YOUR_KEY_HERE\n"
                "  Mac/Linux: export OPENAI_API_KEY=YOUR_KEY_HERE\n\n"
                "Get your API key at:\n"
                "  https://platform.openai.com/api-keys\n"
                "="*60
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        
        print(f"✓ OpenAI API configured successfully!")
        print(f"  Model: {model_name}")
        print(f"  Cost: ~$0.0001 per letter (very affordable!)")
    
    def _create_prompt(self, diary_entry, cluster_info, theme_info):
        """Create a prompt for OpenAI that captures Taylor Swift's lyrical style."""
        top_emotion = cluster_info.get('top_emotion', 'neutral')
        cluster_label = cluster_info.get('cluster_label', 'mixed emotions')
        
        system_prompt = """You are writing a short poem in the style of Taylor Swift's song lyrics — for a friend who just shared something personal.

Taylor's lyrical signatures:
- Vivid, specific imagery (storms, stars, seasons, cities at night, cardigans, scarlet letters)
- Time stamps (3am, golden hour, the last great American dynasty)
- Vulnerability stitched into hope
- Concrete details that feel both intimate and universal
- Soft internal rhyme — never forced, never AABB sing-song
- Transforming pain into beauty

FORMAT — this is strict:
- Write 4–6 short stanzas, each 2–4 lines long
- ONE thought per line, with line breaks doing the emotional work
- A blank line between stanzas
- No "Dear Friend" salutation, no signoff, no prose paragraphs
- No markdown, no asterisks, no bullets
- Title case is fine but don't add a title
- Total length: roughly 80–160 words

Think "verse from a song" not "letter". The reader should be able to read it like lyrics."""

        user_prompt = f"""A friend shared this with me:

"{diary_entry[:400]}"

Emotional context:
- Feeling: {top_emotion}
- Theme: {cluster_label}

Write a Taylor-Swift-style lyric poem responding to them. Stanzas, line breaks, imagery. No salutation, no signature — just the verse."""

        return system_prompt, user_prompt
    
    def generate_letter(self, diary_entry, cluster_info, theme_info, 
                       max_length=250, temperature=0.7, top_p=0.9):
        """
        Generate a letter using OpenAI API.
        
        Args:
            diary_entry (str): User's diary entry
            cluster_info (dict): Cluster information
            theme_info (dict): Theme information
            max_length (int): Max tokens (OpenAI setting)
            temperature (float): Creativity level (0.0-2.0, default 0.7)
            top_p (float): Nucleus sampling
            
        Returns:
            str: Generated letter
        """
        print("\n=== Generating Letter with OpenAI ===")
        print(f"Theme: {theme_info['name']}")
        print(f"Emotional cluster: {cluster_info['cluster_label']}")
        
        try:
            # Create prompts
            system_prompt, user_prompt = self._create_prompt(diary_entry, cluster_info, theme_info)
            
            print("🔍 DEBUG: Calling OpenAI API...")
            print(f"   API Key starts with: {self.api_key[:10]}...")
            print(f"   Model: {self.model_name}")
            
            # Generate with OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.9,  # Higher for more creative, lyrical output
                top_p=0.95,
                frequency_penalty=0.5,  # Encourage varied word choice
                presence_penalty=0.4    # Encourage new topics/imagery
            )
            
            # Extract text
            letter = response.choices[0].message.content.strip()
            
            print("✓ SUCCESS: Got response from OpenAI!")
            
            # Post-process
            letter = self._post_process_letter(letter)
            
            print("✓ Letter generated successfully!")
            
            return letter
            
        except Exception as e:
            print(f"\n❌ ERROR: OpenAI API failed!")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"   Falling back to template...\n")
            return self._generate_fallback_letter(diary_entry, cluster_info, theme_info)
    
    def _post_process_letter(self, letter):
        """Clean up the generated poem."""
        letter = letter.replace('**', '').replace('*', '')

        # Strip stray salutations/signoffs the model sometimes adds despite the prompt
        lines = letter.splitlines()
        while lines and lines[0].strip().lower().startswith(('dear ', 'to ', 'hey ', 'hi ')):
            lines.pop(0)
        while lines and lines[-1].strip().lower().startswith(
            ('with ', 'yours', 'your friend', 'love,', '— ', '-- ', 'sincerely')
        ):
            lines.pop()

        # Collapse 3+ blank lines to 2
        cleaned = "\n".join(lines)
        while '\n\n\n' in cleaned:
            cleaned = cleaned.replace('\n\n\n', '\n\n')

        return cleaned.strip()
    
    def _generate_fallback_letter(self, diary_entry, cluster_info, theme_info):
        """Generate a template-based letter if API fails."""
        emotion = cluster_info.get('top_emotion', 'neutral')
        
        templates = {
            'sadness': """The weight you're carrying tonight
is heavier than rain on cotton sheets,
heavier than a name you don't say out loud.

Healing isn't a straight line —
it's a back road in October,
slow, gold, full of turns.

You're allowed to feel it.
That's how I know you loved it.
This chapter is hard, but it isn't the last.""",

            'joy': """There's a kind of light that finds you
on the days when everything clicks,
when even traffic sounds like music.

Hold this — pocket it like a polaroid,
soft-edged and a little overexposed,
proof that the world can be kind.

You are the sparkle and the spark.
Remember this on the long nights.""",

            'fear': """Fear is just the room before the door.
It hums, it flickers,
it asks if you're sure.

You've been brave before —
in smaller rooms,
on bigger nights.

Courage isn't quiet.
It's the heartbeat you walk in with.""",

            'anger': """There's a fire in the way you said it,
red as a dress in a doorway,
sharp as a key turning twice.

You're allowed to burn the things
that asked too much of you.
You're allowed to take the match back.

This isn't broken —
this is the part where you choose.""",
        }

        return templates.get(emotion, templates['sadness'])
    
    def generate_multiple_options(self, diary_entry, cluster_info, theme_info, num_options=3):
        """Generate multiple letter options."""
        print(f"\n=== Generating {num_options} Letter Options ===")
        
        letters = []
        for i in range(num_options):
            print(f"\nGenerating option {i+1}/{num_options}...")
            
            # Vary temperature for diversity
            temp = 0.7 + (i * 0.15)
            
            letter = self.generate_letter(
                diary_entry,
                cluster_info,
                theme_info,
                temperature=temp
            )
            
            letters.append(letter)
        
        return letters
