import re
import csv
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BertTokenizer

EMOJI_PATH = "data/emojis.txt"


class TweetDataset(Dataset):
    def __init__(
        self,
        file_path,
        tokenizer: PreTrainedTokenizer = None,
        transform=None,
    ):
        TWEET_MAX_LENGTH = 256

        self.features = []
        self.labels = []
        self.transform = transform
        self.tokenizer: PreTrainedTokenizer

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        emojis = self._load_emojis(EMOJI_PATH)
        print(emojis)

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                tweet = self._replace_emojis(row[2], emojis)
                encoding = self.tokenizer(
                    tweet,
                    max_length=TWEET_MAX_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tweet_ids = encoding["input_ids"].squeeze(0)
                attention_mask = encoding["attention_mask"].squeeze(0)

                sentiment = row[3].strip()
                if sentiment not in SENTIMENT_SCORES:
                    raise ValueError(
                        f"Sentiment {sentiment} not found in sentiment scores"
                    )

                self.features.append((tweet_ids, attention_mask))
                self.labels.append(SENTIMENT_SCORES[sentiment])

        # Ensure all features have the same length
        assert all(len(feature) == len(self.features[0]) for feature in self.features)

    def _load_emojis(self, emoji_path):
        emojis = {}
        with open(emoji_path, "r") as f:
            for line in f.readlines():
                # Extract the emoji and description where the line is in the format
                # {code}{\s*};{\s*}{fully-qualified}{\s*}#{\s*}{emoji}{\s*}{description}
                pattern = r"^[A-F0-9\s]+;\s*fully-qualified\s*#\s*(\S+)\s+(.+)$"
                match = re.match(pattern, line.strip())
                if match:
                    emoji, description = match.groups()
                    emojis[emoji] = description.strip()
        return emojis

    def _replace_emojis(self, text, emoji_dict):
        # Unicode ranges for emojis
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U0001f900-\U0001f9ff"  # supplemental symbols
            "\u2600-\u26ff"  # miscellaneous symbols
            "\u2700-\u27bf"  # dingbats
            "]+",
            flags=re.UNICODE,
        )

        def replace_match(match):
            emoji = match.group(0)
            return f"[{emoji_dict.get(emoji, 'emoji')}]"

        return emoji_pattern.sub(replace_match, text)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        (input, mask) = self.features[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            input = self.transform(input)

        return input, mask, label


# LLM assigned sentiment scores where -1.0 is compelte negative and 1.0 is
# complete positive
SENTIMENT_SCORES = {
    # Positive Emotions (0.0 to 1.0)
    "Positive": 1.0,
    "Happiness": 1.0,
    "Joy": 1.0,
    "Love": 1.0,
    "Amusement": 0.9,
    "Enjoyment": 0.9,
    "Admiration": 0.8,
    "Affection": 0.8,
    "Awe": 0.8,
    "Adoration": 0.8,
    "Anticipation": 0.7,
    "Calmness": 0.7,
    "Excitement": 0.8,
    "Kind": 0.7,
    "Pride": 0.7,
    "Elation": 0.9,
    "Euphoria": 1.0,
    "Contentment": 0.8,
    "Serenity": 0.7,
    "Gratitude": 0.8,
    "Hope": 0.7,
    "Empowerment": 0.8,
    "Compassion": 0.7,
    "Tenderness": 0.7,
    "Arousal": 0.6,
    "Enthusiasm": 0.8,
    "Fulfillment": 0.9,
    "Reverence": 0.7,
    "Curiosity": 0.6,
    "Determination": 0.7,
    "Zest": 0.8,
    "Hopeful": 0.7,
    "Proud": 0.8,
    "Grateful": 0.8,
    "Empathetic": 0.7,
    "Compassionate": 0.7,
    "Playful": 0.7,
    "Free-spirited": 0.7,
    "Inspired": 0.8,
    "Confident": 0.8,
    "Thrill": 0.8,
    "Bittersweet": 0.5,
    "Overjoyed": 1.0,
    "Inspiration": 0.8,
    "Motivation": 0.7,
    "Contemplation": 0.6,
    "JoyfulReunion": 0.9,
    "Satisfaction": 0.8,
    "Blessed": 0.9,
    "Reflection": 0.6,
    "Appreciation": 0.8,
    "Confidence": 0.8,
    "Accomplishment": 0.9,
    "Wonderment": 0.7,
    "Optimism": 0.8,
    "Enchantment": 0.8,
    "Intrigue": 0.7,
    "PlayfulJoy": 0.8,
    "Mindfulness": 0.7,
    "DreamChaser": 0.7,
    "Elegance": 0.7,
    "Whimsy": 0.7,
    "Pensive": 0.6,
    "Harmony": 0.8,
    "Creativity": 0.7,
    "Radiance": 0.8,
    "Wonder": 0.7,
    "Rejuvenation": 0.8,
    "Coziness": 0.7,
    "Adventure": 0.7,
    "Melodic": 0.7,
    "FestiveJoy": 0.9,
    "InnerJourney": 0.7,
    "Freedom": 0.8,
    "Dazzle": 0.8,
    "Adrenaline": 0.7,
    "ArtisticBurst": 0.8,
    "CulinaryOdyssey": 0.7,
    "Resilience": 0.8,
    "Immersion": 0.7,
    "Spark": 0.8,
    "Marvel": 0.8,
    "Positivity": 1.0,
    "Kindness": 0.8,
    "Friendship": 0.8,
    "Success": 0.9,
    "Exploration": 0.7,
    "Amazement": 0.8,
    "Romance": 0.9,
    "Captivation": 0.8,
    "Tranquility": 0.7,
    "Grandeur": 0.8,
    "Emotion": 0.6,
    "Energy": 0.7,
    "Celebration": 0.9,
    "Charm": 0.8,
    "Ecstasy": 1.0,
    "Colorful": 0.7,
    "Hypnotic": 0.7,
    "Connection": 0.8,
    "Iconic": 0.8,
    "Journey": 0.7,
    "Engagement": 0.7,
    "Touched": 0.7,
    "Suspense": 0.6,
    "Triumph": 0.9,
    "Heartwarming": 0.9,
    "Obstacle": 0.5,
    "Sympathy": 0.6,
    "Pressure": 0.4,
    "Renewed Effort": 0.6,
    "Miscalculation": 0.3,
    "Challenge": 0.5,
    "Solace": 0.6,
    "Breakthrough": 0.8,
    "Joy in Baking": 0.8,
    "Envisioning History": 0.7,
    "Imagination": 0.7,
    "Vibrancy": 0.8,
    "Mesmerizing": 0.8,
    "Culinary Adventure": 0.7,
    "Winter Magic": 0.7,
    "Thrilling Journey": 0.8,
    "Nature's Beauty": 0.8,
    "Celestial Wonder": 0.8,
    "Creative Inspiration": 0.8,
    "Runway Creativity": 0.8,
    "Ocean's Freedom": 0.8,
    "Whispers of the Past": 0.7,
    "Relief": 0.7,
    "Mischievous": 0.5,
    "Happy": 1.0,
    # Neutral Emotions (0.0)
    "Neutral": 0.0,
    "Acceptance": 0.0,
    "Surprise": 0.0,
    "Indifference": 0.0,
    "Numbness": 0.0,
    "Ambivalence": 0.0,
    # Negative Emotions (-1.0 to 0.0)
    "Negative": -1.0,
    "Anger": -0.9,
    "Fear": -0.8,
    "Sadness": -0.8,
    "Disgust": -0.9,
    "Disappointed": -0.7,
    "Bitter": -0.7,
    "Confusion": -0.6,
    "Shame": -0.8,
    "Despair": -0.9,
    "Grief": -0.9,
    "Loneliness": -0.8,
    "Jealousy": -0.7,
    "Resentment": -0.8,
    "Frustration": -0.7,
    "Boredom": -0.6,
    "Anxiety": -0.8,
    "Intimidation": -0.7,
    "Helplessness": -0.8,
    "Envy": -0.7,
    "Regret": -0.8,
    "Melancholy": -0.7,
    "Nostalgia": -0.5,
    "Bitterness": -0.7,
    "Yearning": -0.6,
    "Fearful": -0.8,
    "Apprehensive": -0.7,
    "Overwhelmed": -0.7,
    "Jealous": -0.7,
    "Devastated": -0.9,
    "Frustrated": -0.7,
    "Envious": -0.7,
    "Dismissive": -0.6,
    "Heartbreak": -0.9,
    "Betrayal": -0.9,
    "Suffering": -0.9,
    "EmotionalStorm": -0.9,
    "Isolation": -0.8,
    "Disappointment": -0.7,
    "LostLove": -0.8,
    "Exhaustion": -0.7,
    "Sorrow": -0.8,
    "Darkness": -0.9,
    "Desperation": -0.9,
    "Ruins": -0.9,
    "Desolation": -0.9,
    "Loss": -0.8,
    "Heartache": -0.8,
    "Solitude": -0.7,
    "Embarrassed": -0.6,
    "Sad": -0.8,
    "Hate": -1.0,
    "Bad": -0.8,
}
