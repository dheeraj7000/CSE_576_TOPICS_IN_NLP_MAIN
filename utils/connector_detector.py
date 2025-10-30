#!/usr/bin/env python3
"""
connector_detector.py - CRD CORE: Discourse Connector Detection & Tagging

Comprehensive connector detection with:
- Regex pattern matching from config
- Disambiguation for polysemous connectors  
- EXACT TAG FORMAT: <connector type="X"> word </connector>
- Type classification (UPPERCASE) for attention weighting
- Optional spaCy enhancement (quiet import, no console spam)

CORE TO CRD IDEA:
- Detects 6 connector types across 4 domains
- Maintains exact tag format throughout
- Returns type for model to learn relative importance
- No architecture changes, just precise detection + tagging

Compatible with: config.py, preprocess.py, discourse_training.py
"""

import re
from typing import Dict, List, Tuple, Optional
import warnings
import spacy

from config import CONNECTOR_PATTERNS, TAG_FORMAT

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATTERN COMPILATION
# ============================================================================

# Compile patterns for efficiency
COMPILED_PATTERNS = {
    category: re.compile('|'.join(patterns), re.IGNORECASE)
    for category, patterns in CONNECTOR_PATTERNS.items()
}


# ============================================================================
# SPACY ENHANCEMENT (OPTIONAL, QUIET)
# ============================================================================

SPACY_AVAILABLE = False
nlp = None

try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    # spaCy not available - OK, we fall back to rule-based
    SPACY_AVAILABLE = False


# ============================================================================
# DISAMBIGUATION FOR POLYSEMOUS CONNECTORS
# ============================================================================

def disambiguate_connector(
    word: str,
    sentence: str,
    position: int
) -> Tuple[Optional[str], float]:
    """
    Disambiguate polysemous connectors using context rules.
    
    IMPORTANT FOR CRD: Different types of connectors have different
    reasoning importance for the model to learn.
    
    Args:
        word: The connector word
        sentence: Full sentence containing connector
        position: Character position in sentence
    
    Returns:
        Tuple of (category, confidence) or (None, 0.0)
    """
    
    word_lower = word.lower()
    
    # RULE 1: "since" - temporal vs causal (CRITICAL FOR REASONING)
    if word_lower == "since":
        temporal_pattern = r'since\s+(\d{4}|last|this|next|then|that time|early|late)'
        if re.search(temporal_pattern, sentence[position:position+40], re.IGNORECASE):
            return ('temporal', 0.9)
        else:
            # In academic text, usually causal
            return ('causal', 0.85)
    
    # RULE 2: "while" - temporal vs adversative (REASONING DISTINCTION)
    if word_lower == "while":
        preceding = sentence[max(0, position-10):position]
        if ',' in preceding:
            # Comma suggests contrast/adversative
            return ('adversative', 0.7)
        else:
            # Otherwise temporal
            return ('temporal', 0.65)
    
    # RULE 3: "as" - causal vs temporal (COMMON IN REASONING)
    if word_lower == "as":
        causal_indicators = ['shown', 'demonstrated', 'indicated', 'suggested',
                            'mentioned', 'discussed', 'noted', 'seen', 'evidenced']
        following = sentence[position:position+60].lower()
        if any(ind in following for ind in causal_indicators):
            return ('causal', 0.85)
        return ('causal', 0.6)
    
    # RULE 4: "then" - temporal vs conditional/conclusive (KEY FOR LOGIC)
    if word_lower == "then":
        preceding = sentence[max(0, position-30):position].lower()
        
        # Conditional: "if...then"
        if 'if' in preceding:
            return ('conditional', 0.85)
        
        # Conclusive: after comma or conclusion words
        if ',' in sentence[max(0, position-5):position]:
            return ('conclusive', 0.8)
        
        # Temporal: sequence context
        sequence_words = ['first', 'second', 'next', 'after', 'before']
        if any(word in preceding for word in sequence_words):
            return ('temporal', 0.85)
        
        return ('temporal', 0.6)
    
    # RULE 5: "for" - distinguish reason vs prepositional
    if word_lower == "for":
        following = sentence[position:position+20].lower()
        
        # Skip if followed by article + noun (prepositional, not connector)
        if re.match(r'for\s+(the|a|an)\s+\w+', following):
            if 'purpose' in following or 'aim' in following or 'reason' in following:
                return ('causal', 0.9)
            else:
                return (None, 0.0)  # Not a connector
        
        return ('causal', 0.7)
    
    return (None, 0.0)


def enhanced_disambiguate_with_spacy(
    word: str,
    sentence: str,
    position: int
) -> Tuple[Optional[str], float]:
    """
    Enhanced disambiguation using spaCy POS tagging.
    
    Falls back to rule-based if spaCy unavailable.
    """
    
    if not SPACY_AVAILABLE or nlp is None:
        return disambiguate_connector(word, sentence, position)
    
    word_lower = word.lower()
    
    # Use spaCy for "while" disambiguation
    if word_lower == "while":
        try:
            window_start = max(0, position - 50)
            window_end = min(len(sentence), position + 50)
            window = sentence[window_start:window_end]
            doc = nlp(window)
            
            connector_idx = -1
            for i, token in enumerate(doc):
                if token.text.lower() == "while":
                    connector_idx = i
                    break
            
            if connector_idx != -1 and connector_idx < len(doc) - 1:
                next_token = doc[connector_idx + 1]
                
                # Temporal: while [VERB]
                if next_token.pos_ in ['VERB', 'AUX']:
                    return ('temporal', 0.8)
                # Adversative: while [NOUN/ADJ]
                else:
                    return ('adversative', 0.75)
        except Exception:
            pass
    
    # Fall back to basic disambiguation
    return disambiguate_connector(word, sentence, position)


# ============================================================================
# CORE DETECTION FUNCTION
# ============================================================================

def detect_connectors(text: str) -> List[Dict]:
    """
    Detect ALL connectors in text using regex patterns.
    
    CRD CORE: Returns connector info needed for:
    - Tagging with exact format
    - Building attention masks
    - Tracking type for model learning
    
    Args:
        text: Input text to analyze
    
    Returns:
        List of dicts with {start, end, word, category, confidence}
    """
    
    matches = []
    
    for category, pattern in COMPILED_PATTERNS.items():
        for match in pattern.finditer(text):
            word = match.group(0)
            start, end = match.span()
            
            # Initial category from regex
            initial_category = category
            confidence = 1.0
            
            # Disambiguate polysemous words (critical for accuracy)
            polysemous = ['since', 'while', 'as', 'then', 'for']
            if word.lower() in polysemous:
                if SPACY_AVAILABLE:
                    disambig_cat, disambig_conf = enhanced_disambiguate_with_spacy(word, text, start)
                else:
                    disambig_cat, disambig_conf = disambiguate_connector(word, text, start)
                
                if disambig_cat:
                    initial_category = disambig_cat
                    confidence = disambig_conf
            
            matches.append({
                'start': start,
                'end': end,
                'word': word,
                'category': initial_category,
                'confidence': confidence
            })
    
    # Sort by position and remove overlaps (keep highest confidence)
    matches.sort(key=lambda x: (x['start'], -x['confidence']))
    
    unique_matches = []
    last_end = -1
    for match in matches:
        if match['start'] >= last_end:
            unique_matches.append(match)
            last_end = match['end']
    
    return unique_matches


def detect_connectors_in_text(text: str, detector=None) -> List[Dict]:
    """
    Detect connectors with uppercase types for tag compatibility.
    
    Returns type in UPPERCASE for exact tag format.
    
    Args:
        text: Input text
        detector: Optional ConnectorDetector instance
    
    Returns:
        List of dicts with {start, end, word, type, confidence}
    """
    
    matches = detect_connectors(text)
    
    # Convert to new format with uppercase type (for tagging)
    return [
        {
            'start': m['start'],
            'end': m['end'],
            'word': m['word'],
            'type': m['category'].upper(),  # UPPERCASE for tags
            'confidence': m['confidence']
        }
        for m in matches
    ]


# ============================================================================
# TEXT TAGGING - EXACT FORMAT
# ============================================================================

def tag_text(text: str) -> Dict:
    """
    Tag text with connector markers in EXACT format.
    
    Format: <connector type="X"> word </connector>
    
    This is the CORE of CRD preprocessing. Tags enable:
    1. Exact identification of connectors during tokenization
    2. Building of connector_mask for attention weighting
    3. Type information for model to learn relative importance
    
    Args:
        text: Raw input text
    
    Returns:
        Dict with:
        - tagged_text: Text with exact format tags
        - connector_positions: List of character positions
        - connector_types: List of category names (UPPERCASE)
        - connector_words: List of actual connector strings
        - connector_confidences: List of confidence scores
    """
    
    matches = detect_connectors(text)
    
    # Apply tags in reverse order to preserve positions
    tagged = text
    matches_reversed = list(reversed(matches))
    
    for match in matches_reversed:
        category = match['category'].upper()  # UPPERCASE for exact format
        word = match['word']
        start = match['start']
        end = match['end']
        
        # EXACT FORMAT: <connector type="X"> word </connector>
        tag_open = f'<connector type="{category}">'
        tag_close = '</connector>'
        replacement = f"{tag_open} {word} {tag_close}"
        
        tagged = tagged[:start] + replacement + tagged[end:]
    
    return {
        'tagged_text': tagged,
        'connector_positions': [m['start'] for m in matches],
        'connector_types': [m['category'].upper() for m in matches],
        'connector_words': [m['word'] for m in matches],
        'connector_confidences': [m['confidence'] for m in matches]
    }


# ============================================================================
# CLASS-BASED INTERFACE
# ============================================================================

class ConnectorDetector:
    """
    Class-based interface for connector detection.
    
    CRD Interface: Provides consistent connector detection
    across preprocessing and training pipelines.
    """
    
    def __init__(self, patterns: Dict[str, List[str]] = None):
        """
        Initialize detector with patterns.
        
        Args:
            patterns: Dict of patterns (uses config.CONNECTOR_PATTERNS if None)
        """
        
        if patterns is None:
            patterns = CONNECTOR_PATTERNS
        
        self.patterns = patterns
        self.connector_patterns = {
            category: re.compile('|'.join(pattern_list), re.IGNORECASE)
            for category, pattern_list in patterns.items()
        }
    
    def detect_in_text(self, text: str) -> List[Dict]:
        """Detect connectors in text."""
        return detect_connectors_in_text(text, self)
    
    def get_connector_types(self) -> List[str]:
        """Get list of connector types (UPPERCASE)."""
        return [t.upper() for t in self.patterns.keys()]
    
    def get_pattern_count(self) -> Dict[str, int]:
        """Get pattern count per type."""
        return {
            conn_type: len(pattern_list)
            for conn_type, pattern_list in self.patterns.items()
        }


# ============================================================================
# VERIFICATION & TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONNECTOR DETECTOR - CRD CORE")
    print("="*80)
    
    # Show status
    print(f"\nEnhancement Status:")
    if SPACY_AVAILABLE:
        print(" ✓ spaCy disambiguation ENABLED (enhanced accuracy)")
    else:
        print(" ℹ spaCy not available - using rule-based disambiguation")
    
    # Test text with reasoning
    test_text = """
The model works because of good data. However, it needs improvement.
While training, we monitor loss. Since 2020, progress has been made.
Therefore, we continue. As shown before, results are promising.
If performance improves, then we scale up.
"""
    
    print(f"\n[Test Text]:\n{test_text}")
    
    # Test 1: Direct detection
    print("\n[Detection Results]:")
    connectors = detect_connectors_in_text(test_text)
    print(f"Found {len(connectors)} connectors:")
    for conn in connectors:
        print(f"  '{conn['word']}' → {conn['type']} (confidence: {conn['confidence']:.2f})")
    
    # Test 2: Tagging
    print("\n[Exact Format Tagging]:")
    result = tag_text(test_text)
    print("Tagged text (first 300 chars):")
    print(result['tagged_text'][:300] + "...")
    
    # Test 3: Class interface
    print("\n[Class-based Interface]:")
    detector = ConnectorDetector()
    print(f"Connector types: {detector.get_connector_types()}")
    print(f"Total patterns: {sum(detector.get_pattern_count().values())}")
    
    print("\n" + "="*80 + "\n")