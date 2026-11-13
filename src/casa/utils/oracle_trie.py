import torch
from typing import Optional, Dict


class TrieNode:
    """Node in a trie structure for tracking token sequences.
    
    Attributes:
        parent: Parent node in the trie.
        children: Dictionary mapping token IDs to child nodes.
        raw_logprob: Raw log probabilities from the model for this node.
        log_theta: Adjusted log probabilities after applying constraints.
        leaf_gap: MARS only. Log acceptance still owed at this node's end-of-sequence leaf.
        log_survival: MARS only. Log survival probability at the node's first expansion.
        prepared: MARS only. Cached cumulative draw table (cached rejection sampling).
    """
    
    def __init__(self, parent: Optional['TrieNode'] = None):
        self.parent = parent
        self.children: Dict[int, 'TrieNode'] = {}
        self.raw_logprob: Optional[torch.Tensor] = None
        self.log_theta: Optional[torch.Tensor] = None
        # Set by MARS (casa.samplers.mars); CARS leaves them untouched.
        self.leaf_gap: Optional[float] = None
        self.log_survival: Optional[float] = None
        self.prepared = None
    
    def create_child(self, token_id: int) -> 'TrieNode':
        """Create a child node for the given token ID.
        
        Args:
            token_id: The token ID for the new child node.
            
        Returns:
            The newly created child node.
            
        Raises:
            AssertionError: If a child with this token_id already exists.
        """
        assert token_id not in self.children, f"Child node for token {token_id} already exists"
        self.children[token_id] = TrieNode(parent=self)
        return self.children[token_id]


class Trie:
    """Trie structure for efficient oracle-based sampling."""
    
    def __init__(self):
        self.root = TrieNode()