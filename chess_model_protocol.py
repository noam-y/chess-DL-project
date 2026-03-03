from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import Dataset
from enum import Enum

class Piece(Enum):
    EMPTY = 'e'
    WHITE_PAWN = 'P'
    WHITE_KNIGHT = 'N'
    WHITE_BISHOP = 'B'
    WHITE_ROOK = 'R'
    WHITE_QUEEN = 'Q'
    WHITE_KING = 'K'
    BLACK_PAWN = 'p'
    BLACK_KNIGHT = 'n'
    BLACK_BISHOP = 'b'
    BLACK_ROOK = 'r'
    BLACK_QUEEN = 'q'
    BLACK_KING = 'k'
    OOD = 'ood'

class ChessModelProtocol(ABC):
    """
    Protocol for Chess Deep Learning Models.
    """

    # =========================================================================
    # Methods implemented in BaseChessModel (Common Logic)
    # These are typically NOT overridden in specific model files.
    # =========================================================================

    @abstractmethod
    def create_dataset(self, root_dir, mode='train', val_game_name='') -> Dataset:
        """
        Creates and returns the Dataset.
        mode: 'train', 'val', or 'test'
        val_game_name: The name of the game to hold out for validation (e.g., 'game1').
        
        Implemented in BaseChessModel to use BaseChessDataset.
        """
        pass

    # =========================================================================
    # Methods to be implemented in specific Model files (e.g. model_v2.py)
    # These define the unique behavior of each model version.
    # =========================================================================

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Creates and returns the PyTorch model."""
        pass

    @abstractmethod
    def fen_to_labels(self, fen: str) -> tuple:
        """
        Converts a FEN string to a tuple of tensors (8x8) representing targets.
        Used by the BaseChessDataset to prepare labels.
        
        Example V2: returns (labels_tensor,)
        Example V3: returns (occ_tensor, color_tensor, piece_tensor)
        """
        pass

    @abstractmethod
    def compute_loss(self, model, batch, device, criterion=None):
        """
        Computes the loss for a batch.
        Returns: loss (Tensor), metrics (dict)
        """
        pass
    
    @abstractmethod
    def get_optimizer(self, model, lr=0.001):
        """Returns the optimizer."""
        pass
        
    @abstractmethod
    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        """
        Performs inference on a single tile tensor.
        Returns: Piece enum
        """
        pass

    def on_epoch_end(self, model, epoch, optimizer):
        """
        Optional hook called at the end of each epoch.
        Can be used for unfreezing layers, scheduling, etc.
        """
        pass
