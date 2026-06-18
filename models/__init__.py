"""ASEF-Net Models"""
from .sef_net_v4 import SEFNet
from .asef_net_paper_version import ASEFNet

# For backward compatibility, ASEFNet is also available as SEFNet
__all__ = ['SEFNet', 'ASEFNet']
