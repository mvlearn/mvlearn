import sys

sys.path.append("../")

import pytest
from multiview.embed.base import BaseEmbed

def test_base_embed():
    BaseEmbed()