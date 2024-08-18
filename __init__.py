# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
# 02110-1301  USA

import torch
import comfy
from comfy.ldm.modules.attention import (
    attention_sub_quad,
    attention_pytorch,
    attention_split,
    attention_xformers,
)
from . import fused_attention

def attention_triton(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )    

    dtype = q.dtype
    if dtype != torch.float16:
        q, k, v = map(
            lambda t: t.to(torch.float16),
            (q, k, v),
        )    

    out = fused_attention.attention(q, k, v, False, dim_head ** -0.5)

    out = (
        out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    )

    return out.to(dtype)

class AttnSelectorWithTriton:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        available_attns = []
        available_attns.append("triton")
        available_attns.append("xformers")
        available_attns.append("pytorch")
        available_attns.append("split")
        available_attns.append("sub-quad")

        return {
            "required": {
                "attention": (available_attns,),
                "Model": ("MODEL", )
            },
        }

    RETURN_TYPES = ("MODEL", )

    FUNCTION = "test"
    OUTPUT_NODE = True

    CATEGORY = "_for_testing"

    def test(self, attention, Model):
        print("Select optimized attention:", attention)
        if attention == "xformers":
            attention_algorithm = attention_xformers
        elif attention == "pytorch":
            attention_algorithm = attention_pytorch
        elif attention == "split":
            attention_algorithm = attention_split
        elif attention == "sub-quad":
            attention_algorithm = attention_sub_quad
        elif attention == "triton":
            attention_algorithm = attention_triton

        comfy.ldm.flux.math.optimized_attention = attention_algorithm 
        
        return (Model, attention)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"AttnSelectorWithTriton": AttnSelectorWithTriton}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"AttnSelectorWithTriton": "Attention selector (with triton)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

