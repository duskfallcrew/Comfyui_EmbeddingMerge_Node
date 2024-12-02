import torch
import os
import folder_paths
from safetensors.torch import save_file, load_file
import re
from typing import Union, List, Tuple

class EmbeddingMergerAdvanced:
    """
    The heart of our embedding operations - handles both text and embeddings
    just like the original A1111 extension
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_path = folder_paths.get_folder_paths("embeddings")[0]
        
    def encode_tokens_to_embeddings(self, text: str, model) -> torch.Tensor:
        """Convert text directly to embeddings using the model's CLIP tokenizer"""
        tokens = model.tokenize([text])[0]
        token_embedding = model.clip.wrapped.text_model.embeddings.token_embedding.wrapped
        return token_embedding(tokens.to(self.device))

    def parse_merge_expression(self, expression: str) -> List[Tuple[str, str, float]]:
        """
        Parse expressions like:
        <'word1' + 'word2'*0.5>
        {'style1' + 'style2'/2}
        """
        # Strip the outer brackets if present
        expression = expression.strip()
        if expression.startswith(('<', '{')):
            expression = expression[1:-1]
        if expression.startswith("'"):
            expression = expression[1:]
            
        operations = []
        # Match our various syntax patterns
        pattern = r"""
            (?:
                '([^']+)'                  # Quoted text
                (?:
                    \s*([+\-*/])\s*        # Operators
                    (?:
                        (\d*\.?\d*)        # Numbers
                        |
                        '([^']+)'          # Or another quoted text
                    )
                )?
                |
                ([^+\-*/'\s][^+\-*/']*)   # Unquoted text
            )
        """
        
        matches = re.finditer(pattern, expression, re.VERBOSE)
        last_op = '+'
        
        for match in matches:
            quoted, op, number, next_text, unquoted = match.groups()
            text = quoted if quoted else unquoted
            if text:
                operations.append((text.strip(), last_op, 1.0))
            if op:
                last_op = op
            if number:
                operations[-1] = (operations[-1][0], operations[-1][1], float(number))
            if next_text:
                operations.append((next_text.strip(), last_op, 1.0))
                
        return operations

    def merge_embeddings(self, vectors: List[Tuple[torch.Tensor, str, float]]) -> torch.Tensor:
        """Merge embeddings according to their operations"""
        result = None
        
        for tensor, op, value in vectors:
            if result is None:
                result = tensor * value if op in ['*', '/'] else tensor
                continue
                
            if op == '+':
                result = result + (tensor * value)
            elif op == '-':
                result = result - (tensor * value)
            elif op == '*':
                result = result * value
            elif op == '/':
                result = result / value
                
        return result

    def load_embedding(self, name: str) -> Union[torch.Tensor, None]:
        """Load embedding files with support for both .pt and .safetensors"""
        try:
            path = os.path.join(self.embedding_path, name)
            if not os.path.exists(path):
                for ext in ['.safetensors', '.pt']:
                    test_path = path + ext
                    if os.path.exists(test_path):
                        path = test_path
                        break
            
            if path.endswith('.pt'):
                data = torch.load(path, map_location=self.device)
                if 'string_to_param' in data:
                    return list(data['string_to_param'].values())[0]
                elif 'emb_params' in data:
                    return data['emb_params']
            else:
                data = load_file(path, device=self.device)
                if 'emb_params' in data:
                    return data['emb_params']
                # Handle SDXL style
                if 'clip_l' in data or 'clip_g' in data:
                    return [data.get('clip_l'), data.get('clip_g')]
        except Exception as e:
            print(f"Failed to load embedding {name}: {e}")
        return None

class MergerNode:
    """The ComfyUI node that brings it all together"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {
                    "default": "<'1girl, flowers' + 'detailed'*0.5>", 
                    "multiline": True
                }),
                "model": ("MODEL",),
                "save_as": ("STRING", {
                    "default": "merged_result",
                    "multiline": False
                })
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "merge"
    CATEGORY = "conditioning"
    
    def merge(self, expression: str, model, save_as: str = ""):
        merger = EmbeddingMergerAdvanced()
        operations = merger.parse_merge_expression(expression)
        tensors = []
        
        # Process each part of the expression
        for text, op, value in operations:
            # First try as embedding
            embedding = merger.load_embedding(text)
            if embedding is None:
                # If not an embedding, convert text to embeddings
                embedding = merger.encode_tokens_to_embeddings(text, model)
            tensors.append((embedding, op, value))
        
        # Merge everything
        result = merger.merge_embeddings(tensors)
        
        # Save if requested
        if save_as:
            path = os.path.join(merger.embedding_path, f"{save_as}.safetensors")
            save_file({'emb_params': result.cpu()}, path)
        
        # Return as conditioning for ComfyUI workflow
        return ({"pooled": result, "embeds": result},)

# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "EmbeddingMerger": MergerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmbeddingMerger": "🎨 Embedding Merger (A1111 Style)"
}
